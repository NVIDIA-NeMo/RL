# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Shared helpers for the uv cache mount's materialize/sync-back/strip dance.
# Source this at the top of every RUN that mounts the durable uv cache
# (BuildKit cache mount at /root/.uv-cache-store) and writes to UV_CACHE_DIR
# via uv. See docker/Dockerfile's dependency-install RUN for the full
# explanation of why the dance exists: UV_LINK_MODE=hardlink installs need
# UV_CACHE_DIR on the same filesystem as the venvs, which a cache mount is
# not, so each RUN works against a plain same-filesystem copy instead.

# Materialize a same-filesystem working copy of the durable store so
# UV_LINK_MODE=hardlink installs can hardlink into it. Pass "force" only on
# the first uv-writing RUN in a stage, where UV_CACHE_DIR does not exist yet.
# Every later RUN already inherited it from that layer -- copying the whole
# store in again there would re-ship every store-only wheel that RUN never
# touches, in that RUN's own layer. The plain `[ ! -d ]` check below also
# covers the one case BuildKit can't prevent: if STRIP_UV_CACHE deleted
# UV_CACHE_DIR in an earlier RUN, it is genuinely missing again here, and
# must be copied back in.
# Intentionally not suppressed (unlike uv_cache_persist below): a failure
# here means UV_CACHE_DIR is missing or incomplete, which would silently
# break hardlink installs or offline extras for the rest of this RUN. Better
# to fail the build loudly (under bash -exu's -e) than continue on a broken
# cache.
uv_cache_restore() {
    if [ "${1:-}" = "force" ] || [ ! -d "${UV_CACHE_DIR}" ]; then
        mkdir -p "${UV_CACHE_DIR}"
        rsync -a /root/.uv-cache-store/ "${UV_CACHE_DIR}/"
    fi
}

# Sync new downloads back to the durable store and, if STRIP_UV_CACHE is set,
# delete the materialized copy before this RUN's layer commits. Pass "prune"
# only on the first uv-writing RUN in a stage, so cleanup performed there (the
# lockfile prune, the aiohttp/yarl removal) is reflected in the store. Later
# RUNs must NOT prune: the store can hold entries this RUN never copied in,
# and --delete there would discard them from the next build's seed.
#
# Both rsyncs here are suppressed on purpose (unlike uv_cache_restore above):
# this runs from an EXIT trap after this RUN's actual build work already
# succeeded, so a transient failure writing back to the durable cache mount
# should degrade the next build's cache hit rate, not fail this build.
uv_cache_persist() {
    if [ "${1:-}" = "prune" ]; then
        rsync -a --delete "${UV_CACHE_DIR}/" /root/.uv-cache-store/ 2>/dev/null || true
    else
        rsync -a "${UV_CACHE_DIR}/" /root/.uv-cache-store/ 2>/dev/null || true
    fi
    if [ -n "${STRIP_UV_CACHE:-}" ]; then rm -rf "${UV_CACHE_DIR}"; fi
}
