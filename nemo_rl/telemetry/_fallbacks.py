# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""No-op fallbacks for when nemo-lens is not installed.

When nemo-lens IS installed, re-exports from nemo.lens.fallbacks for
consistency. When it is NOT installed, provides identical local no-ops.
"""

try:
    from nemo.lens.fallbacks import (  # noqa: F401
        is_span_group_enabled,
        managed_span,
        safe_set_span_attributes,
        span_cm,
        trace_fn,
    )
except ImportError:
    from contextlib import contextmanager

    def trace_fn(group, name, tracer=None):
        """No-op decorator — returns the function unchanged."""
        def decorator(func):
            return func
        return decorator

    @contextmanager
    def managed_span(group, name, tracer=None, **attributes):
        """No-op context manager — yields None."""
        yield None

    def is_span_group_enabled(group):
        """Always returns False when nemo-lens is not installed."""
        return False

    def safe_set_span_attributes(span, attributes, redact_keys=None):
        """No-op."""
        pass

    @contextmanager
    def span_cm(name, tracer=None, record_exception=True, **attributes):
        """No-op context manager — yields None."""
        yield None
