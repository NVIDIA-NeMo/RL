# unit-tests-use-session-ray-fixture

> NeMo-RL unit tests must rely on conftest's session-wide autouse init_ray_cluster; a test module must not ray.init/ray.shutdown itself (its shutdown kills the shared cluster for later modules)

`tests/unit/conftest.py` defines `init_ray_cluster` as `scope="session", autouse=True` ("This fixture doesn't need to be called directly"). Every unit test already has Ray up. When a new test module adds its own fixture with `ray.init(..., ignore_reinit_error=True)` and `ray.shutdown()`, the init does nothing and the shutdown stops the shared session cluster. Every later module in the lane then runs without `init_ray()`'s settings, or on a plain auto-started cluster.

When reviewing: flag any test-module `ray.init`/`ray.shutdown` in the unit lanes. The ask is to **delete the fixture and drop its argument from the tests**, and rely on the session fixture. Don't suggest turning it into a no-op fixture. The user prefers not having each module manage Ray at all. Precedent to cite: `tests/unit/algorithms/test_async_utils.py` defines `@ray.remote` actors without its own init. (The sglang tests under `models/generation/sglang/` do call `ray.shutdown()`; those predate this rule and run in their own lane.)

**Why:** PR #4052 added `test_instrumentation_ray.py` with its own module fixture; reproduced that a follow-up test sees `ray.is_initialized()` False. The user: "should we ask them to reuse the test fixture for initializing ray? i feel like we don't really want everyone to init it."
**How to apply:** grep new test files for `ray.init(` / `ray.shutdown(`. Deleting the fixture needs a change at every test that takes it, so give a fenced example rather than a suggestion block.
