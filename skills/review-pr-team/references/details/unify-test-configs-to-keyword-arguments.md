# unify-test-configs-to-keyword-arguments

> A unit test that builds a BaseModel config from a dict literal is a consistency finding; ask to unify it to keyword arguments

**Why:** tests across the repo build config models in several ways (`Foo(k=v)`, `Foo.model_construct(...)`,
`Foo.model_validate({...})`, `Foo(**{...})`). One way makes them easy to read and grep, and keyword
arguments fail on a misspelled field name where a dict literal can hide it.

**How to apply:**

1. In the unit tests a PR adds or edits, look for `model_validate({...})`, `Foo(**{...})` and nested
   dict literals passed where a model is expected.
2. The preferred order is: construct directly (`Foo(k=v)`) whenever the required fields are given;
   otherwise `Foo.model_construct(k=v, nested=Bar.model_construct(...))`, or a bare
   `Foo.model_construct()` when no values are needed.
3. Skip the finding when the test validates a config loaded from a real recipe YAML; there validation
   is the thing under test.
4. Phrase the ask as "unify to keyword arguments to keep consistency", not as a ban on dicts.

**Seen:** #4278 · #3916
