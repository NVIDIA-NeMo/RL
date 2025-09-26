import pytest


@pytest.mark.parametrize("a", list(range(8)))
def test_hello(a):
    a == a


def test_foobar():
    assert 3 == 4
