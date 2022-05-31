import pytest
from inFairness.utils.misc import initializer


def test_initializer():
    class MyClass:
        @initializer
        def __init__(self, a, b=1):
            pass

    x = MyClass(a=1, b=2)
    assert x.a == 1 and x.b == 2
    x = MyClass(a=1)
    assert x.a == 1 and x.b == 1


if __name__ == "__main__":
    test_initializer()
