from functools import wraps
import inspect

import torch
from functorch import vmap


def initializer(func):
    """
    Automatically assigns the parameters.

    >>> class process:
    ...     @initializer
    ...     def __init__(self, cmd, reachable=False, user='root'):
    ...         pass
    >>> p = process('halt', True)
    >>> p.cmd, p.reachable, p.user
    ('halt', True, 'root')
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        (
            names,
            varargs,
            varkw,
            defaults,
            kwonlyargs,
            kwonlydefaults,
            annotations,
        ) = inspect.getfullargspec(func)
        for name, arg in list(zip(names[1:], args)) + list(kwargs.items()):
            setattr(self, name, arg)

        for name, default in zip(reversed(names), reversed(defaults)):
            if not hasattr(self, name):
                setattr(self, name, default)

        func(self, *args, **kwargs)

    return wrapper

"""
vectorizes torch that gather so that the index tensor can have a batch dimension.

it's the same thing as torch.gather but it would perform the same operation on the source
tensor B times.
"""
vect_gather = vmap(torch.gather, (None,None, 0))