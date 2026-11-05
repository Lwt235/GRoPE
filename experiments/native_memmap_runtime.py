"""Avoid a redundant NumPy copy before torch.tensor makes its owning copy."""
import numpy as np


class MemmapReadView:
    def __getattr__(self, name):
        return getattr(np, name)

    def array(self, value, *args, **kwargs):
        if isinstance(value, np.memmap) and not args and not kwargs:
            return np.asarray(value)
        return np.array(value, *args, **kwargs)


def install(runner):
    if isinstance(runner.np, MemmapReadView):
        return
    assert runner.np is np
    runner.np = MemmapReadView()
