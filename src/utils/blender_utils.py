from typing import Optional
from torch import Tensor
import smplx
from dataclasses import dataclass, fields
import json


def read_json(p):
    with open(p, 'r') as fp:
        json_contents = json.load(fp)
    return json_contents

def write_json(data, p):
    with open(p, 'w') as fp:
        json.dump(data, fp, indent=2)


@dataclass
class Datastruct:
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __iter__(self):
        return self.keys()

    def keys(self):
        keys = [t.name for t in fields(self)]
        return iter(keys)

    def values(self):
        values = [getattr(self, t.name) for t in fields(self)]
        return iter(values)

    # TODO define slice [:]

    def values(self):
        values = [getattr(self, t.name) for t in fields(self)]
        return iter(values)

    def items(self):
        data = [(t.name, getattr(self, t.name)) for t in fields(self)]
        return iter(data)

    def to(self, *args, **kwargs):
        for key in self.datakeys:
            if self[key] is not None:
                self[key] = self[key].to(*args, **kwargs)
        return self

    @property
    def device(self):
        return self[self.datakeys[0]].device

    def detach(self):
        def detach_or_none(tensor):
            if tensor is not None:
                return tensor.detach()
            return None

        kwargs = {key: detach_or_none(self[key])
                  for key in self.datakeys}
        return self.transforms.Datastruct(**kwargs)


from typing import List, Dict
from torch import Tensor
import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_line(v, out='out.png'):
    fig = plt.figure()  # create a figure object
    # ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
    fig, ax = plt.subplots(figsize=(5, 3))
    x = np.arange(v.shape[0])
    ax.plot(x, v)
    fig.savefig(out)   # save the figure to file
    plt.close(fig)    # close the figure window


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    
    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def collate_tensor_with_padding(batch: List[Tensor]) -> Tensor:
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch), ) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas



class Transform:
    def collate(self, lst_datastruct):
        from space.data.tools import collate_tensor_with_padding
        example = lst_datastruct[0]
        def collate_or_none(key):
            if example[key] is None:
                return None            
            key_lst = [x[key] for x in lst_datastruct]
            return collate_tensor_with_padding(key_lst)

        kwargs = {key: collate_or_none(key)
                  for key in example.datakeys}

        return self.Datastruct(**kwargs)



class RotIdentityTransform(Transform):
    def __init__(self, **kwargs):
        return

    def Datastruct(self, **kwargs):
        return RotTransDatastruct(**kwargs)

    def __repr__(self):
        return "RotIdentityTransform()"


@dataclass
class RotTransDatastruct(Datastruct):
    rots: Tensor
    trans: Tensor

    transforms: RotIdentityTransform = RotIdentityTransform()

    def __post_init__(self):
        self.datakeys = ["rots", "trans"]

    def __len__(self):
        return len(self.rots)
class RotIdentityTransform(Transform):
    def __init__(self, **kwargs):
        return

    def Datastruct(self, **kwargs):
        return RotTransDatastruct(**kwargs)

    def __repr__(self):
        return "RotIdentityTransform()"
