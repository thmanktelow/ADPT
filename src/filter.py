# Functions to determine cutoff frequency, filtering and finite differencing.
# Note that to improve performance, particulary in the optimization of cut-off frequency
# many of these functions are compiled using numba's JIT compiler.
import numpy as np
import scipy as sp
from numba import jit


# Define logit and inverse logit functions
@jit(nopython=True)
def logit(p):
    return np.log(p) - np.log(1 - p)


@jit(nopython=True)
def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))


# Define bounded transformation and its inverse
# these take a variable bounded between a and b and transform it to the real line
@jit(nopython=True)
def bounded_transform(y, a, b):
    # Transforms data bounded between a and b to the real line
    return logit((y - a) / (b - a))


@jit(nopython=True)
def inv_bounded_transform(y, a, b):
    # Transforms data from the real line to the bounded interval [a, b]
    return a + (b - a) * inv_logit(y)


@jit(nopython=True)
def acf(x, lag_max=20):
    return np.array([1] + [np.corrcoef(x[:-i], x[i:])[0, 1] for i in range(1, lag_max)])


@jit(nopython=True)
def mse(y, yhat):
    return np.mean(np.power(y - yhat, 2))


@jit(nopython=True)
def rmse(y, yhat):
    return np.sqrt(mse(y, yhat))


def filt_butter(y, cutoff, fs, ord=4, type="lowpass"):
    # Filters data y with butterworth filter coefficients b using circular padding to ensure continuity.
    # See Derrick 1998 for details on padding for circular continuity

    # Design filter - note order is divided by 2 to account for
    # the fact that the order is doubled via forward/reverse filtering in filtfilt
    b, a = sp.signal.butter(ord / 2, cutoff, fs=fs, btype=type)
    # Pad data to ensure continuity
    halfidx = int(np.floor(len(y) / 2))
    firsthalf = -np.flip(y[:halfidx])
    secondhalf = -np.flip(y[(halfidx):])
    ypad = np.concatenate(
        (
            (firsthalf + (y[0] - firsthalf[-1])),
            y,
            (secondhalf + (y[-1] - secondhalf[0])),
        )
    )
    # Remove linear trend
    p = np.polyfit(np.arange(len(ypad)), ypad, 1)
    fitted_vals = np.polyval(p, np.arange(len(ypad)))
    ypad = ypad - fitted_vals
    # Filter data
    yhat = sp.signal.filtfilt(b, a, ypad)
    # Remove padding
    yhat = (
        yhat[halfidx : (halfidx + len(y))] + fitted_vals[halfidx : (halfidx + len(y))]
    )
    return yhat


def filt_acf(y, fs):
    # Function to determine the cutoff frequency of a butterworth filter
    # based on minimization of residual autocorrelation
    # See Challis 1999 for details.

    # Local cost functions
    def _cost(lfc, y, fs):
        fc = inv_bounded_transform(lfc, 0, fs / 2)
        yhat = filt_butter(y, fc, fs)
        a = acf(y - yhat, lag_max=len(y))
        a = a[1:-1]
        return np.sum(np.power(a, 2))

    def _f(lfc):
        return _cost(lfc, y, fs)

    # Optimize cutoff frequency
    res = sp.optimize.minimize(_f, bounded_transform(10, 0, fs / 2), method="L-BFGS-B")
    fc = inv_bounded_transform(res.x, 0, fs / 2)

    # Filter
    yhat = filt_butter(y, fc, fs)

    # Return filtered data and optimal cutoff frequency
    return yhat, fc


# Simple finite difference function
@jit(nopython=True)
def _cent_finite_diff(y, t, pad=True):
    # local central finite differnece function
    if pad:
        y_pad = np.zeros((len(y) + 2))
        y_pad[0] = y[0]
        y_pad[1:-1] = y
        y_pad[-1] = y[-1]
        t_pad = np.zeros((len(t) + 2))
        t_pad[0] = t[0]
        t_pad[1:-1] = t
        t_pad[-1] = t[-1]
    else:
        y_pad = y
        t_pad = t

    y = np.zeros(len(y))
    for i in range(1, len(y_pad) - 1):
        y[i - 1] = (y_pad[i + 1] - y_pad[i - 1]) / (t_pad[i + 1] - t_pad[i - 1])

    return y


@jit(nopython=True)
def finite_diff(y, t, ord=1, pad=True):

    ytmp = y
    if ord == 0:
        return ytmp
    else:
        for i in range(ord):
            ytmp = _cent_finite_diff(ytmp, t)
        return ytmp
