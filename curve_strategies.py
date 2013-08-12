from scipy.optimize import *
import numpy as np

class CurveStrategy:
  def __init__(self, x=None):
    self.x = x

  def __function(self, x, *params):
    return x

  def fit(self, y):
    raise "Not implemented"

  def get_params(self):
    return self.params

  def get_curve(self, x=None):
    if x is None:
      x = self.x
    return self.__function(x, self.params)

class GuassianStrategy(CurveStrategy):
  def __function(self, x, *params):
    sigmag, mu, a = params
    return a*np.exp(-(x-mu)**2/(2.*sigmag**2))

  def fit(self, y):
    if self.x is None:
      self.x = np.arange(float(len(y)))
    x = self.x

    p0 = (1./np.std(y), np.argmax(y), max(y))

    try:
      popt, pcov = curve_fit(self.__function, x, y, maxfev=10000, p0=p0)
    except RuntimeError:
      popt = p0

    self.params = popt

class PolyFitStrategy(CurveStrategy):
  DEGREE = 4

  def __function(self, x, *params):
    np.poly1d(*params)

  def fit(self, y):
    if self.x is None:
      self.x = np.arange(float(len(y)))

    self.params = np.polyfit(self.x, y, self.DEGREE)
