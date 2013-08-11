from scipy.io import loadmat
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG)

class Normalization:
  def __init__(self, buffer_):
    self.buffer_ = buffer_

class HF:
  def __init__(self, buffer_):
    index = np.int_(buffer_['TimeTicksHF'][0][0].flatten())
    self._hf = pd.DataFrame(buffer_['HF'][0][0].transpose(), index=index)

class Phase:
  def __init__(self, buffer_, phase):
    phase = str(phase)
    index = np.int_(buffer_['TimeTicks' + phase][0][0].flatten())
    self._phase = pd.DataFrame({
      'voltage': buffer_['LF' + phase + 'V'][0][0][:,0],
      'current': buffer_['LF' + phase + 'I'][0][0][:,0] },
      index=index)
    self._phase['factor'] = self.factor()
    self.threshold = self._phase.factor.std() * .25
    self.initial_value = self._phase.factor.iloc[0]

  def factor(self):
    power = np.asarray(self._phase.voltage) * \
        np.conjugate(np.asarray(self._phase.current))
    return np.cos(np.angle(power))

  def at_rest(self):
    mask = abs(self._phase.factor - 1) < self.threshold
    return self._phase.factor.where(mask)

class Tagging:
  def __init__(self, buffer_):
    self.tagging_info = buffer_['TaggingInfo'][0][0]

  def each_event(self):
    for event in self.tagging_info:
      appliance_id = event[0][0][0]
      appliance_name = event[1][0][0][0]
      on_time = event[2][0][0]
      off_time = event[3][0][0]
      yield (appliance_id, appliance_name, on_time, off_time)

if __name__ == '__main__':
  for file_ in glob('data/H1/Tagged_Training*.mat'):
    data = loadmat(file_)
    buffer_ = data['Buffer']

    l1 = Phase(buffer_, 1)
    l2 = Phase(buffer_, 2)

    on_off = l1.at_rest()

    hf = HF(buffer_)

    tagging = Tagging(buffer_)

    plt.figure()

    bottom = min(l1._phase.factor.min(), l2._phase.factor.min())
    on_times, off_times, devices = [], [], set()
    for (_, appliance_name, on_time, off_time) in tagging.each_event():
      on_times.append(on_time)
      off_times.append(off_time)
      devices.add(appliance_name)
      plt.plot((on_time, on_time), (1, bottom), '-g')
      plt.plot((off_time, off_time), (1, bottom), '-m')

    plt.title(', '.join(devices))

    plt.plot(l1._phase.factor.index, l1._phase.factor, '-r')
    plt.plot(l2._phase.factor.index, l2._phase.factor, '-b')

    plt.xlim(min(on_times), max(off_times))

    plt.show()
