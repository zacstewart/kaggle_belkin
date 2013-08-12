from scipy.io import loadmat
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG)

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

class Normalization:
  def __init__(self, buffer_):
    self.buffer_ = buffer_

class HF:
  WINDOW = 25
  THRESHOLD = 8.0

  def __init__(self, buffer_):
    index = np.int_(buffer_['TimeTicksHF'][0][0].flatten())
    self._hf = pd.DataFrame(buffer_['HF'][0][0].transpose(), index=index)

  def segment(self, s):
    i = s * self.WINDOW
    return self._hf.iloc[i:i + self.WINDOW]

  def signature(self, s):
    return self.segment(s).mean()

  def difference(self, s, signature_offset=0):
    return self.segment(s) - self.signature(s+signature_offset)

  def reject_outliers(self, data, m=2.):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

  def max_row(self, segment):
    a = np.asarray(segment)
    return a[a[:,a.max(axis=0).argmax()].argmax(),:]

  def event(self, difference_):
    max_row = self.max_row(difference_)
    smooth_max_row = self.reject_outliers(max_row)
    max_peak = smooth_max_row.max()
    logging.info("Max peak: %(mp)f" % {'mp': max_peak})

    if max_peak > self.THRESHOLD:
      return max_row
    else:
      return None

  def detect_events(self):
    segments = self._hf.shape[0] / self.WINDOW
    for s in range(segments):

      event = self.event(self.difference(s))
      if event is not None:
        difference_ = self.difference(s, signature_offset=1)
        event = self.max_row(difference_)

        logging.info(event.shape)
        plt.figure()
        plt.subplot(311)
        plt.imshow(self.segment(s), interpolation='nearest')
        plt.subplot(312)
        plt.imshow(difference_, interpolation='nearest')
        plt.subplot(313)
        plt.plot(event)
        plt.xlim((0, 4096))
        plt.show()
        plt.close()
        logging.info("Event detected in %(s)d" % {'s': s})
      else:
        logging.info("No event found in %(s)d" % {'s': s})

if __name__ == '__main__':
  for file_ in glob('data/H1/Tagged_Training*.mat'):
    data = loadmat(file_)
    buffer_ = data['Buffer']
    hf = HF(buffer_)
    hf.detect_events()
