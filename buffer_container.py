from scipy.optimize import *
import logging
import numpy as np
import pandas as pd

from curve_strategies import *

class BufferNormalizer:
  def __init__(self, buffer_):
    self.hf = buffer_.hf
    self.lf1 = buffer_.lf1
    self.lf2 = buffer_.lf2

    self.l1_mean = self.lf1['factor'].mean()
    self.l2_mean = self.lf2['factor'].mean()

    self.l1_threshold = self.lf1['factor'].std() * .8
    self.l2_threshold = self.lf2['factor'].std() * .8

    self.start_time = min(
        self.hf.index[0], buffer_.lf1.index[0], buffer_.lf2.index[0])
    self.end_time = max(
        self.hf.index[-1], buffer_.lf1.index[-1], buffer_.lf2.index[-1])

  def event_at(self, second):
    try:
      lf1 = self.lf1['factor']
    except:
      lf1 = self.lf1['factor'].index.searchsorted(second)
      lf1 = self.lf1.iloc[lf1]

    l1_diff = abs(lf1.loc[second].mean() - self.l1_mean)

    try:
      lf2 = self.lf2['factor']
    except:
      lf2 = self.lf2['factor'].index.searchsorted(second)
      lf2 = self.lf2.iloc[lf2]

    l2_diff = abs(lf2.mean() - self.l2_mean)

    if l1_diff < self.l1_threshold \
        and l2_diff < self.l2_threshold:
      return False

    return 10

  def baseline_for(self, second):
    index = self.hf.index.searchsorted(second)
    start_idx = max(0, index - 12)
    end_idx = min(index + 12, len(self.hf.index) - 1)
    return self.hf.iloc[start_idx:end_idx].mean()

  def normalize(self):
    second = self.start_time
    while second <= self.end_time:
      logging.info(second)
      duration = self.event_at(second)
      if duration:
        second += duration
      else:
        try:
          self.hf.loc[second] = self.baseline_for(second)
        except:
          print 'Nothing to do'
        second += 1

class BufferContainer:
  def __init__(self, buffer_=None):
    self.hf = pd.DataFrame()
    self.lf1 = pd.DataFrame()
    self.lf2 = pd.DataFrame()

    if buffer_ is not None:
      self.add_buffer(buffer_)

  def add_buffer(self, buffer_):
    time_ticks_hf = np.int_(buffer_['TimeTicksHF'][0][0].flatten())
    self.hf       = self.hf.append(pd.DataFrame(buffer_['HF'][0][0].transpose(), index=time_ticks_hf))

    time_ticks_1  = np.int_(buffer_['TimeTicks1'][0][0].flatten())
    lf1 = pd.DataFrame(index=time_ticks_1)
    lf1['voltage'] = buffer_['LF1V'][0][0][:,0]
    lf1['current'] = buffer_['LF1I'][0][0][:,0]
    lf1['factor'] = \
        self.get_power_factor(lf1['voltage'], lf1['current'])
    self.lf1 = self.lf1.append(lf1)

    time_ticks_2  = np.int_(buffer_['TimeTicks2'][0][0].flatten())
    lf2 = pd.DataFrame(index=time_ticks_2)
    lf2['voltage'] = buffer_['LF2V'][0][0][:,0]
    lf2['current'] = buffer_['LF2I'][0][0][:,0]
    lf2['factor'] = \
        self.get_power_factor(lf2['voltage'], lf2['current'])
    self.lf2 = self.lf2.append(lf2)

    BufferNormalizer(self).normalize()

  def get_power_factor(self, voltage, current):
    power = np.asarray(voltage) * np.conjugate(np.asarray(current))
    return np.cos(np.angle(power))

  def features_between(self, on_time, off_time, appliance_name=''):
    hf_on_time = self.hf.index.searchsorted(on_time)
    hf_off_time = self.hf.index.searchsorted(off_time)

    hf_pre_time = self.hf.index.searchsorted(on_time - 25)
    hf_post_time = self.hf.index.searchsorted(off_time + 25)

    before = self.hf.iloc[hf_pre_time:hf_on_time]
    after = self.hf.iloc[hf_off_time:hf_post_time]

    surrounding_signature = before.append(after).mean()

    event_hf = self.hf.iloc[hf_on_time:hf_off_time]

    #plt.subplot(221)
    #plt.imshow(event_hf, interpolation='nearest', aspect='auto')

    #event_hf = event_hf - surrounding_signature

    #plt.subplot(223)
    #plt.imshow(event_hf, interpolation='nearest', aspect='auto')

    y  = event_hf.mean()
    curve_strategy = PolyFitStrategy()
    curve_strategy.fit(y)
    params = curve_strategy.get_params()

    #plt.subplot(224)
    #plt.plot(
        #y, '-r',
        #curve_strategy.get_curve(), '-g')

    #plt.show()
    #plt.close()

    return params

  def features_at(self, timestamp):
    try:
      features = self.hf.loc[timestamp]
    except KeyError:
      meta_index = self.hf.index.searchsorted(timestamp)
      features = self.hf.iloc[meta_index]

    x = np.arange(float(len(features)))
    return np.polyfit(x, features, 4)
