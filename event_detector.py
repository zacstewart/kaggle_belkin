import logging
import numpy as np
import pandas as pd

class EventDetector:
  def __init__(self, buffer_, on_time=None, off_time=None, appliance_name=''):
    self.buffer_ = buffer_
    self.appliance_name = appliance_name
    self.on_time, self.off_time = on_time, off_time

    self.set_phase1()
    self.set_phase2()

  def get_power_factor(self, voltage, current):
    power = voltage * np.conjugate(current)
    return np.cos(np.angle(np.asarray(power)))

  def set_phase1(self):
    on_time   = self.buffer_.lf1.index.searchsorted(self.on_time)
    off_time  = self.buffer_.lf1.index.searchsorted(self.off_time)
    self.lf1v = self.buffer_.lf1['voltage'].iloc[on_time:off_time]
    self.lf1i = self.buffer_.lf1['current'].iloc[on_time:off_time]
    self.l1_power_factor = self.buffer_.lf1['factor'].iloc[on_time:off_time]

  def set_phase2(self):
    on_time   = self.buffer_.lf2.index.searchsorted(self.on_time)
    off_time  = self.buffer_.lf2.index.searchsorted(self.off_time)
    self.lf2v = self.buffer_.lf2['voltage'].iloc[on_time:off_time]
    self.lf2i = self.buffer_.lf2['current'].iloc[on_time:off_time]
    self.l2_power_factor = self.buffer_.lf2['factor'].iloc[on_time:off_time]

  def evented_phase(self):
    l1_std, l2_std = self.l1_power_factor.std(), self.l2_power_factor.std()

    if l1_std > l2_std:
      return (self.lf1v.index, self.l1_power_factor, l1_std)
    else:
      return (self.lf2v.index, self.l2_power_factor, l2_std)

  def detect_events(self):
    logging.info('Detecting events')
    index, phase, std = self.evented_phase()
    threshold = std * 0.8

    start_value = phase[1]
    end_value   = phase[-1]

    logging.info('Start vs end: ' + str(start_value - end_value))

    start_index = index[0]
    end_index = index[-1]

    prev_i = 0
    for i in range(len(phase)):
      diff = abs(phase[i] - start_value)
      if diff > threshold:
        start_index = index[i]
        prev_i = i
        break

    if abs(end_value - start_value) < std:
      for i in range(len(phase) - 1, prev_i - 1, -1):
        diff = abs(phase[i] - start_value)
        if diff > threshold:
          end_index = index[i]
          break

    # plt.figure()
    # plt.title(self.appliance_name)

    # ax1 = plt.subplot(222)

    # ax1.plot(
    #    self.lf1v.index, self.l1_power_factor, 'r',
    #    self.lf2v.index, self.l2_power_factor, 'b')

    # ax1.plot([start_index, start_index], [0, 1], color='k', linestyle='-')
    # ax1.plot([end_index, end_index], [0, 1], color='k', linestyle='-')

    return (start_index, end_index)
