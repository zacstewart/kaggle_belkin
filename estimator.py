from scipy import linspace, io
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import glob
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special as sp

from event_detector import *
from buffer_container import *

logging.basicConfig(level=logging.DEBUG)

class TargetContainer:
  def __init__(self, buffer_):
    self.tagging_info = buffer_['TaggingInfo'][0][0]

  def each_event(self):
    for event in self.tagging_info:
      appliance_id = event[0][0][0]
      appliance_name = event[1][0][0][0]
      on_time = event[2][0][0]
      off_time = event[3][0][0]
      yield (appliance_id, appliance_name, on_time, off_time)


# Load the .mat files
houses = ['H1', 'H2', 'H3', 'H4']
house_models = {}
examples = {}

for house in houses:
  logging.info('House %(house)s' % {'house': house})

  model = KNeighborsClassifier(n_neighbors=1)
  for training_file in glob.glob('data/' + house + '/Tagged_Training*.mat'):
    logging.info('Loading ' + training_file)
    data = io.loadmat(training_file)
    buffer_ = data['Buffer']

    hf_container = BufferContainer(buffer_)
    target_container = TargetContainer(buffer_)

    for (appliance_id, appliance_name, on_time, off_time) in target_container.each_event():
      logging.info('\t'.join(str(val) for val in ('good', appliance_id, appliance_name, on_time, off_time)))

      if appliance_id not in examples:
        examples[appliance_id] = 0
      examples[appliance_id] += 1

      #Pad segment times
      on_time  -= 15
      off_time += 15

      # Somewhere in the minute if it's a zero segment
      if (off_time - on_time) < 30:
        off_time = on_time + 60

      detector = EventDetector(
          hf_container, on_time=on_time, off_time=off_time, appliance_name=appliance_name)
      off_time, off_time = detector.detect_events()

      features = hf_container.features_between(on_time, off_time, appliance_name)
      logging.info(features)
      if features.shape[0] > 0:
        logging.info('%(events)s events found' % {'events': features.shape[0]})
        features = np.array([features])

        print (features, [appliance_id])
        model.fit(features, [appliance_id])
      else:
        logging.warning('Zero events found')

  house_models[house] = model

test_hf_containers = {}

for house in houses:
  hf_container = BufferContainer()
  for test_file in glob.glob('data/' + house + '/Testing*.mat'):
    logging.info('Loading test file %(test_file)s' % {'test_file': test_file})
    data = io.loadmat(test_file)
    buffer_ = data['Buffer']
    hf_container.add_buffer(buffer_)
  test_hf_containers[house] = hf_container

def predict(row):
  model = house_models[row['House']]
  hf_container = test_hf_containers[row['House']]
  features = hf_container.features_at(row['TimeStamp'])
  prediction = model.predict([features])[0]
  logging.info('%(row_id)s\t\t: %(prediction)d' % {'row_id': row['Id'], 'prediction': prediction})
  row['Predicted'] = 1 if prediction == row['Appliance'] else 0
  return row

submission_file = pd.read_csv('SampleSubmission.csv')

logging.info('Predicting...')
submission_file.apply(predict, axis=1)

logging.info('Writing predictions to disk')
submission_file.to_csv('submission.csv')

import pdb; pdb.set_trace()
