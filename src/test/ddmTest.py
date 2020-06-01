import numpy as np
import sklearn.datasets
from skmultiflow.drift_detection import DDM

ddm = DDM()

TEST_DATA_SIZE = 10000
DRIFT_START = 5000

# Simulating a data stream
data_stream = np.random.randint(2, size=TEST_DATA_SIZE)
# Add a concept drift at index 5000
for i in range(DRIFT_START, TEST_DATA_SIZE):
    data_stream[i] = np.random.randint(5, 10)

# Adding stream elements to ddm and verifying if drift occurred
for i in range(len(data_stream)):
    ddm.add_element(data_stream[i])
    if ddm.detected_warning_zone():
        print('Warning zone has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
    if ddm.detected_change():
        print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))