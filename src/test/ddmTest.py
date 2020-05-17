import numpy as np
import sklearn.datasets
from skmultiflow.drift_detection import DDM

ddm = DDM()

# Simulating a data stream as a normal distribution of 1's and 0's
# data_stream = np.random.randint(2, size=10000)
# Changing the data concept from index 4999 to 10000
# for i in range(4999, 10000):
#    data_stream[i] = np.random.randint(4, high=8)

X, y = sklearn.datasets.load_digits(return_X_y=True)

# Adding stream elements to ddm and verifying if drift occurred
for i in range(len(y)):
    ddm.add_element(y[i])
    if ddm.detected_warning_zone():
        print('Warning zone has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
    if ddm.detected_change():
        print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))