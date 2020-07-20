import numpy as np
import sklearn.datasets
from skmultiflow.drift_detection import DDM
from sklearn import linear_model
import math

ddm = DDM()

TEST_DATA_SIZE = 10000
DRIFT_START = 5000
actual_d = 0
detected_d = 0

# Simulating a data stream
# data_stream = np.random.randint(2, size=TEST_DATA_SIZE)
# Add a concept drift at index 5000
# for i in range(DRIFT_START, TEST_DATA_SIZE):
#     data_stream[i] = np.random.randint(5, 10)

data_stream = np.array([x * 2 + 1 for x in range(1, TEST_DATA_SIZE+1)])

for i in range(50, TEST_DATA_SIZE):
    if i % 50 == 0:
        actual_d += 1
        data_stream[i] = i * 4 + 2

regr = linear_model.LinearRegression()

regr.fit(np.array([x for x in range(1, 21)]).reshape((-1, 1)), data_stream[0:20])

# Adding stream elements to ddm and verifying if drift occurred
for i in range(21, len(data_stream)):
    y_next = data_stream[i - 1]
    y_predict = regr.predict(np.array([i]).reshape((-1, 1)))
    ddm.add_element(math.floor(y_next) != math.floor(y_predict))
    if ddm.detected_warning_zone():
        print('Warning zone has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
    if ddm.detected_change():
        detected_d += 1
        print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))

print("Detected drifts " + str(detected_d) + " out of " + str(actual_d) + " actual drifts")