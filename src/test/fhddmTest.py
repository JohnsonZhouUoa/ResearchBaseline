from thirdparty.tornado.drift_detection import FHDDM
import numpy as np

fhddm = FHDDM()

TEST_DATA_SIZE = 10000
DRIFT_START = 5000

# Simulating a data stream
data_stream = np.random.randint(2, size=TEST_DATA_SIZE)
# Add a concept drift at index 5000
for i in range(DRIFT_START, TEST_DATA_SIZE):
    data_stream[i] = np.random.randint(5, 10)

for i in range(TEST_DATA_SIZE):
    warning_status, drift_status = fhddm.detect(data_stream[i])
    if warning_status:
        print("Warning at " + str(i))
    if drift_status:
        print("Drift at " + str(i))
        fhddm.reset()



