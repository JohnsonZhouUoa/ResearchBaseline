from skmultiflow.drift_detection import ADWIN
from sklearn.tree import DecisionTreeClassifier
from skika.data.reccurring_concept_stream import RCStreamType, RecurringConceptStream, conceptOccurence
import matplotlib.pyplot as plt
import warnings
import time
import numpy as np


start_time = time.time()

warnings.filterwarnings('ignore')
plt.style.use("seaborn-whitegrid")

# Global variable
DEFAULT_PR = 0.5
GLOBAL_RATE = 0
INITAL_TRAINING = 100
STREAM_SIZE = 15000
DRIFT_INTERVAL = 100
total_D = []
total_TP = []
total_FP = []
total_RT = []

for i in range(0, 10):

    concept_chain = {0:0}
    concept = 0
    for i in range(1,STREAM_SIZE):
        if i % DRIFT_INTERVAL == 0:
            if concept == 0:
                concept_chain[i] = 1
                concept = 1
            else:
                concept_chain[i] = 0
                concept = 0

    concept_0 = conceptOccurence(id = 0, difficulty = 2, noise = 0,
                            appearences = 2, examples_per_appearence = 5000)
    concept_1 = conceptOccurence(id = 1, difficulty = 3, noise = 0,
                        appearences = 1, examples_per_appearence = 5000)
    desc = {0: concept_0, 1: concept_1}

    datastream = RecurringConceptStream(
                        rctype = RCStreamType.STAGGER,
                        num_samples =STREAM_SIZE,
                        noise = 0,
                        concept_chain = concept_chain,
                        desc = desc,
                        boost_first_occurance = False)

    X_train, y_train = datastream.next_sample(INITAL_TRAINING)

    clf = DecisionTreeClassifier()

    clf.fit(X_train, y_train)

    n_global = INITAL_TRAINING # Cumulative Number of observations
    n_local = 0 # Number of observations
    d_global = 0 # Number of detected drifts
    warning = 0
    TP = []
    FP = []

    ddm = ADWIN()
    while datastream.has_more_samples():
        n_global += 1
        n_local += 1

        X_test, y_test = datastream.next_sample()
        y_predict = clf.predict(X_test)

        ddm.add_element(float(y_test != y_predict))
        if ddm.detected_warning_zone():
            #print('Warning zone has been detected at n: ' + str(n_global) + ' - of x: ' + str(X_test))
            warning += 1
        if ddm.detected_change():
            d_global += 1
            clf.fit(X_test, y_test)
            if(n_global // DRIFT_INTERVAL in TP):
                FP.append(n_global // DRIFT_INTERVAL)
            else:
                TP.append(n_global // DRIFT_INTERVAL)
            #print('Change has been detected at n: ' + str(n_global) + ' - of x: ' + str(X_test))
    print("Number of drifts detected: " + str(d_global))
    total_D.append(d_global)
    print("TP:" + str(len(TP)))
    total_TP.append(len(TP))
    print("FP:" + str(len(FP)))
    total_FP.append(len(FP))
    print("Actual:" + str(STREAM_SIZE/DRIFT_INTERVAL))

    print("--- %s seconds ---" % (time.time() - start_time))
    total_RT.append(time.time() - start_time)

print("Average Drift Detected: ", str(np.mean(total_D)))
print("Minimum Drift Detected: ", str(np.min(total_D)))
print("Maximum Drift Detected: ", str(np.max(total_D)))
print("Drift Detected Standard Deviation: ", str(np.std(total_D)))


print("Average TP: ", str(np.mean(total_TP)))
print("Minimum TP: ", str(np.min(total_TP)))
print("Maximum TP: ", str(np.max(total_TP)))
print("TP Standard Deviation: ", str(np.std(total_TP)))

print("Average FP: ", str(np.mean(total_FP)))
print("Minimum FP: ", str(np.min(total_FP)))
print("Maximum FP: ", str(np.max(total_FP)))
print("FP Standard Deviation: ", str(np.std(total_FP)))

print("Average RT: ", str(np.mean(total_RT)))
print("Minimum RT: ", str(np.min(total_RT)))
print("Maximum RT: ", str(np.max(total_RT)))
print("RT Standard Deviation: ", str(np.std(total_RT)))