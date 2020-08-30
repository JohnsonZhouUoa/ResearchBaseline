from src.detector.MineDDM import MineDDM
from skmultiflow.drift_detection import ADWIN
from skmultiflow.drift_detection import PageHinkley
from skmultiflow.drift_detection import DDM
from sklearn.tree import DecisionTreeClassifier
from skika.data.reccurring_concept_stream import RCStreamType, RecurringConceptStream, conceptOccurence
import matplotlib.pyplot as plt
import warnings
import time
import numpy as np
import random
import collections



warnings.filterwarnings('ignore')
plt.style.use("seaborn-whitegrid")

# Global variable
DEFAULT_PR = 0.5
GLOBAL_RATE = 0
INITAL_TRAINING = 100
STREAM_SIZE = 100000
DRIFT_ONE_INTERVAL = 500
DRIFT_TWO_INTERVAL = 3000
concepts = [1, 2]
total_D = []
total_TP = []
total_FP = []
total_RT = []
ACTUAL_DRIFT = 50

for i in range(0, 10):
    start_time = time.time()
    # drift_points = np.array(sorted(random.sample(range(STREAM_SIZE), ACTUAL_DRIFT)))
    concept_chain = {0:0}
    current_concept = 0
    for i in range(1,STREAM_SIZE):
        # if i in drift_points:
        if i % DRIFT_ONE_INTERVAL == 0 or i % DRIFT_TWO_INTERVAL == 0:
            if current_concept == 0:
                concept_chain[i] = 1
                current_concept = 1
            else:
                concept_chain[i] = 0
                current_concept = 0


    x = collections.Counter(concept_chain.values())

    concept_0 = conceptOccurence(id = 0, difficulty = 2, noise = 0,
                            appearences = x[0], examples_per_appearence = DRIFT_ONE_INTERVAL)
    concept_1 = conceptOccurence(id = 1, difficulty = 3, noise = 0,
                        appearences = x[1], examples_per_appearence = DRIFT_TWO_INTERVAL)
    # concept_2 = conceptOccurence(id=2, difficulty=2, noise=0,
    #                              appearences=x[2], examples_per_appearence=DRIFT_INTERVAL)
    desc = {0: concept_0, 1: concept_1}

    datastream = RecurringConceptStream(
                        rctype = RCStreamType.AGRAWAL,
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

    ddm = DDM()
    while datastream.has_more_samples():
        n_global += 1
        n_local += 1

        X_test, y_test = datastream.next_sample()
        y_predict = clf.predict(X_test)

        ddm.add_element(y_test != y_predict)
        # ddm.add_element(float(y_test != y_predict))
        if ddm.detected_warning_zone():
            #print('Warning zone has been detected at n: ' + str(n_global) + ' - of x: ' + str(X_test))
            warning += 1
        if ddm.detected_change():
            d_global += 1
            clf.fit(X_test, y_test)
            # drift_point = drift_points.flat[np.abs(drift_points - n_global).argmin()]
            # if(drift_point in TP):
            #     FP.append(drift_point)
            # else:
            #     TP.append(drift_point)
            if (n_global // DRIFT_TWO_INTERVAL in TP):
                if (n_global // DRIFT_ONE_INTERVAL in TP):
                    FP.append(n_global // DRIFT_ONE_INTERVAL)
                else:
                    TP.append(n_global // DRIFT_ONE_INTERVAL)
            else:
                TP.append(n_global // DRIFT_TWO_INTERVAL)
            #print('Change has been detected at n: ' + str(n_global) + ' - of x: ' + str(X_test))
    print("Number of drifts detected: " + str(d_global))
    total_D.append(d_global)
    print("TP:" + str(len(TP)))
    total_TP.append(len(TP))
    print("FP:" + str(len(FP)))
    total_FP.append(len(FP))
    print("Actual:" + str(len(concept_chain)))

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