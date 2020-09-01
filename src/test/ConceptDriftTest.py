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
DRIFT_INTERVALS = [500, 3000]
concepts = [1, 2]
total_D = []
total_TP = []
total_FP = []
total_RT = []
actuals = []

for i in range(0, 10):

    concept_chain = {0:0}
    current_concept = 0
    for i in range(1,STREAM_SIZE):
        # if i in drift_points:
        for j in DRIFT_INTERVALS:
            if i % j == 0:
                if i not in concept_chain.keys():
                    actuals.append(i)
                    concept = concepts[random.randint(0, len(concepts)-1)]
                    concept_chain[i] = concept
                    concepts.remove(concept)
                    concepts.append(current_concept)
                    current_concept = concept

    x = collections.Counter(concept_chain.values())

    concept_0 = conceptOccurence(id = 0, difficulty = 2, noise = 0,
                            appearences = x[0], examples_per_appearence = max(DRIFT_INTERVALS))
    concept_1 = conceptOccurence(id = 1, difficulty = 3, noise = 0,
                        appearences = x[1], examples_per_appearence = max(DRIFT_INTERVALS))
    concept_2 = conceptOccurence(id=2, difficulty=2, noise=0,
                                 appearences=x[2], examples_per_appearence=max(DRIFT_INTERVALS))
    desc = {0: concept_0, 1: concept_1, 2:concept_2}

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

    mineDDM = MineDDM()
    ddm = DDM()
    ph = PageHinkley()
    adwin = ADWIN()
    while datastream.has_more_samples():
        n_global += 1
        n_local += 1

        X_test, y_test = datastream.next_sample()
        y_predict = clf.predict(X_test)

        start_time = time.time()
        ddm.add_element(y_test != y_predict)
        running_time = time.time() - start_time
        # ddm.add_element(float(y_test != y_predict))
        if ddm.detected_warning_zone():
            #print('Warning zone has been detected at n: ' + str(n_global) + ' - of x: ' + str(X_test))
            warning += 1
        if ddm.detected_change():
            d_global += 1
            clf.fit(X_test, y_test)
            drift_point = min(actuals, key=lambda x:abs(x-n_global))
            if(drift_point in TP):
                FP.append(drift_point)
            else:
                TP.append(drift_point)

    print("Number of drifts detected: " + str(d_global))
    total_D.append(d_global)
    print("TP:" + str(len(TP)))
    total_TP.append(len(TP))
    print("FP:" + str(len(FP)))
    total_FP.append(len(FP))
    print("Actual:" + str(len(concept_chain)))

    print("--- %s seconds ---" % (running_time))
    total_RT.append(running_time)

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