from src.detector.MineDDM import MineDDM
from skmultiflow.drift_detection import ADWIN
from skmultiflow.drift_detection import PageHinkley
from skmultiflow.drift_detection import DDM
from sklearn.tree import DecisionTreeClassifier
from skika.data.reccurring_concept_stream import RCStreamType, RecurringConceptStream, conceptOccurence, RecurringConceptGradualStream
import matplotlib.pyplot as plt
import warnings
import time
import numpy as np
import random
import collections
import sys



warnings.filterwarnings('ignore')
plt.style.use("seaborn-whitegrid")

# Global variable
INITAL_TRAINING = 1
STREAM_SIZE = 150000
DRIFT_INTERVALS = [500]
concepts = [0, 1, 2]
total_D_mine = []
total_TP_mine = []
total_FP_mine = []
total_RT_mine = []
total_DIST_mine = []
total_D_ddm = []
total_TP_ddm = []
total_FP_ddm = []
total_RT_ddm = []
total_DIST_ddm = []
total_D_ph = []
total_TP_ph = []
total_FP_ph = []
total_RT_ph = []
total_DIST_ph = []
total_D_adwin = []
total_TP_adwin = []
total_FP_adwin = []
total_RT_adwin = []
total_DIST_adwin = []
RANDOMNESS = 50


for k in range(0, 10):
    actuals = [0]
    concept_chain = {0:0}
    current_concept = 0
    for i in range(1,STREAM_SIZE):
        # if i in drift_points:
        for j in DRIFT_INTERVALS:
            if i % j == 0:
                randomness = random.randint(0, RANDOMNESS)
                d = i + ((randomness * 1) if (random.randint(0, 1) > 0) else (randomness * -1))
                if d not in concept_chain.keys():
                    concept_index = random.randint(0, len(concepts)-1)
                    while concepts[concept_index] == current_concept:
                        concept_index = random.randint(0, len(concepts) - 1)
                    concept = concepts[concept_index]
                    concept_chain[d] = concept
                    actuals.append(d)
                    current_concept = concept

    x = collections.Counter(concept_chain.values())
    print(x)

    concept_0 = conceptOccurence(id = 0, difficulty = 2, noise = 0,
                            appearences = x[0], examples_per_appearence = max(DRIFT_INTERVALS))
    concept_1 = conceptOccurence(id = 1, difficulty = 3, noise = 0,
                        appearences = x[1], examples_per_appearence = max(DRIFT_INTERVALS))
    concept_2 = conceptOccurence(id=2, difficulty=2, noise=0,
                                 appearences=x[2], examples_per_appearence=max(DRIFT_INTERVALS))
    desc = {0: concept_0, 1: concept_1, 2:concept_2}

    datastream = RecurringConceptGradualStream(
                        rctype = RCStreamType.STAGGER,
                        num_samples =STREAM_SIZE,
                        noise = 0.1,
                        concept_chain = concept_chain,
                        desc = desc,
                        boost_first_occurance = False)

    X_train, y_train = datastream.next_sample(INITAL_TRAINING)

    clf = DecisionTreeClassifier()

    clf.fit(X_train, y_train)

    n_global = INITAL_TRAINING # Cumulative Number of observations
    d_mine = 0
    d_ddm = 0
    d_ph = 0
    d_adwin = 0
    w_mine= 0
    w_ddm = 0
    w_ph = 0
    w_adwin = 0
    TP_mine = []
    TP_ddm = []
    TP_ph = []
    TP_adwin = []
    FP_mine = []
    FP_ddm = []
    FP_ph = []
    FP_adwin = []
    RT_mine = []
    RT_ddm = []
    RT_ph = []
    RT_adwin = []
    DIST_mine = [0]
    DIST_ddm = [0]
    DIST_ph = [0]
    DIST_adwin = [0]

    mineDDM = MineDDM()
    ddm = DDM()
    ph = PageHinkley()
    adwin = ADWIN()
    while datastream.has_more_samples():
        n_global += 1

        X_test, y_test = datastream.next_sample()
        y_predict = clf.predict(X_test)

        mine_start_time = time.time()
        mineDDM.add_element(y_test != y_predict)
        mine_running_time = time.time() - mine_start_time
        RT_mine.append(mine_running_time)
        if mineDDM.detected_warning_zone():
            #print('Warning zone has been detected at n: ' + str(n_global) + ' - of x: ' + str(X_test))
            w_mine += 1
        if mineDDM.detected_change():
            d_mine += 1
            drift_point = max([i for i in actuals if i <= n_global])
            if(drift_point == 0 or drift_point in TP_mine):
                FP_mine.append(drift_point)
            else:
                DIST_mine.append(abs(n_global - drift_point))
                TP_mine.append(drift_point)
            #clf.fit(X_test, y_test)

        ddm_start_time = time.time()
        ddm.add_element(y_test != y_predict)
        ddm_running_time = time.time() - ddm_start_time
        RT_ddm.append(ddm_running_time)
        if ddm.detected_warning_zone():
            #print('Warning zone has been detected at n: ' + str(n_global) + ' - of x: ' + str(X_test))
            w_ddm += 1
        if ddm.detected_change():
            d_ddm += 1
            drift_point = max([i for i in actuals if i <= n_global])
            if(drift_point == 0 or drift_point in TP_ddm):
                FP_ddm.append(drift_point)
            else:
                DIST_ddm.append(abs(n_global - drift_point))
                TP_ddm.append(drift_point)
            #clf.fit(X_test, y_test)

        ph_start_time = time.time()
        ph.add_element(y_test != y_predict)
        ph_running_time = time.time() - ph_start_time
        RT_ph.append(ph_running_time)
        if ph.detected_warning_zone():
            #print('Warning zone has been detected at n: ' + str(n_global) + ' - of x: ' + str(X_test))
            w_ph += 1
        if ph.detected_change():
            d_ph += 1
            drift_point = max([i for i in actuals if i <= n_global])
            if(drift_point == 0 or drift_point in TP_ph):
                FP_ph.append(drift_point)
            else:
                DIST_ph.append(abs(n_global - drift_point))
                TP_ph.append(drift_point)
            #clf.fit(X_test, y_test)

        adwin_start_time = time.time()
        adwin.add_element(float(y_test != y_predict))
        adwin_running_time = time.time() - adwin_start_time
        RT_adwin.append(adwin_running_time)
        if adwin.detected_warning_zone():
            #print('Warning zone has been detected at n: ' + str(n_global) + ' - of x: ' + str(X_test))
            w_adwin += 1
        if adwin.detected_change():
            d_adwin += 1
            drift_point = max([i for i in actuals if i <= n_global])
            if(drift_point == 0 or drift_point in TP_adwin):
                FP_adwin.append(drift_point)
            else:
                DIST_adwin.append(abs(n_global - drift_point))
                TP_adwin.append(drift_point)
            clf.fit(X_test, y_test)
        if (n_global in actuals):
            clf.fit(X_test, y_test)

    print("Round " + str(k+1) + " out of 10 rounds")
    print("Actual drifts:" + str(len(actuals)))

    print("Number of drifts detected by mine: " + str(d_mine))
    total_D_mine.append(d_mine)
    print("TP by mine:" + str(len(TP_mine)))
    total_TP_mine.append(len(TP_mine))
    print("FP by mine:" + str(len(FP_mine)))
    total_FP_mine.append(len(FP_mine))
    print("Mean RT  %s seconds" % (np.mean(mine_running_time)))
    total_RT_mine.append(np.mean(mine_running_time))
    print("Mean DIST by mine:" + str(np.mean(DIST_mine)))
    total_DIST_mine.append(np.mean(DIST_mine))


    print("Number of drifts detected by ddm: " + str(d_ddm))
    total_D_ddm.append(d_ddm)
    print("TP by ddm:" + str(len(TP_ddm)))
    total_TP_ddm.append(len(TP_ddm))
    print("FP by ddm:" + str(len(FP_ddm)))
    total_FP_ddm.append(len(FP_ddm))
    print("Mean RT  %s seconds" % np.mean((ddm_running_time)))
    total_RT_ddm.append(np.mean(ddm_running_time))
    print("Mean DIST by ddm:" + str(np.mean(DIST_ddm)))
    total_DIST_ddm.append(np.mean(DIST_ddm))

    print("Number of drifts detected by page-hinkley: " + str(d_ph))
    total_D_ph.append(d_ph)
    print("TP by page-hinkley:" + str(len(TP_ph)))
    total_TP_ph.append(len(TP_ph))
    print("FP by page-hinkley:" + str(len(FP_ph)))
    total_FP_ph.append(len(FP_ph))
    print("Mean RT  %s seconds" % (np.mean(ph_running_time)))
    total_RT_ph.append(np.mean(ph_running_time))
    print("Mean DIST by ph:" + str(np.mean(DIST_ph)))
    total_DIST_ph.append(np.mean(DIST_ph))

    print("Number of drifts detected by adwin: " + str(d_adwin))
    total_D_adwin.append(d_adwin)
    print("TP by adwin:" + str(len(TP_adwin)))
    total_TP_adwin.append(len(TP_adwin))
    print("FP by adwin:" + str(len(FP_adwin)))
    total_FP_adwin.append(len(FP_adwin))
    print("Mean RT  %s seconds" % (np.mean(adwin_running_time)))
    total_RT_adwin.append(np.mean(adwin_running_time))
    print("Mean DIST by adwin:" + str(np.mean(DIST_adwin)))
    total_DIST_adwin.append(np.mean(DIST_adwin))

print("Overall result:")
print("Stream size: " + str(STREAM_SIZE))
print("Drift intervals: " + str(DRIFT_INTERVALS))
print("Actual drifts:" + str(len(actuals)))

print("Overall result for mine:")
print("Average Drift Detected: ", str(np.mean(total_D_mine)))
print("Minimum Drift Detected: ", str(np.min(total_D_mine)))
print("Maximum Drift Detected: ", str(np.max(total_D_mine)))
print("Drift Detected Standard Deviation: ", str(np.std(total_D_mine)))

print("Average TP: ", str(np.mean(total_TP_mine)))
print("Minimum TP: ", str(np.min(total_TP_mine)))
print("Maximum TP: ", str(np.max(total_TP_mine)))
print("TP Standard Deviation: ", str(np.std(total_TP_mine)))

print("Average FP: ", str(np.mean(total_FP_mine)))
print("Minimum FP: ", str(np.min(total_FP_mine)))
print("Maximum FP: ", str(np.max(total_FP_mine)))
print("FP Standard Deviation: ", str(np.std(total_FP_mine)))

print("Average RT: ", str(np.mean(total_RT_mine)))
print("Minimum RT: ", str(np.min(total_RT_mine)))
print("Maximum RT: ", str(np.max(total_RT_mine)))
print("RT Standard Deviation: ", str(np.std(total_RT_mine)))

print("Average DIST: ", str(np.mean(total_DIST_mine)))
print("Minimum DIST: ", str(np.min(total_DIST_mine)))
print("Maximum DIST: ", str(np.max(total_DIST_mine)))
print("DIST Standard Deviation: ", str(np.std(total_DIST_mine)))


print("Overall result for ddm:")
print("Average Drift Detected: ", str(np.mean(total_D_ddm)))
print("Minimum Drift Detected: ", str(np.min(total_D_ddm)))
print("Maximum Drift Detected: ", str(np.max(total_D_ddm)))
print("Drift Detected Standard Deviation: ", str(np.std(total_D_ddm)))

print("Average TP: ", str(np.mean(total_TP_ddm)))
print("Minimum TP: ", str(np.min(total_TP_ddm)))
print("Maximum TP: ", str(np.max(total_TP_ddm)))
print("TP Standard Deviation: ", str(np.std(total_TP_ddm)))

print("Average FP: ", str(np.mean(total_FP_ddm)))
print("Minimum FP: ", str(np.min(total_FP_ddm)))
print("Maximum FP: ", str(np.max(total_FP_ddm)))
print("FP Standard Deviation: ", str(np.std(total_FP_ddm)))

print("Average RT: ", str(np.mean(total_RT_ddm)))
print("Minimum RT: ", str(np.min(total_RT_ddm)))
print("Maximum RT: ", str(np.max(total_RT_ddm)))
print("RT Standard Deviation: ", str(np.std(total_RT_ddm)))

print("Average DIST: ", str(np.mean(total_DIST_ddm)))
print("Minimum DIST: ", str(np.min(total_DIST_ddm)))
print("Maximum DIST: ", str(np.max(total_DIST_ddm)))
print("DIST Standard Deviation: ", str(np.std(total_DIST_ddm)))


print("Overall result for page-hinkley:")
print("Average Drift Detected: ", str(np.mean(total_D_ph)))
print("Minimum Drift Detected: ", str(np.min(total_D_ph)))
print("Maximum Drift Detected: ", str(np.max(total_D_ph)))
print("Drift Detected Standard Deviation: ", str(np.std(total_D_ph)))

print("Average TP: ", str(np.mean(total_TP_ph)))
print("Minimum TP: ", str(np.min(total_TP_ph)))
print("Maximum TP: ", str(np.max(total_TP_ph)))
print("TP Standard Deviation: ", str(np.std(total_TP_ph)))

print("Average FP: ", str(np.mean(total_FP_ph)))
print("Minimum FP: ", str(np.min(total_FP_ph)))
print("Maximum FP: ", str(np.max(total_FP_ph)))
print("FP Standard Deviation: ", str(np.std(total_FP_ph)))

print("Average RT: ", str(np.mean(total_RT_ph)))
print("Minimum RT: ", str(np.min(total_RT_ph)))
print("Maximum RT: ", str(np.max(total_RT_ph)))
print("RT Standard Deviation: ", str(np.std(total_RT_ph)))

print("Average DIST: ", str(np.mean(total_DIST_ph)))
print("Minimum DIST: ", str(np.min(total_DIST_ph)))
print("Maximum DIST: ", str(np.max(total_DIST_ph)))
print("DIST Standard Deviation: ", str(np.std(total_DIST_ph)))


print("Overall result for adwin:")
print("Average Drift Detected: ", str(np.mean(total_D_adwin)))
print("Minimum Drift Detected: ", str(np.min(total_D_adwin)))
print("Maximum Drift Detected: ", str(np.max(total_D_adwin)))
print("Drift Detected Standard Deviation: ", str(np.std(total_D_adwin)))

print("Average TP: ", str(np.mean(total_TP_adwin)))
print("Minimum TP: ", str(np.min(total_TP_adwin)))
print("Maximum TP: ", str(np.max(total_TP_adwin)))
print("TP Standard Deviation: ", str(np.std(total_TP_adwin)))

print("Average FP: ", str(np.mean(total_FP_adwin)))
print("Minimum FP: ", str(np.min(total_FP_adwin)))
print("Maximum FP: ", str(np.max(total_FP_adwin)))
print("FP Standard Deviation: ", str(np.std(total_FP_adwin)))

print("Average RT: ", str(np.mean(total_RT_adwin)))
print("Minimum RT: ", str(np.min(total_RT_adwin)))
print("Maximum RT: ", str(np.max(total_RT_adwin)))
print("RT Standard Deviation: ", str(np.std(total_RT_adwin)))

print("Average DIST: ", str(np.mean(total_DIST_adwin)))
print("Minimum DIST: ", str(np.min(total_DIST_adwin)))
print("Maximum DIST: ", str(np.max(total_DIST_adwin)))
print("DIST Standard Deviation: ", str(np.std(total_DIST_adwin)))