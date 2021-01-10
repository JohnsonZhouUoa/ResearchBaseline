from src.detector.AutoDDM import AutoDDM
from src.detector.AutoPH import AutoPageHinkley
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
from skmultiflow.trees import HoeffdingTreeClassifier, HoeffdingAdaptiveTreeClassifier, ExtremelyFastDecisionTreeClassifier
from memory_profiler import memory_usage
import arff
import pandas



warnings.filterwarnings('ignore')
plt.style.use("seaborn-whitegrid")

# Global variable
TRAINING_SIZE = 1
STREAM_SIZE = 1000000
grace = 1000
DRIFT_INTERVALS = [10000]
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
total_D_minePH = []
total_TP_minePH = []
total_FP_minePH = []
total_RT_minePH = []
total_DIST_minePH = []
total_D_adwin = []
total_TP_adwin = []
total_FP_adwin = []
total_RT_adwin = []
total_DIST_adwin = []
RANDOMNESS = 100
seeds = [6976, 2632, 2754, 5541, 3681, 1456, 7041, 328, 5337, 4622,
         2757, 1788, 3399, 4639, 5306, 5742, 3015, 1554, 8548, 1313,
         4738, 9458, 8145, 3624, 1913, 1654, 2988, 2031, 1802, 4338]
ignore = 0
random.seed(6976)


for k in range(len(seeds)):
    seed = seeds[k]#random.randint(0, 10000)

    data = arff.load('/hdd2/SINE_1NOISE/SINE' + str(seed) + '.arff')
    df = pandas.DataFrame(data)

    #X_train, y_train = datastream.next_sample(TRAINING_SIZE)
    X_train = []
    y_train = []

    ht = HoeffdingTreeClassifier()

    ht.partial_fit(X_train, y_train)

    n_global = ignore + TRAINING_SIZE # Cumulative Number of observations
    d_mine = 0
    w_mine= 0
    TP_mine = []
    FP_mine = []
    RT_mine = []
    MEMORY_mine = []
    grace_end = n_global
    DIST_mine = [0]
    last = -1
    retrain = False
    mine_pr = []
    mine_std = []
    mine_alpha = []
    pr_min = []
    std_min = []
    pi = []
    mine_x_mean = []
    mine_sum = []
    mine_threshold = []

    mineDDM = AutoDDM(actuals=actuals)
    while datastream.has_more_samples():
        n_global += 1

        X_test, y_test = datastream.next_sample()
        y_predict = ht.predict(X_test)
        mine_start_time = time.time()
        mineDDM.add_element(y_test != y_predict, n_global)
        # mem_use = memory_usage(mineDDM.add_element(y_test != y_predict),max_usage=True)
        # print("Memory_usage:")
        # print(mem_use)

        mine_running_time = time.time() - mine_start_time
        RT_mine.append(mine_running_time)
        # mine_pr.append(mineDDM.get_pr())
        # mine_std.append(mineDDM.get_std())
        # mine_alpha.append(mineDDM.get_global_ratio())
        # pr_min.append(mineDDM.get_min_pi())
        # pi.append(mineDDM.get_pi())
        if (n_global > grace_end):
            if mineDDM.detected_warning_zone():
                w_mine += 1
            if mineDDM.detected_change():
                d_mine += 1
                drift_point = min(actuals, key=lambda x: abs(x - n_global))
                # if(drift_point == 0 or drift_point in TP_mine):
                if mineDDM.get_TP() and not mineDDM.get_FP():
                    print("A true positive detected at " + str(n_global))
                    DIST_mine.append(abs(n_global - drift_point))
                    TP_mine.append(drift_point)
                    ht = HoeffdingTreeClassifier()
                    grace_end = n_global + grace
                else:
                    print("A false positive detected at " + str(n_global))
                    FP_mine.append(drift_point)

        ht.partial_fit(X_test, y_test)

    print("Round " + str(k+1) + " out of 30 rounds")
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

    # plt.plot(mine_pr, color="blue")
    # plt.plot(mine_alpha, color="green")
    # plt.show()
    # plt.plot(pi, color="red")
    # plt.plot(pr_min, color="yellow")
    # plt.show()

    # plt.plot(mine_x_mean, color="red")
    # plt.show()
    # plt.plot(mine_sum, color="blue")
    # plt.plot(mine_threshold, color="green")
    # plt.show()

print("Overall result:")
print("Stream size: " + str(STREAM_SIZE))
print("Drift intervals: " + str(DRIFT_INTERVALS))
print("Actual drifts:" + str(len(actuals)))
print("Seeds: " + str(seeds))

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

precision = np.mean(total_TP_mine)/(np.mean(total_TP_mine) + np.mean(total_FP_mine))
recall = np.mean(total_TP_mine)/(len(actuals) - 1)

print("Precision: ", precision)
print("Recall: ", recall)
print("F1: ", 2 * precision * recall / (precision + recall))
print("F2: ", 5 * precision * recall / (4 * precision + recall))