from src.detector.MineDDM import MineDDM
from src.detector.MinePH import MinePageHinkley
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



warnings.filterwarnings('ignore')
plt.style.use("seaborn-whitegrid")

# Global variable
TRAINING_SIZE = 1
STREAM_SIZE = 5000000
grace = 1000
DRIFT_INTERVALS = [30000]
concepts = [0, 1, 2]
total_D_ddm = []
total_TP_ddm = []
total_FP_ddm = []
total_RT_ddm = []
total_DIST_ddm = []
precisions = []
recalls = []
f1_scores = []
f2_scores = []
RANDOMNESS = 0
seeds = [6976, 2632, 2754, 5541, 3681, 1456, 7041, 328, 5337, 4622,
         2757, 1788, 3399, 4639, 5306, 5742, 3015, 1554, 8548, 1313,
         4738, 9458, 8145, 3624, 1913, 1654, 2988, 2031, 1802, 4338]
ignore = 0
random.seed(6976)


for k in range(9, 14):
    seed = seeds[k]#random.randint(0, 10000)
    #seeds.append(seed)
    keys = []
    actuals = [0]
    concept_chain = {0:0}
    current_concept = 0
    for i in range(1,STREAM_SIZE+1):
        # if i in drift_points:
        for j in DRIFT_INTERVALS:
            if i % j == 0:
                if i not in keys:
                    keys.append(i)
                    randomness = random.randint(0, RANDOMNESS)
                    d = i + ((randomness * 1) if (random.randint(0, 1) > 0) else (randomness * -1))
                    concept_index = random.randint(0, len(concepts)-1)
                    while concepts[concept_index] == current_concept:
                        concept_index = random.randint(0, len(concepts) - 1)
                    concept = concepts[concept_index]
                    concept_chain[d] = concept
                    actuals.append(d)
                    current_concept = concept

                    i2 = i + 7000
                    keys.append(i2)
                    randomness = random.randint(0, RANDOMNESS)
                    d = i2 + ((randomness * 1) if (random.randint(0, 1) > 0) else (randomness * -1))
                    concept_index = random.randint(0, len(concepts) - 1)
                    while concepts[concept_index] == current_concept:
                        concept_index = random.randint(0, len(concepts) - 1)
                    concept = concepts[concept_index]
                    concept_chain[d] = concept
                    actuals.append(d)
                    current_concept = concept

    x = collections.Counter(concept_chain.values())
    print(x)
    # desc = {}
    # for i in concept_chain.keys():
    #     if i == concept_chain.keys()[-1]:
    #         desc[i] = conceptOccurence(id = 0, difficulty = 6, noise = 0,
    #                         appearences = x[0], examples_per_appearence = max(DRIFT_INTERVALS))
    #     desc[i] = conceptOccurence

    concept_0 = conceptOccurence(id = 0, difficulty = 6, noise = 0,
                            appearences = x[0], examples_per_appearence = max(DRIFT_INTERVALS))
    concept_1 = conceptOccurence(id = 1, difficulty = 6, noise = 0,
                        appearences = x[1], examples_per_appearence = max(DRIFT_INTERVALS))
    concept_2 = conceptOccurence(id=2, difficulty=6, noise=0,
                                 appearences=x[2], examples_per_appearence=max(DRIFT_INTERVALS))
    # concept_3 = conceptOccurence(id=3, difficulty=6, noise=0,
    #                              appearences=x[2], examples_per_appearence=max(DRIFT_INTERVALS))
    desc = {0: concept_0, 1: concept_1, 2:concept_2}

    datastream = RecurringConceptGradualStream(
        rctype=RCStreamType.SINE,
        num_samples=STREAM_SIZE,
        noise=0,
        concept_chain=concept_chain,
        seed=seed,
        desc=desc,
        boost_first_occurance=False)

    #X_train, y_train = datastream.next_sample(TRAINING_SIZE)
    X_train = []
    y_train = []
    for i in range(0, ignore + TRAINING_SIZE):
        if i < ignore:
            continue
        X, y = datastream.next_sample()
        X_train.append(X[0])
        y_train.append(y[0])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    ht = HoeffdingTreeClassifier()

    ht.partial_fit(X_train, y_train)

    n_global = ignore + TRAINING_SIZE # Cumulative Number of observations
    d_ddm = 0
    w_ddm = 0
    TP_ddm = []
    FP_ddm = []
    RT_ddm = []
    DIST_ddm = [0]
    retrain = False
    grace_end = n_global
    detect_end = n_global
    mine_pr = []
    mine_std = []
    mine_alpha = []
    pr_min = []
    std_min = []
    pi = []
    mine_x_mean = []
    mine_sum = []
    mine_threshold = []
    pred_grace_ht = []
    pred_grace_ht_p = []
    ht_p = None
    TP_var = []

    ddm = DDM()
    while datastream.has_more_samples():
        n_global += 1

        X_test, y_test = datastream.next_sample()
        y_predict = ht.predict(X_test)
        ddm_start_time = time.time()
        ddm.add_element(y_test != y_predict)

        ddm_running_time = time.time() - ddm_start_time
        RT_ddm.append(ddm_running_time)
        if (n_global > grace_end):
            if (n_global > detect_end):
                if ht_p is not None:
                    drift_point = detect_end - 2 * grace
                    print("Accuracy of ht: " + str(np.mean(pred_grace_ht)))
                    print("Accuracy of ht_p: " + str(np.mean(pred_grace_ht_p)))
                    if (np.mean(pred_grace_ht_p) > np.mean(pred_grace_ht)):
                        print("TP detected at: " + str(drift_point))
                        TP_ddm.append(drift_point)
                        ht = ht_p
                        TP_var.append(abs(np.mean(pred_grace_ht_p) - np.mean(pred_grace_ht)))
                    else:
                        print("FP detected at: " + str(drift_point))
                        FP_ddm.append(drift_point)


                    ht_p = None
                    pred_grace_ht = []
                    pred_grace_ht_p = []
                if ddm.detected_warning_zone():
                    w_ddm += 1
                if ddm.detected_change():
                    d_ddm += 1
                    ht_p = HoeffdingTreeClassifier()
                    grace_end = n_global + grace
                    detect_end = n_global + 2 * grace
            else:
                pred_grace_ht.append(y_test == y_predict)
                pred_grace_ht_p.append(y_test == ht_p.predict(X_test))

        if ht_p is not None:
            ht_p.partial_fit(X_test, y_test)
        ht.partial_fit(X_test, y_test)

    print("Round " + str(k+1) + " out of 10 rounds")
    print("Actual drifts:" + str(len(actuals)))

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

    precision = len(TP_ddm) / (len(TP_ddm) + len(FP_ddm))
    recall = len(TP_ddm) / (len(actuals) - 1)
    f1 = 2 * precision * recall / (precision + recall)
    f2 = 5 * precision * recall / (4 * precision + recall)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    f2_scores.append(f2)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    print("F2: ", f2)

print("Overall result:")
print("Stream size: " + str(STREAM_SIZE))
print("Drift intervals: " + str(DRIFT_INTERVALS))
print("Actual drifts:" + str(len(actuals) - 1))
print("Seeds: " + str(seeds))

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

print("Precisions: " + str(precisions))
print("Average: " + str(np.mean(precisions)))
print("Deviation: " + str(np.std(precisions)))

print("Recalls: " + str(recalls))
print("Average: " + str(np.mean(recalls)))
print("Deviation: " + str(np.std(recalls)))

print("F1 scores: " + str(f1_scores))
print("Average: " + str(np.mean(f1_scores)))
print("Deviation: " + str(np.std(f1_scores)))

print("F2 scores: " + str(f2_scores))
print("Average: " + str(np.mean(f2_scores)))
print("Deviation: " + str(np.std(f2_scores)))