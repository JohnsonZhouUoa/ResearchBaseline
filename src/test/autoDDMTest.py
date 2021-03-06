from src.detector.AutoDDM import AutoDDM
from skika.data.reccurring_concept_stream import RCStreamType, RecurringConceptStream, conceptOccurence, RecurringConceptGradualStream
import matplotlib.pyplot as plt
import warnings
import time
import numpy as np
import random
import collections
from skmultiflow.trees import HoeffdingTreeClassifier, HoeffdingAdaptiveTreeClassifier, ExtremelyFastDecisionTreeClassifier
from guppy import hpy



warnings.filterwarnings('ignore')
plt.style.use("seaborn-whitegrid")

# Global variable
TRAINING_SIZE = 1
STREAM_SIZE = 6000000
grace = 1000
tolence = 600
DRIFT_INTERVALS = [50000]
concepts = [0, 1, 2]
total_D_mine = []
total_TP_mine = []
total_FP_mine = []
total_RT_mine = []
total_DIST_mine = []
total_mean_mem = []
precisions = []
recalls = []
f1_scores = []
f2_scores = []
RANDOMNESS = 100
seeds = [6976, 2632, 2754, 5541, 3681, 1456, 7041, 328, 5337, 4622,
         2757, 1788, 3399, 4639, 5306, 5742, 3015, 1554, 8548, 1313,
         4738, 9458, 8145, 3624, 1913, 1654, 2988, 2031, 1802, 4338]
ignore = 0
random.seed(6976)


for k in range(len(seeds)):
    seed = seeds[k]#random.randint(0, 10000)
    keys = []
    actuals = [0]
    concept_chain = {0:0}
    current_concept = 0
    for i in range(1,STREAM_SIZE+1):
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

                    i2 = i + 17000
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

    concept_0 = conceptOccurence(id = 0, difficulty = 6, noise = 0,
                            appearences = x[0], examples_per_appearence = max(DRIFT_INTERVALS))
    concept_1 = conceptOccurence(id = 1, difficulty = 6, noise = 0,
                        appearences = x[1], examples_per_appearence = max(DRIFT_INTERVALS))
    concept_2 = conceptOccurence(id=2, difficulty=6, noise=0,
                                 appearences=x[2], examples_per_appearence=max(DRIFT_INTERVALS))
    # concept_3 = conceptOccurence(id=3, difficulty=6, noise=0,
    #                              appearences=x[2], examples_per_appearence=max(DRIFT_INTERVALS))
    desc = {0: concept_0, 1:concept_1, 2:concept_2}

    datastream = RecurringConceptStream(
        rctype=RCStreamType.RBF,
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
    d_mine = 0
    w_mine= 0
    TP_mine = []
    FP_mine = []
    RT_mine = []
    mem_mine = []
    grace_end = n_global
    detect_end = n_global
    DIST_mine = []
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
    wrong_pred = []
    ML_accuracy = []

    mineDDM = AutoDDM()
    h = hpy()
    while datastream.has_more_samples():
        n_global += 1

        X_test, y_test = datastream.next_sample()
        y_predict = ht.predict(X_test)
        mine_start_time = time.time()
        mineDDM.add_element(y_test != y_predict, n_global)
        wrong_pred.append(y_test != y_predict)
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
                drift_point = key = min(actuals, key=lambda x: abs(x - n_global))
                if (drift_point != 0 and drift_point not in TP_mine and abs(drift_point - n_global) <= tolence):
                    print("A true positive detected at " + str(n_global))
                    DIST_mine.append(abs(n_global - drift_point))
                    TP_mine.append(drift_point)
                    ht = HoeffdingTreeClassifier()
                    mineDDM.detect_TP(n_global)
                    grace_end = n_global + grace
                else:
                    print("A false positive detected at " + str(n_global))
                    mineDDM.detect_FP(n_global)
                    FP_mine.append(drift_point)
        ht.partial_fit(X_test, y_test)
    x = h.heap()
    mem_mine.append(x.size)

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
    print("Mean Memory by mine:" + str(mem_mine))
    total_mean_mem.append(mem_mine)

    precision = len(TP_mine) / (len(TP_mine) + len(FP_mine))
    recall = len(TP_mine) / (len(actuals) - 1)
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

print("D: ", str(total_D_mine))
print("Average Drift Detected: ", str(np.mean(total_D_mine)))
print("Drift Detected Standard Deviation: ", str(np.std(total_D_mine)))

print("TP: ", str(total_TP_mine))
print("Average TP: ", str(np.mean(total_TP_mine)))
print("TP Standard Deviation: ", str(np.std(total_TP_mine)))

print("FP: ", str(total_FP_mine))
print("Average FP: ", str(np.mean(total_FP_mine)))
print("FP Standard Deviation: ", str(np.std(total_FP_mine)))

print("RT: ", str(total_RT_mine))
print("Average RT: ", str(np.mean(total_RT_mine)))
print("RT Standard Deviation: ", str(np.std(total_RT_mine)))

print("DIST: ", str(total_DIST_mine))
print("Average DIST: ", str(np.mean(total_DIST_mine)))
print("DIST Standard Deviation: ", str(np.std(total_DIST_mine)))

print("MEM: ", str(total_mean_mem))
print("Average Memory: ", str(np.mean(total_mean_mem)))
print("Memory Standard Deviation: ", str(np.std(total_mean_mem)))

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