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



warnings.filterwarnings('ignore')
plt.style.use("seaborn-whitegrid")

# Global variable
TRAINING_SIZE = 1
STREAM_SIZE = 500000
DRIFT_INTERVALS = [5000]
concepts = [0, 1]
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
RANDOMNESS = 0
seeds = [1235, 5614, 95, 1648, 5356]
ignore = 0


for k in range(0, 5):
    seed = seeds[k]#random.randint(0, 10000)
    #seeds.append(seed)
    keys = []
    actuals = [0]
    concept_chain = {0:0}
    current_concept = 0
    for i in range(1,STREAM_SIZE):
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
    # concept_2 = conceptOccurence(id=2, difficulty=2, noise=0,
    #                              appearences=x[2], examples_per_appearence=max(DRIFT_INTERVALS))
    desc = {0: concept_0, 1: concept_1}

    datastream = RecurringConceptStream(
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
    d_mine = 0
    d_ddm = 0
    d_ph = 0
    d_minePH = 0
    d_adwin = 0
    w_mine= 0
    w_ddm = 0
    w_ph = 0
    w_minePH = 0
    w_adwin = 0
    TP_mine = []
    TP_ddm = []
    TP_ph = []
    TP_minePH = []
    TP_adwin = []
    FP_mine = []
    FP_ddm = []
    FP_ph = []
    FP_minePH = []
    FP_adwin = []
    RT_mine = []
    RT_ddm = []
    RT_ph = []
    RT_minePH = []
    RT_adwin = []
    DIST_mine = [0]
    DIST_ddm = [0]
    DIST_ph = [0]
    DIST_minePH = [0]
    DIST_adwin = [0]
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

    mineDDM = MineDDM(actuals=actuals, initial_training=n_global)
    ddm = DDM()
    ph = PageHinkley()
    minePH = MinePageHinkley(actuals=actuals, initial_training=n_global)
    adwin = ADWIN()
    while datastream.has_more_samples():
        n_global += 1

        X_test, y_test = datastream.next_sample()
        y_predict = ht.predict(X_test)

        mine_start_time = time.time()
        mineDDM.add_element(y_test != y_predict)
        mine_running_time = time.time() - mine_start_time
        RT_mine.append(mine_running_time)
        mine_pr.append(mineDDM.get_pr())
        mine_std.append(mineDDM.get_std())
        mine_alpha.append(mineDDM.get_global_ratio())
        pr_min.append(mineDDM.get_min_pi())
        pi.append(mineDDM.get_pi())
        if mineDDM.detected_warning_zone():
            w_mine += 1
        if mineDDM.detected_change():
            d_mine += 1
            drift_point = max([i for i in actuals if i <= n_global])
            if(drift_point == 0 or drift_point in TP_mine):
                FP_mine.append(drift_point)
            else:
                DIST_mine.append(abs(n_global - drift_point))
                TP_mine.append(drift_point)

        ddm_start_time = time.time()
        ddm.add_element(y_test != y_predict)
        ddm_running_time = time.time() - ddm_start_time
        RT_ddm.append(ddm_running_time)
        if ddm.detected_warning_zone():
            w_ddm += 1
        if ddm.detected_change():
            d_ddm += 1
            drift_point = max([i for i in actuals if i <= n_global])
            if(drift_point == 0 or drift_point in TP_ddm):
                FP_ddm.append(drift_point)
            else:
                DIST_ddm.append(abs(n_global - drift_point))
                TP_ddm.append(drift_point)

        ph_start_time = time.time()
        ph.add_element(y_test != y_predict)
        ph_running_time = time.time() - ph_start_time
        RT_ph.append(ph_running_time)
        if ph.detected_warning_zone():
            w_ph += 1
        if ph.detected_change():
            d_ph += 1
            drift_point = max([i for i in actuals if i <= n_global])
            if(drift_point == 0 or drift_point in TP_ph):
                FP_ph.append(drift_point)
            else:
                DIST_ph.append(abs(n_global - drift_point))
                TP_ph.append(drift_point)

        minePH_start_time = time.time()
        minePH.add_element(y_test != y_predict)
        minePH_running_time = time.time() - minePH_start_time
        RT_minePH.append(minePH_running_time)
        mine_x_mean.append(minePH.get_mean())
        mine_sum.append(minePH.get_sum())
        mine_threshold.append(minePH.get_threshold())
        if minePH.detected_warning_zone():
            w_minePH += 1
        if minePH.detected_change():
            d_minePH += 1
            drift_point = max([i for i in actuals if i <= n_global])
            if (drift_point == 0 or drift_point in TP_minePH):
                FP_minePH.append(drift_point)
            else:
                DIST_minePH.append(abs(n_global - drift_point))
                TP_minePH.append(drift_point)

        adwin_start_time = time.time()
        adwin.add_element(float(y_test != y_predict))
        adwin_running_time = time.time() - adwin_start_time
        RT_adwin.append(adwin_running_time)
        if adwin.detected_warning_zone():
            w_adwin += 1
        if adwin.detected_change():
            d_adwin += 1
            drift_point = max([i for i in actuals if i <= n_global])
            if(drift_point == 0 or drift_point in TP_adwin):
                FP_adwin.append(drift_point)
            else:
                DIST_adwin.append(abs(n_global - drift_point))
                TP_adwin.append(drift_point)
        # if (n_global in actuals):
        #     last = n_global
        #     #print("Last drift point: "+str(last))
        #     #print("Start re-training")
        #     # X_train = []
        #     # y_train = []
        #     retrain = True
        #
        # if retrain and n_global - last <= TRAINING_SIZE:
        #     #print("Re-training")
        #     # X_train.append(X_test[0])
        #     # y_train.append(y_test[0])
        #     ht.partial_fit(X_test, y_test)
        # else:
        #     retrain = False
        ht.partial_fit(X_test, y_test)

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

    print("Number of drifts detected by mine PH: " + str(d_minePH))
    total_D_minePH.append(d_minePH)
    print("TP by minePH:" + str(len(TP_minePH)))
    total_TP_minePH.append(len(TP_minePH))
    print("FP by minePH:" + str(len(FP_minePH)))
    total_FP_minePH.append(len(FP_minePH))
    print("Mean RT  %s seconds" % (np.mean(minePH_running_time)))
    total_RT_minePH.append(np.mean(minePH_running_time))
    print("Mean DIST by minePH:" + str(np.mean(DIST_minePH)))
    total_DIST_minePH.append(np.mean(DIST_minePH))

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

    plt.plot(mine_pr, color="blue")
    plt.plot(mine_alpha, color="green")
    plt.show()
    plt.plot(pi, color="red")
    plt.plot(pr_min, color="yellow")
    plt.show()

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

print("Overall result for minePH:")
print("Average Drift Detected: ", str(np.mean(total_D_minePH)))
print("Minimum Drift Detected: ", str(np.min(total_D_minePH)))
print("Maximum Drift Detected: ", str(np.max(total_D_minePH)))
print("Drift Detected Standard Deviation: ", str(np.std(total_D_minePH)))

print("Average TP: ", str(np.mean(total_TP_minePH)))
print("Minimum TP: ", str(np.min(total_TP_minePH)))
print("Maximum TP: ", str(np.max(total_TP_minePH)))
print("TP Standard Deviation: ", str(np.std(total_TP_minePH)))

print("Average FP: ", str(np.mean(total_FP_minePH)))
print("Minimum FP: ", str(np.min(total_FP_minePH)))
print("Maximum FP: ", str(np.max(total_FP_minePH)))
print("FP Standard Deviation: ", str(np.std(total_FP_minePH)))

print("Average RT: ", str(np.mean(total_RT_minePH)))
print("Minimum RT: ", str(np.min(total_RT_minePH)))
print("Maximum RT: ", str(np.max(total_RT_minePH)))
print("RT Standard Deviation: ", str(np.std(total_RT_minePH)))

print("Average DIST: ", str(np.mean(total_DIST_minePH)))
print("Minimum DIST: ", str(np.min(total_DIST_minePH)))
print("Maximum DIST: ", str(np.max(total_DIST_minePH)))
print("DIST Standard Deviation: ", str(np.std(total_DIST_minePH)))

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