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
import arff
import pandas
from skmultiflow.data.data_stream import DataStream
from sklearn.preprocessing import LabelEncoder




warnings.filterwarnings('ignore')
plt.style.use("seaborn-whitegrid")

# Global variable
TRAINING_SIZE = 1
STREAM_SIZE = 500000
DRIFT_INTERVALS = [2000, 3000, 5000]
concepts = [0, 1, 2, 3]
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
seeds = [334, 9719, 8360, 6917, 4721]
ignore = 0
random.seed(249)

data_1 = arff.load("SINE1.arff")
df_1 = pandas.DataFrame(data_1)
df_1 = df_1.iloc[1:, ]
data_2 = arff.load("SINE2.arff")
df_2 = pandas.DataFrame(data_2)
df_2 = df_2.iloc[1:, ]
data_3 = arff.load("SINE3.arff")
df_3 = pandas.DataFrame(data_3)
df_3 = df_3.iloc[1:, ]
data_4 = arff.load("SINE4.arff")
df_4 = pandas.DataFrame(data_4)
df_4 = df_4.iloc[1:, ]

concept_chain = {}
current_concept = 0
stream_size = 500000


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
    num_of_instance = 0
    data_1_current = 0
    data_2_current = 0
    data_3_current = 0
    data_4_current = 0
    datastream = None

    for i in range(len(actuals)):
        new_data = []
        if actuals[i] == actuals[-1]:
            num_of_instance = stream_size - actuals[i]
        else:
            num_of_instance = actuals[i + 1] - actuals[i]
        new_concept = concept_chain[actuals[i]]
        if new_concept == 0:
            new_data = df_1.iloc[data_1_current:(data_1_current + num_of_instance), ]
            data_1_current = data_1_current + num_of_instance
        elif new_concept == 1:
            new_data = df_2.iloc[data_2_current:(data_2_current + num_of_instance), ]
            data_2_current = data_2_current + num_of_instance
        elif new_concept == 2:
            new_data = df_3.iloc[data_3_current:(data_3_current + num_of_instance), ]
            data_3_current = data_3_current + num_of_instance
        elif new_concept == 3:
            new_data = df_4.iloc[data_4_current:(data_4_current + num_of_instance), ]
            data_4_current = data_4_current + num_of_instance
        if datastream is None:
            datastream = new_data
        else:
            datastream = datastream.append(new_data)

    labelEncoder = LabelEncoder()
    datastream.iloc[:,-1] = labelEncoder.fit_transform(datastream.iloc[:,-1])
    #X_train, y_train = datastream.next_sample(TRAINING_SIZE)
    datastream = DataStream(datastream)
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

    ht = HoeffdingAdaptiveTreeClassifier()

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

    ddm = DDM()
    while datastream.has_more_samples():
        n_global += 1

        X_test, y_test = datastream.next_sample()
        y_predict = ht.predict(X_test)

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
            ht = HoeffdingTreeClassifier()
        ht.partial_fit(X_test, y_test)

    print("Round " + str(k+1) + " out of 30 rounds")
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

print("Overall result:")
print("Stream size: " + str(STREAM_SIZE))
print("Drift intervals: " + str(DRIFT_INTERVALS))
print("Actual drifts:" + str(len(actuals)))
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