from operator import mul
from fractions import Fraction
from functools import reduce
import math
from skmultiflow.drift_detection import DDM
import numpy as np
from scipy.io import arff
import pandas as pd
from skmultiflow.data import DataStream
from sklearn.tree import DecisionTreeClassifier



# Global variable
DEFAULT_PR = 0
SAMPLE_SIZE = 10000
GLOBAL_RATE = 1


def nCk(n, k):
    return int(reduce(mul, (Fraction(n-i, i+1) for i in range(k)), 1))


def calculate_pr(ove, spe, n=1, x=1):
    if ove == 1:
        return DEFAULT_PR
    if spe == 0:
        return DEFAULT_PR
    else:
        return nCk(spe, x) * nCk(ove-spe, n-x) / nCk(ove, n)

elec_data = arff.loadarff('../data/repository/elec.arff')

elec_df = pd.DataFrame(elec_data[0])
elec_df = elec_df.astype({'day':'float'})
mapping = {"class":{b"UP":0, b"DOWN":1}}
elec_df.replace(mapping, inplace=True)

elec_stream = DataStream(elec_df, name="elec")
elec_stream.prepare_for_use()

X_train, y_train = elec_stream.next_sample(100)

clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

pr_global = DEFAULT_PR # Probability with cumulative samples
pr_local = DEFAULT_PR # Probability with samples within drifts
n_global = 0 # Cumulative Number of observations
n_local = 0 # Number of observations
d_global = 0 # Number of detected drifts
warning = 0
dist = 0

ddm = DDM()
while elec_stream.has_more_samples():
    n_global += 1
    n_local += 1

    X_test, y_test = elec_stream.next_sample()
    y_predict = clf.predict(X_test)

    ddm.add_element(y_test != y_predict)
    if ddm.detected_warning_zone():
        print('Warning zone has been detected at n: ' + str(n_global) + ' - of x: ' + str(X_test))
        warning += 1
    if ddm.detected_change():
        d_global += 1
        n_local = 0
        print('Change has been detected at n: ' + str(n_global) + ' - of x: ' + str(X_test))
print("Number of warning detected: " + str(warning))
print("Number of drifts detected: " + str(d_global))
