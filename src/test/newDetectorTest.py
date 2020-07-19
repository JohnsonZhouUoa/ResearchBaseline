from operator import mul
from fractions import Fraction
from functools import reduce
from sklearn import linear_model
import math
from src.test.DiffDDM import DiffDDM
import numpy as np


# Global variable
DEFAULT_PR = 0.5
SAMPLE_SIZE = 10000
GLOBAL_RATE = 0.5


def nCk(n, k):
    return int(reduce(mul, (Fraction(n-i, i+1) for i in range(k)), 1))


def calculate_pr(ove, spe, n=1, x=1):
    if spe == 0:
        return DEFAULT_PR
    else:
        return nCk(spe, x) * nCk(ove-spe, n-x) / nCk(ove, n)


y_train = np.array([x * 2 + 1 for x in range(1, SAMPLE_SIZE+1)])

for i in range(50, SAMPLE_SIZE):
    if i % 50 == 0:
        for j in range(i, i+20):
            y_train[j] = j * 4 + 2

regr = linear_model.LinearRegression()

regr.fit(np.array([x for x in range(1, 21)]).reshape((-1, 1)), y_train[0:20])


pr_global = DEFAULT_PR # Probability with cumulative samples
pr_local = DEFAULT_PR # Probability with samples within drifts
n_global = 0 # Cumulative Number of observations
n_local = 0 # Number of observations
d_global = 0 # Number of detected drifts
warning = 0

ddm = DiffDDM()
for i in range(21, SAMPLE_SIZE):

    n_global += 1
    n_local += 1

    x_next = i

    y_next = y_train[x_next-1]
    y_predict = regr.predict(np.array([x_next]).reshape((-1, 1)))

    pr_global = calculate_pr(n_global, d_global)
    pr_local = calculate_pr(n_local, 1)

    pr = GLOBAL_RATE * pr_global + pr_local

    ddm.add_element(math.floor(y_next) != math.floor(y_predict), pr)
    if ddm.detected_warning_zone():
        print('Warning zone has been detected in data: ' + str(y_next))
        warning += 1
    if ddm.detected_change():
        d_global += 1
        n_local = 0
        print('Change has been detected in data: ' + str(y_next))
    #print("Accuracy Score: ", sklearn.metrics.accuracy_score(y_next, y_predict))
