from operator import mul
from fractions import Fraction
from functools import reduce
import math
from src.test.DiffDDM import DiffDDM
from sklearn.tree import DecisionTreeClassifier
from skika.data.reccurring_concept_stream import RCStreamType, RecurringConceptStream, conceptOccurence
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd
import numpy as np
import warnings
import time


start_time = time.time()

warnings.filterwarnings('ignore')
plt.style.use("seaborn-whitegrid")

# Global variable
DEFAULT_PR = 0.5
GLOBAL_RATE = 0
INITAL_TRAINING = 100
PREDICT_SIZE = 10000
STREAM_SIZE = 1500000
DRIFT_INTERVAL = 5000


def nCk(n, k):
    return int(reduce(mul, (Fraction(n-i, i+1) for i in range(k)), 1))


def calculate_pr(ove, spe, n=1, x=1):
    if ove == 1:
        return DEFAULT_PR
    if spe == 0:
        return DEFAULT_PR
    else:
        return nCk(spe, x) * nCk(ove-spe, n-x) / nCk(ove, n)


def sigmoid_transformation(pr):
    #return 1/(1+math.exp(-pr))
    return math.exp(pr)/(GLOBAL_RATE+math.exp(pr))


def gaussian_transformation(x):
    return math.log(1+x)


concept_chain = {0:0}
concept = 0
for i in range(1,STREAM_SIZE):
    if i % DRIFT_INTERVAL == 0:
        if concept == 0:
            concept_chain[i] = 1
            concept = 1
        else:
            concept_chain[i] = 0
            concept = 0

concept_0 = conceptOccurence(id = 0, difficulty = 2, noise = 0,
                            appearences = 2, examples_per_appearence = 5000)
concept_1 = conceptOccurence(id = 1, difficulty = 3, noise = 0,
                        appearences = 1, examples_per_appearence = 5000)
desc = {0: concept_0, 1: concept_1}

datastream = RecurringConceptStream(
                        rctype = RCStreamType.STAGGER,
                        num_samples =STREAM_SIZE,
                        noise = 0,
                        concept_chain = concept_chain,
                        seed = 42,
                        desc = desc,
                        boost_first_occurance = False)


X_train, y_train = datastream.next_sample(INITAL_TRAINING)

clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)





pr_global = DEFAULT_PR # Probability with cumulative samples
pr_local = DEFAULT_PR # Probability with samples within drifts
n_global = INITAL_TRAINING # Cumulative Number of observations
n_local = 0 # Number of observations
d_global = 0 # Number of detected drifts
warning = 0
#pr_hist = []
ts = []
re_predict = False
pred_results = []
TP = []
FP = []

ddm = DiffDDM()
while datastream.has_more_samples():
    n_global += 1
    n_local += 1

    if(re_predict):
        time = pd.date_range('2000-01-01', periods=len(ts),
                           freq='H')
        df = pd.DataFrame({'time': time, 'ts': ts})
        df.set_index('time', inplace=True)
        hw_model = ExponentialSmoothing(df, trend='add', seasonal='add').fit(optimized=True)
        pred = np.array(hw_model.predict(len(ts), len(ts) + PREDICT_SIZE))
        pred_res = np.argwhere(pred > 1)
        if(len(pred_res) != 0):
            pred_results.append(pred_res[0])
        else:
            pred_results.append(0.01)
        re_predict = False

    if d_global > 0:
        pred_next = pred_results[d_global-1]
        radio = n_local / pred_next
        if (radio < 1):
            GLOBAL_RATE = radio

    X_test, y_test = datastream.next_sample()
    y_predict = clf.predict(X_test)

    pr_global = calculate_pr(n_global, d_global)
    pr_local = calculate_pr(n_local, 1)

    pr = sigmoid_transformation(GLOBAL_RATE * pr_global + (1 - GLOBAL_RATE) * pr_local)

    #pr_hist.append(pr)
    ddm.add_element(y_test != y_predict, pr)
    if ddm.detected_warning_zone():
        #print('Warning zone has been detected at n: ' + str(n_global) + ' - of x: ' + str(X_test))
        warning += 1
    if ddm.detected_change():
        d_global += 1
        n_local = 0
        clf.fit(X_test, y_test)
        ts.append(1)
        re_predict = True
        GLOBAL_RATE = 0
        if(n_global // DRIFT_INTERVAL in TP):
            FP.append(n_global // DRIFT_INTERVAL)
        else:
            TP.append(n_global // DRIFT_INTERVAL)
        print('Change has been detected at n: ' + str(n_global) + ' - of x: ' + str(X_test))
    else:
        ts.append(0)
print("Number of drifts detected: " + str(d_global))
print("TP:" + str(len(TP)))
print("FP:" + str(len(FP)))
print("Actual:" + str(STREAM_SIZE/DRIFT_INTERVAL))
#plt.plot(pr_hist)
#plt.show()

print("--- %s seconds ---" % (time.time() - start_time))

