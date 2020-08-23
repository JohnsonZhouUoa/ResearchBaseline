from operator import mul
from fractions import Fraction
from functools import reduce
from skmultiflow.drift_detection import DDM
from sklearn.tree import DecisionTreeClassifier
from skika.data.reccurring_concept_stream import RCStreamType, RecurringConceptStream, conceptOccurence
import time

start_time = time.time()

# Global variable
DEFAULT_PR = 1
GLOBAL_RATE = 0
STREAM_SIZE = 15000
DRIFT_INTERVAL = 100


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
    return

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


X_train, y_train = datastream.next_sample(100)

clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

pr_global = DEFAULT_PR # Probability with cumulative samples
pr_local = DEFAULT_PR # Probability with samples within drifts
n_global = 0 # Cumulative Number of observations
n_local = 0 # Number of observations
d_global = 0 # Number of detected drifts
warning = 0
TP = []
FP = []

ddm = DDM()
while datastream.has_more_samples():
    n_global += 1
    n_local += 1

    X_test, y_test = datastream.next_sample()
    y_predict = clf.predict(X_test)

    ddm.add_element(y_test != y_predict)
    if ddm.detected_warning_zone():
        #print('Warning zone has been detected at n: ' + str(n_global) + ' - of x: ' + str(X_test))
        warning += 1
    if ddm.detected_change():
        d_global += 1
        n_local = 0
        clf.fit(X_test, y_test)
        if (n_global // DRIFT_INTERVAL in TP):
            FP.append(n_global // DRIFT_INTERVAL)
        else:
            TP.append(n_global // DRIFT_INTERVAL)
        print('Change has been detected at n: ' + str(n_global) + ' - of x: ' + str(X_test))
print("Number of drifts detected: " + str(d_global))
print("TP:" + str(len(TP)))
print("FP:" + str(len(FP)))
print("Actual:" + str(STREAM_SIZE/DRIFT_INTERVAL))

print("--- %s seconds ---" % (time.time() - start_time))
