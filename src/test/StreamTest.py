from skmultiflow.evaluation import EvaluatePrequential
from skika.data.reccurring_concept_stream import RCStreamType, RecurringConceptStream, conceptOccurence, \
    RecurringConceptGradualStream
import numpy as np
import random
import collections
from skmultiflow.data.data_stream import DataStream
from skmultiflow.trees import HoeffdingTreeClassifier,HoeffdingAdaptiveTreeClassifier, ExtremelyFastDecisionTreeClassifier

# Global variable
TRAINING_SIZE = 200
STREAM_SIZE = 500000
DRIFT_INTERVALS = [3000, 5000]
concepts = [0, 1, 2]
RANDOMNESS = 50

actuals = [0]
concept_chain = {0: 0}
current_concept = 0
for i in range(1, STREAM_SIZE):
    # if i in drift_points:
    for j in DRIFT_INTERVALS:
        if i % j == 0:
            randomness = random.randint(0, RANDOMNESS)
            d = i + ((randomness * 1) if (random.randint(0, 1) > 0) else (randomness * -1))
            if d not in concept_chain.keys():
                concept_index = random.randint(0, len(concepts) - 1)
                while concepts[concept_index] == current_concept:
                    concept_index = random.randint(0, len(concepts) - 1)
                concept = concepts[concept_index]
                concept_chain[d] = concept
                actuals.append(d)
                current_concept = concept

x = collections.Counter(concept_chain.values())
# print(x)

concept_0 = conceptOccurence(id=0, difficulty=2, noise=0,
                                 appearences=x[0], examples_per_appearence=max(DRIFT_INTERVALS))
concept_1 = conceptOccurence(id=1, difficulty=3, noise=0,
                                 appearences=x[1], examples_per_appearence=max(DRIFT_INTERVALS))
concept_2 = conceptOccurence(id=2, difficulty=2, noise=0,
                                 appearences=x[2], examples_per_appearence=max(DRIFT_INTERVALS))
desc = {0: concept_0, 1: concept_1, 2: concept_2}

datastream = RecurringConceptStream(
        rctype=RCStreamType.SEA,
        num_samples=STREAM_SIZE,
        noise=0,
        concept_chain=concept_chain,
        desc=desc,
        boost_first_occurance=False)

    # X_train, y_train = datastream.next_sample(TRAINING_SIZE)
X_train = []
y_train = []
for i in range(STREAM_SIZE):
    X, y = datastream.next_sample()
    X_train.append(X[0])
    y_train.append(y[0])

X_train = np.array(X_train)
y_train = np.array(y_train)

stream = DataStream(X_train, y_train)

ht = HoeffdingTreeClassifier()

evaluator = EvaluatePrequential(show_plot=True,
                                    pretrain_size=TRAINING_SIZE,
                                    max_samples=STREAM_SIZE)
evaluator.evaluate(stream=stream, model=ht)
