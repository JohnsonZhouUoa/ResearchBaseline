from skika.data.reccurring_concept_stream import RCStreamType, RecurringConceptStream, conceptOccurence, RecurringConceptGradualStream
import random
import collections
import arff
import numpy as np
import pandas


TRAINING_SIZE = 500000
STREAM_SIZE = 500000
DRIFT_INTERVALS = [3000, 5000]
concepts = [0, 1]
RANDOMNESS = 0
ignore = 0


seed = 1235#random.randint(0, 10000)
print("seed: " + str(seed))
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
#print(x)

concept_0 = conceptOccurence(id = 0, difficulty = 6, noise = 0,
                            appearences = x[0], examples_per_appearence = max(DRIFT_INTERVALS))
concept_1 = conceptOccurence(id = 1, difficulty = 6, noise = 0,
                        appearences = x[1], examples_per_appearence = max(DRIFT_INTERVALS))
# concept_2 = conceptOccurence(id=2, difficulty=2, noise=0,
#                                  appearences=x[2], examples_per_appearence=max(DRIFT_INTERVALS))
desc = {0: concept_0, 1: concept_1}

datastream = RecurringConceptStream(
        rctype=RCStreamType.SEA,
        num_samples=STREAM_SIZE,
        noise=0.3,
        concept_chain=concept_chain,
        seed=seed,
        desc=desc,
        boost_first_occurance=False)

    #X_train, y_train = datastream.next_sample(TRAINING_SIZE)
X_train = []
y_train = []
for i in range(ignore, ignore + TRAINING_SIZE):
    X, y = datastream.next_sample()
    X_train.append(X[0])
    y_train.append(y[0])

X_train = np.array(X_train)
y_train = np.array(y_train)

df = pandas.concat(
    [pandas.DataFrame(X_train, columns=['x' + str(i) for i in range(1, X_train.shape[1]+1)]),
    pandas.DataFrame(y_train, columns=['y'])], axis=1)


arff.dump('SEA.arff', df.values, relation='myrel', names=df.columns)