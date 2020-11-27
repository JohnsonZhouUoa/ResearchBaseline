from skika.data.reccurring_concept_stream import RCStreamType, RecurringConceptStream, conceptOccurence, RecurringConceptGradualStream
import random
import collections
import arff
import numpy as np
import pandas

DRIFT_INTERVALS = [5000]
concepts = [0, 1]
RANDOMNESS = 0
ignore = 0
num_samples = 500000
current = 0
concept_chain = {}

for i in range(0, num_samples):
    if i % DRIFT_INTERVALS[0] == 0:
        concept_chain[i] = current
        if current == 0:
            current = 1
        else:
            current = 0

print(concept_chain)

seed = 42 #random.randint(0, 10000)
print("seed: " + str(seed))

x = collections.Counter(concept_chain.values())

concept_0 = conceptOccurence(id=0, difficulty=2, noise=0,
                    appearences=x[0], examples_per_appearence=5000)

concept_1 = conceptOccurence(id=1, difficulty=3, noise=0,
                    appearences=x[1], examples_per_appearence=5000)
desc = {0: concept_0, 1: concept_1}

datastream = RecurringConceptStream(
                    rctype=RCStreamType.STAGGER,
                    num_samples=num_samples,
                    noise=0,
                    concept_chain=concept_chain,
                    seed=seed,
                    desc=desc,
                    boost_first_occurance=False)

    #X_train, y_train = datastream.next_sample(TRAINING_SIZE)
X_train = []
y_train = []
for i in range(ignore, ignore + num_samples):
    X, y = datastream.next_sample()
    X_train.append(X[0])
    y_train.append(y[0])

X_train = np.array(X_train)
y_train = np.array(y_train)

df = pandas.concat(
    [pandas.DataFrame(X_train, columns=['x' + str(i) for i in range(1, X_train.shape[1]+1)]),
    pandas.DataFrame(y_train, columns=['y'])], axis=1)


arff.dump('Original.arff', df.values, relation='myrel', names=df.columns)