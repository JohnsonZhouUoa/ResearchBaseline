from skika.data.reccurring_concept_stream import RCStreamType, RecurringConceptStream, conceptOccurence, RecurringConceptGradualStream
import random
import collections
import arff
import numpy as np
import pandas


RAINING_SIZE = 1
STREAM_SIZE = 10000000
grace = 2000
DRIFT_INTERVALS = [10000, 25000, 35000]
concepts = [0, 1, 2]
RANDOMNESS = 0
seeds = [6976, 2632, 2754, 5541, 3681, 1456, 7041, 328, 5337, 4622,
         2757, 1788, 3399, 4639, 5306, 5742, 3015, 1554, 8548, 1313,
         4738, 9458, 8145, 3624, 1913, 1654, 2988, 2031, 1802, 4338]
ignore = 0
random.seed(6976)


for k in range(0, 30):
    seed = seeds[k]#random.randint(0, 10000)
    #seeds.append(seed)
    keys = []
    actuals = [0]
    concept_chain = {0:0}
    current_concept = 0
    for i in range(1,STREAM_SIZE+1):
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
        rctype=RCStreamType.SINE,
        num_samples=STREAM_SIZE,
        noise=0.2,
        concept_chain=concept_chain,
        seed=seed,
        desc=desc,
        boost_first_occurance=False)

    #X_train, y_train = datastream.next_sample(TRAINING_SIZE)
    X_train = []
    y_train = []
    for i in range(0, STREAM_SIZE):
        X, y = datastream.next_sample()
        X_train.append(X[0])
        y_train.append(y[0])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    df = pandas.concat([pandas.DataFrame(X_train, columns=['x' + str(i) for i in range(1, X_train.shape[1]+1)]), pandas.DataFrame(y_train, columns=['y'])], axis=1)
    arff.dump('/hdd2/SINE_2NOISE/SINE' + str(seed) + '.arff', df.values, relation='myrel', names=df.columns)