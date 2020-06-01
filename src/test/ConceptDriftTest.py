from skmultiflow.data import ConceptDriftStream
from skmultiflow.data import HyperplaneGenerator
from skmultiflow.data import DataStream


BATCH_SIZE = 100

stream = HyperplaneGenerator(n_features=2)
stream.prepare_for_use()

X, y = stream.next_sample(BATCH_SIZE)

stream1 = DataStream(X, y, name="test1")
stream2 = DataStream(X, y, name="test2")
stream1.prepare_for_use()
stream2.prepare_for_use()


drift_stream = ConceptDriftStream(
    stream=stream1,
    drift_stream=stream2
)

drift_stream.prepare_for_use()

X, y = drift_stream.next_sample(BATCH_SIZE)





