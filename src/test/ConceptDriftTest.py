from skmultiflow.data import ConceptDriftStream
from skmultiflow.data import HyperplaneGenerator
from skmultiflow.data import DataStream
import shutil
from autosklearn.classification import AutoSklearnClassifier


BATCH_SIZE = 100
TMP_FOLDER = "/tmp/autosklearn_example_tmp"
OUT_FOLER = "/tmp/autosklearn_example_out"


for dir in [TMP_FOLDER, OUT_FOLER]:
    try:
        shutil.rmtree(dir)
    except OSError as e:
        print(e)

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

automl = AutoSklearnClassifier(time_left_for_this_task=60,
                               per_run_time_limit=15,
                               ml_memory_limit=1024,
                               shared_mode=True,
                               ensemble_size=0,
                               tmp_folder=TMP_FOLDER,
                               output_folder=OUT_FOLER,
                               initial_configurations_via_metalearning=0,
                               seed=1)

#TODO: open issues 856 and 868 on github
automl.fit(X, y)




