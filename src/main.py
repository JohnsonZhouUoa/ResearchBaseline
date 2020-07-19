import multiprocessing
import shutil
import sklearn.datasets
from thirdparty.tornado.drift_detection import FHDDM
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.constants import *
from autosklearn.metrics import accuracy
from skmultiflow.drift_detection import DDM
from skmultiflow.data import ConceptDriftStream
from skmultiflow.data import HyperplaneGenerator
from skmultiflow.data import DataStream
from sklearn import metrics

# GLobal variable
SAMPLE_SIZE = 5000
BATCH_SIZE = 10
TMP_FOLDER = "/tmp/autosklearn_parallel_example_tmp"
OUT_FOLER = "/tmp/autosklearn_parallel_example_out"


# def spawn_classifier(seed, X_train, y_train):
#     """Spawn a subprocess.
#     auto-sklearn does not take care of spawning worker processes. This
#     function, which is called several times in the main block is a new
#     process which runs one instance of auto-sklearn.
#     """
#
#     # Use the initial configurations from meta-learning only in one out of
#     # the four processes spawned. This prevents auto-sklearn from evaluating
#     # the same configurations in four processes.
#     if seed == 0:
#         initial_configurations_via_metalearning = 25
#     else:
#         initial_configurations_via_metalearning = 0
#
#     # Arguments which are different to other runs of auto-sklearn:
#     # 1. all classifiers write to the same output directory
#     # 2. shared_mode is set to True, this enables sharing of data between
#     # models.
#     # 3. all instances of the AutoSklearnClassifier must have a different seed!
#     automl = AutoSklearnClassifier(
#         time_left_for_this_task=60,  # sec., how long should this seed fit
#         # process run
#         per_run_time_limit=15,  # sec., each model may only take this long before it's killed
#         ml_memory_limit=1024,  # MB, memory limit imposed on each call to a ML algorithm
#         shared_mode=True,  # tmp folder will be shared between seeds
#         tmp_folder=TMP_FOLDER,
#         output_folder=OUT_FOLER,
#         delete_tmp_folder_after_terminate=False,
#         ensemble_size=0,  # ensembles will be built when all optimization runs are finished
#         initial_configurations_via_metalearning=initial_configurations_via_metalearning,
#         seed=seed)
#     automl.fit(X_train, y_train)


for dir in [TMP_FOLDER, OUT_FOLER]:
    try:
        shutil.rmtree(dir)
    except OSError as e:
        print(e)

stream = HyperplaneGenerator(n_features=2)
stream.prepare_for_use()

X_t, y_t = stream.next_sample(SAMPLE_SIZE)

stream1 = DataStream(X_t, y_t, name="test1")
stream2 = DataStream(X_t, y_t, name="test2")
stream1.prepare_for_use()
stream2.prepare_for_use()


drift_stream = ConceptDriftStream(
    stream=stream1,
    drift_stream=stream2
)

drift_stream.prepare_for_use()

X, y = drift_stream.next_sample(BATCH_SIZE)

print('Starting to build the initial classifier!')

# processes = []
# for i in range(4):  # set this at roughly half of your cores
#     p = multiprocessing.Process(target=spawn_classifier, args=(i, X, y))
#     p.start()
#     processes.append(p)
# for p in processes:
#     p.join()

automl = AutoSklearnClassifier(time_left_for_this_task=60,
                               per_run_time_limit=15,
                               ml_memory_limit=1024,
                               shared_mode=True,
                               ensemble_size=2,
                               tmp_folder=TMP_FOLDER,
                               output_folder=OUT_FOLER,
                               initial_configurations_via_metalearning=0,
                               seed=1)

#TODO: open issues 856 and 868 on github
automl.fit(X, y)
# automl.fit_ensemble(y,
#                     task=MULTICLASS_CLASSIFICATION,
#                     metric=accuracy,
#                     precision='32',
#                     dataset_name='digits',
#                     ensemble_size=20,
#                     ensemble_nbest=50)
# print("The initial model is: ")
# print(automl.show_models())


#fhddm = FHDDM()
ddm = DDM()
while drift_stream.has_more_samples():
    X_next, y_next = drift_stream.next_sample(BATCH_SIZE)
    y_predict = automl.predict(X_next)

    if len(y_next) != len(y_predict):
        print("Predictions length not correct")
        break
    for j in range(len(y_next)):
        ddm.add_element(y_next[j] != y_predict[j])
        if ddm.detected_warning_zone():
            print('Warning zone has been detected in data: ' + str(y_next[j]) + ' - of index: ' + str(j))
        if ddm.detected_change():
            print('Change has been detected in data: ' + str(y_next[j]) + ' - of index: ' + str(j))
            automl = AutoSklearnClassifier(time_left_for_this_task=60,
                                           per_run_time_limit=15,
                                           ml_memory_limit=4096,
                                           shared_mode=True,
                                           ensemble_size=0,
                                           tmp_folder=TMP_FOLDER,
                                           output_folder=OUT_FOLER,
                                           initial_configurations_via_metalearning=0,
                                           seed=1)
            automl.fit(X_next, y_next)
    # warning_status, drift_status = fhddm.detect(corr)
    # if warning_status:
    # #if ddm.detected_warning_zone():
    #     print("Warning at " + str(i))
    # if drift_status:
    # #if ddm.detected_change():
    #     print("Drift at " + str(i))
        # fhddm.reset()
    print("Accuracy Score: ", sklearn.metrics.accuracy_score(y_next, y_predict))
