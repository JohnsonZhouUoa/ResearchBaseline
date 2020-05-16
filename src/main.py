import multiprocessing
import shutil
import sklearn.datasets
from thirdparty.tornado.drift_detection import FHDDM
from autosklearn.classification import AutoSklearnClassifier

# GLobal variable
BATCH_SIZE = 10
TMP_FOLDER = "/tmp/autosklearn_parallel_example_tmp"
OUT_FOLER = "/tmp/autosklearn_parallel_example_out"


def spawn_classifier(seed, dataset_name, X_train, y_train):
    """Spawn a subprocess.
    auto-sklearn does not take care of spawning worker processes. This
    function, which is called several times in the main block is a new
    process which runs one instance of auto-sklearn.
    """

    # Use the initial configurations from meta-learning only in one out of
    # the four processes spawned. This prevents auto-sklearn from evaluating
    # the same configurations in four processes.
    if seed == 0:
        initial_configurations_via_metalearning = 25
    else:
        initial_configurations_via_metalearning = 0

    # Arguments which are different to other runs of auto-sklearn:
    # 1. all classifiers write to the same output directory
    # 2. shared_mode is set to True, this enables sharing of data between
    # models.
    # 3. all instances of the AutoSklearnClassifier must have a different seed!
    automl = AutoSklearnClassifier(
        time_left_for_this_task=60,  # sec., how long should this seed fit
        # process run
        per_run_time_limit=15,  # sec., each model may only take this long before it's killed
        ml_memory_limit=1024,  # MB, memory limit imposed on each call to a ML algorithm
        shared_mode=True,  # tmp folder will be shared between seeds
        tmp_folder=TMP_FOLDER,
        output_folder=OUT_FOLER,
        delete_tmp_folder_after_terminate=False,
        ensemble_size=0,  # ensembles will be built when all optimization runs are finished
        initial_configurations_via_metalearning=initial_configurations_via_metalearning,
        seed=seed)
    automl.fit(X_train, y_train)


for dir in [TMP_FOLDER, OUT_FOLER]:
    try:
        shutil.rmtree(dir)
    except OSError as e:
        print(e)

X, y = sklearn.datasets.load_digits(return_X_y=True)

print('Starting to build the initial classifier!')
automl = AutoSklearnClassifier(time_left_for_this_task=60,
                               per_run_time_limit=15,
                               ml_memory_limit=1024,
                               shared_mode=True,
                               ensemble_size=50,
                               ensemble_nbest=200,
                               tmp_folder=TMP_FOLDER,
                               output_folder=OUT_FOLER,
                               initial_configurations_via_metalearning=0,
                               seed=1)

automl.fit(X[0:BATCH_SIZE-1], y[0:BATCH_SIZE-1])
print("The initial model is: ")
print(automl.show_models())

i = BATCH_SIZE
adapt = False
fhddm = FHDDM()
while i < len(X):
    if (i + 2 * BATCH_SIZE - 1) > len(X):
        break
    X_next = X[i + BATCH_SIZE:i + 2 * BATCH_SIZE - 1]
    y_next = y[i + BATCH_SIZE:i + 2 * BATCH_SIZE - 1]
    y_predict = automl.predict(X_next)
    warning_status, drift_status = fhddm.detect([y_next, y_predict])
    if warning_status:
        print("Warning at " + str(i))
    if drift_status:
        print("Drift at " + str(i))
        automl = AutoSklearnClassifier(time_left_for_this_task=30,
                               per_run_time_limit=15,
                               ml_memory_limit=1024,
                               shared_mode=True,
                               ensemble_size=50,
                               ensemble_nbest=200,
                               tmp_folder=TMP_FOLDER,
                               output_folder=OUT_FOLER,
                               initial_configurations_via_metalearning=25,
                               seed=1)
    i += BATCH_SIZE
    print("An iteration finished, the next i is: " + str(i))
