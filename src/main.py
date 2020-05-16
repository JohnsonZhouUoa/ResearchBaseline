import multiprocessing
import shutil
import sklearn.datasets
from thirdparty.tornado.drift_detection import FHDDM

# GLobal variable
BATCH_SIZE = 10
TMP_FOLDER = "/tmp/autosklearn_parallel_example_tmp"
OUT_FOLER = "/tmp/autosklearn_parallel_example_out"


for dir in [TMP_FOLDER, OUT_FOLER]:
    try:
        shutil.rmtree(dir)
    except OSError as e:
        print(e)

X, y = sklearn.datasets.load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(X, y, random_state=1)
