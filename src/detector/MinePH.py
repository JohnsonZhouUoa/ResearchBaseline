from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
import numpy as np
import math
from operator import mul
from fractions import Fraction
from functools import reduce
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd

class MinePageHinkley(BaseDriftDetector):
    """ Page-Hinkley method for concept drift detection.

    Notes
    -----
    This change detection method works by computing the observed
    values and their mean up to the current moment. Page-Hinkley
    won't output warning zone warnings, only change detections.
    The method works by means of the Page-Hinkley test [1]_. In general
    lines it will detect a concept drift if the observed mean at
    some instant is greater then a threshold value lambda.

    References
    ----------
    .. [1] E. S. Page. 1954. Continuous Inspection Schemes.
       Biometrika 41, 1/2 (1954), 100–115.

    Parameters
    ----------
    min_instances: int (default=30)
        The minimum number of instances before detecting change.
    delta: float (default=0.005)
        The delta factor for the Page Hinkley test.
    threshold: int (default=50)
        The change detection threshold (lambda).
    alpha: float (default=1 - 0.0001)
        The forgetting factor, used to weight the observed value
        and the mean.

    Examples
    --------
    >>> # Imports
    >>> import numpy as np
    >>> from skmultiflow.drift_detection import PageHinkley
    >>> ph = PageHinkley()
    >>> # Simulating a data stream as a normal distribution of 1's and 0's
    >>> data_stream = np.random.randint(2, size=2000)
    >>> # Changing the data concept from index 999 to 2000
    >>> for i in range(999, 2000):
    ...     data_stream[i] = np.random.randint(4, high=8)
    >>> # Adding stream elements to the PageHinkley drift detector and verifying if drift occurred
    >>> for i in range(2000):
    ...     ph.add_element(data_stream[i])
    ...     if ph.detected_change():
    ...         print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))

    """

    def __init__(self, min_instances=30, delta=0.005, threshold=50, alpha=1 - 0.0001, default_prob=1,actuals = [], initial_training = 0):
        super().__init__()
        self.min_instances = min_instances
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.x_mean = None
        self.sample_count = None
        self.global_sample_count = initial_training + 1
        self.sum = None
        self.default_prob = default_prob
        self.global_prob = default_prob
        self.local_prob = default_prob
        self.drift_detected = False
        self.drift_count = 0
        self.drift_ts = []
        self.diff = -1
        self.ts_prediction = -1
        self.actuals = dict.fromkeys(actuals, 0)
        self.reset()

    def reset(self):
        """ reset

        Resets the change detector parameters.

        """
        super().reset()
        self.sample_count = 1
        self.x_mean = 0.0
        self.global_ratio = 0
        self.sum = 0.0

    def add_element(self, x):
        """ Add a new element to the statistics

        Parameters
        ----------
        x: numeric value
            The observed value, from which we want to detect the
            concept change.

        Notes
        -----
        After calling this method, to verify if change was detected, one
        should call the super method detected_change, which returns True
        if concept drift was detected and False otherwise.

        """
        if self.in_concept_change:
            self.reset()

        if (self.drift_detected):
            if (len(self.drift_ts) > 14):
                # TRUE POSITIVE detected
                ts = np.diff(self.drift_ts)
                temp_df = pd.DataFrame(
                    {'timestamp': pd.date_range('2000-01-01', periods=len(ts), freq='D'),
                     'ts': ts})
                temp_df.set_index('timestamp', inplace=True)
                hw_model = ExponentialSmoothing(temp_df, trend='add', seasonal='add').fit(optimized=True)
                predict_result = np.array(hw_model.predict(len(self.drift_ts), len(self.drift_ts)))[0]
                #print("Predicted:")
                #print(predict_result)
                self.diff = predict_result
                self.ts_prediction = self.global_sample_count + self.diff
                self.drift_ts.pop(0)
            else:
                # FALSE POSITIVE detected
                # Wrong prediction from TS Model
                self.diff = self.ts_prediction - self.global_sample_count
            self.drift_detected = False

        if self.diff > 0:
            if self.sample_count <= self.diff:
                self.global_ratio = self.sample_count / self.diff
            else:
                # DDM miss it
                self.global_ratio = 1 - (self.sample_count - self.diff) / self.sample_count
        else:
            self.global_ratio = 0

        self.x_mean = self.x_mean + (x - self.x_mean) / float(self.sample_count)
        global_prob = self.calculate_pr(self.global_sample_count, self.drift_count)
        local_prob = self.calculate_pr(self.sample_count, 1)
        pr = self.sigmoid_transformation(self.global_ratio * global_prob + (1 - self.global_ratio) * local_prob)

        if pr < self.delta and self.global_ratio > 0.5:
            self.sum = max(0., self.alpha * self.sum + (x - self.x_mean - pr))
        else:
            self.sum = max(0., self.alpha * self.sum + (x - self.x_mean - self.delta))

        self.sample_count += 1
        self.global_sample_count += 1

        self.estimation = self.x_mean
        self.in_concept_change = False
        self.in_warning_zone = False

        self.delay = 0

        if self.sample_count < self.min_instances:
            return None

        if self.sum > self.threshold:
            self.in_concept_change = True
            self.learn_ts = True
            self.drift_count += 1
            key = max([i for i in self.actuals.keys() if i <= self.global_sample_count])
            if key != 0 and self.actuals[key] == 0:
                # print("A true positive detected at " + str(self.global_sample_count))
                # print("TP: " + str(key))
                self.drift_ts.append(self.global_sample_count)
            else:
                print("A false positive in PH detected at " + str(self.global_sample_count))
            self.actuals[key] += 1
            self.drift_detected = True

    def nCk(self, n, k):
        return int(reduce(mul, (Fraction(n - i, i + 1) for i in range(k)), 1))

    def calculate_pr(self, ove, spe, n=1, x=1):
        if ove == 1:
            return self.default_prob
        if spe == 0:
            return self.default_prob
        else:
            return self.nCk(spe, x) * self.nCk(ove - spe, n - x) / self.nCk(ove, n)

    def activation_function(self, pr):
        return (math.sqrt(pr) / ((1 - self.global_ratio) + math.sqrt(pr)) + (1 - self.global_ratio))
        #return (math.exp(pr)/(1+math.exp(pr)))

    def sigmoid_transformation(self, p):
        return 1 / (1 + math.exp(-p))

    def get_mean(self):
        return self.x_mean

    def get_sum(self):
        return self.sum

    def get_threshold(self):
        return self.threshold
