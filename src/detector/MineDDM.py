import numpy as np
import math
from operator import mul
from fractions import Fraction
from functools import reduce
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd


class MineDDM(BaseDriftDetector):

    def __init__(self, min_num_instances=30, warning_level=2.0, out_control_level=3.0,
                 default_prob=1, ts_length=20, confidence=0.95):
        super().__init__()
        self.sample_count = 1
        self.global_ratio = 1.0
        self.pr = 0
        self.std = 0
        self.miss_prob = None
        self.miss_std = None
        self.miss_prob_sd_min = None
        self.miss_prob_min = None
        self.miss_sd_min = None
        self.global_prob = default_prob
        self.local_prob = default_prob
        self.min_instances = min_num_instances
        self.warning_level = warning_level
        self.out_control_level = out_control_level
        self.default_prob = default_prob
        self.drift_ts = []
        self.reset()
        self.diff = -1
        self.ts_prediction = -1
        self.TP_detected = False
        self.FP_detected = False
        self.ts_length = ts_length
        self.period = 1
        self.confidence = confidence

    def reset(self):
        """ reset

        Resets the change detector parameters.

        """
        super().reset()
        self.sample_count = 1
        self.miss_prob = 1.0
        self.miss_std = 0.0
        self.local_prob = self.default_prob
        self.miss_prob_sd_min = float("inf")
        self.miss_prob_min = float("inf")
        self.miss_sd_min = float("inf")
        self.global_ratio = 0
        self.pr = 1
        self.std = 0


    def add_element(self, prediction, n):

        if self.in_concept_change:
            self.reset()

        self.miss_prob = self.miss_prob + (prediction - self.miss_prob) / float(self.sample_count)
        self.miss_std = np.sqrt(self.miss_prob * (1 - self.miss_prob) / float(self.sample_count))
        if (len(self.drift_ts) >= self.ts_length):
            self.global_prob = self.calculate_pr(n - self.drift_ts[0], len(self.drift_ts))
        else:
            self.global_prob = 1
        self.local_prob = self.calculate_pr(self.sample_count, 1)
        self.pr = self.activation_function(self.global_ratio * self.global_prob + (1 - self.global_ratio) * self.local_prob)
        self.std = np.sqrt(self.pr * (1 - self.pr) / float(self.sample_count))

        self.sample_count += 1

        self.estimation = self.miss_prob
        self.in_concept_change = False
        self.in_warning_zone = False
        self.delay = 0

        if self.diff > 0:
            if self.sample_count <= self.diff:
                self.global_ratio = self.sample_count / self.diff
            else:
                # Drift missed
                self.global_ratio = 1 - (self.sample_count - self.diff) / self.sample_count
        else:
            self.global_ratio = 0


        if self.sample_count < self.min_instances:
            return

        if self.miss_prob + self.miss_std <= self.miss_prob_sd_min:
            self.miss_prob_min = self.miss_prob
            self.miss_sd_min = self.miss_std
            self.miss_prob_sd_min = self.miss_prob + self.miss_std


        if (self.miss_prob + self.miss_std > self.miss_prob_min + self.out_control_level * self.miss_sd_min)\
                or (self.miss_prob + self.miss_std > self.pr + self.out_control_level * self.std and self.global_ratio > self.confidence):
            self.in_concept_change = True
            key = min(self.actuals.keys(), key=lambda x:abs(x - n))
            if key != 0 and self.actuals[key] == 0 and abs(key - n) <= 1000:
                self.drift_ts.append(n)
                self.detect_TP()
            else:
                self.detect_FP()

        elif self.miss_prob + self.miss_std > self.miss_prob_min + self.warning_level * self.miss_sd_min:
            self.in_warning_zone = True

        else:
            self.in_warning_zone = False


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
        return math.sqrt(pr) / (0.1 * self.global_ratio + math.sqrt(pr))

    def sigmoid_transformation(self, p):
        return 1/(1 + math.exp(-p))

    def get_pr(self):
        return self.pr

    def get_std(self):
        return self.std

    def get_global_ratio(self):
        return self.global_ratio

    def get_drift_ts(self):
        return self.drift_ts

    def get_min_pi(self):
        return self.miss_prob_min

    def get_min_si(self):
        return self.miss_sd_min

    def get_pi(self):
        return self.miss_prob

    def detect_TP(self, n):
        self.drift_ts.append(n)
        if (len(self.drift_ts) > self.ts_length):
            ts = np.diff(self.drift_ts)

            if (self.period <= 1):
                self.diff = ts[-1]
            else:
                # TRUE POSITIVE detected
                learn = ts[-(2 * self.period):]
                print("Learning: " + str(learn))
                temp_df = pd.DataFrame(
                    {'timestamp': pd.date_range('2000-01-01', periods=len(learn), freq=None),
                     'ts': learn})
                temp_df.set_index('timestamp', inplace=True)
                hw_model = ExponentialSmoothing(temp_df, trend=None, seasonal='add', seasonal_periods=self.period).fit(
                    optimized=True)
                predict_result = np.array(hw_model.predict(start=len(learn), end=len(learn)))[0]
                print("Predicted:")
                print(predict_result)
                self.diff = predict_result

            self.ts_prediction = n + self.diff
            self.drift_ts.pop(0)


    def detect_FP(self, n):
        if (self.FP_detected and len(self.drift_ts) >= self.ts_length):
            # Majority Vote approach
            ts = np.diff(self.drift_ts)
            periods = []
            for i in range(0, math.floor(self.ts_length / 2)):
                current = i
                for j in range(current + 2, self.ts_length):
                    if j + 1 < self.ts_length - 1:
                        if abs(ts[j] - ts[current]) < 1000:
                            if abs(ts[j + 1] - ts[current + 1]) < 1000:
                                periods.append(j - current)
                                current = j

            if len(periods) == 0:
                self.period = 1
            else:
                self.period = math.floor(max(set(periods), key=periods.count))

            print("Period: " + str(self.period))
            self.diff = self.ts_prediction - n

    def get_TP(self):
        return self.TP_detected

    def get_FP(self):
        return self.FP_detected

    def get_ts_object(self):
        return self.drift_ts
