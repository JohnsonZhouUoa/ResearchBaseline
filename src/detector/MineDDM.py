import numpy as np
import math
from operator import mul
from fractions import Fraction
from functools import reduce
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
from .HoltWinters import ExponentialSmoothing
import pandas as pd
from memory_profiler import profile

#fp = open('memory_mine.log', 'w')

class MineDDM(BaseDriftDetector):

    def __init__(self, min_num_instances=30, warning_level=2.0, out_control_level=3.0,
                 default_prob=1, actuals = []):
        super().__init__()
        self.sample_count = 1
        self.global_ratio = 1.0
        self.drift_count = 0
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
        self.drift_detected = False
        self.drift_ts = []
        self.reset()
        self.diff = -1
        self.ts_prediction = -1
        self.actuals = dict.fromkeys(actuals, 0)
        self.TP_detected = False
        self.FP_detected = False

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
        self.pr = 0
        self.std = 0

    #@profile(stream=fp)
    def add_element(self, prediction, n_global):

        if self.in_concept_change:
            self.reset()

        self.miss_prob = self.miss_prob + (prediction - self.miss_prob) / float(self.sample_count)
        self.miss_std = np.sqrt(self.miss_prob * (1 - self.miss_prob) / float(self.sample_count))
        if (len(self.drift_ts) > 14):
            self.global_prob = self.calculate_pr(
                n_global - self.drift_ts[0],
                len(self.drift_ts))

        else:
            self.global_prob = self.calculate_pr(n_global,
                                                 self.drift_count
                                                 )
        self.local_prob = self.calculate_pr(self.sample_count, 1)
        self.pr = self.activation_function(self.global_ratio * self.global_prob + (1 - self.global_ratio) * self.local_prob)
        self.std = np.sqrt(self.pr * (1 - self.pr) / float(self.sample_count))

        self.sample_count += 1

        self.estimation = self.miss_prob
        self.in_concept_change = False
        self.in_warning_zone = False
        self.delay = 0

        if (self.drift_detected):
            if (len(self.drift_ts) > 10):
                # TRUE POSITIVE detected
                # ts = np.diff(self.drift_ts)
                #
                # res = ts[1:]
                # min_i = 1
                # for i in range(len(res)):
                #     if abs(res[i] - ts[0]) < 500:
                #         min_i += 1
                #
                # temp_df = pd.DataFrame(
                #         {'timestamp': pd.date_range('2000-01-01', periods=len(ts), freq='A'),
                #          'ts': ts})
                # temp_df.set_index('timestamp', inplace=True)
                # hw_model = ExponentialSmoothing(temp_df, trend='add', seasonal='add', seasonal_periods=(math.floor(len(ts) / min_i) if min_i > 1 else 1)).fit(optimized=True)
                # predict_result = np.array(hw_model.predict(len(self.drift_ts), len(self.drift_ts)))[0]
                # print("Predicted:")
                # print(predict_result)
                # self.diff = predict_result
                # self.ts_prediction = n_global + self.diff
                self.drift_ts.pop(0)

                keys = list(self.actuals.keys())
                key = min(keys, key=lambda x: abs(x - n_global))
                if key == keys[-1]:
                    self.diff = 5000
                else:
                    self.diff = keys[keys.index(key) + 1] - n_global
                self.ts_prediction = n_global + self.diff

            else:
                # FALSE POSITIVE detected
                # Wrong prediction from TS Model
                self.diff = self.ts_prediction - n_global
            self.drift_detected = False

        if self.diff > 0:
            if self.sample_count <= self.diff:
                self.global_ratio = self.sample_count / self.diff
            else:
                # DDM miss it
                self.global_ratio = 1 - (self.sample_count - self.diff) / self.sample_count
        else:
            self.global_ratio = 0


        if self.sample_count < self.min_instances:
            return

        if self.miss_prob + self.miss_std <= self.miss_prob_sd_min:
            self.miss_prob_min = self.miss_prob
            self.miss_sd_min = self.miss_std
            self.miss_prob_sd_min = self.miss_prob + self.miss_std


        if self.miss_prob + self.miss_std > self.miss_prob_min + self.out_control_level * self.miss_sd_min:
            self.in_concept_change = True
            self.drift_count += 1
            key = min(self.actuals.keys(), key=lambda x:abs(x-n_global))
            if key != 0 and self.actuals[key] == 0 and abs(key-n_global) <= 500:
                print("A true positive detected at " + str(n_global))
                print("TP: " + str(key))
                self.drift_ts.append(n_global)
                self.actuals[key] += 1
                self.detect_TP()
            else:
                print("A false positive detected at " + str(n_global))
                print("GLobal ratio:" + str(self.global_ratio))
                print("Key: " + str(key))
                self.detect_FP()
            self.drift_detected = True

        elif self.miss_prob + self.miss_std > self.pr + self.out_control_level * self.std and self.global_ratio > 0.95:
            self.in_concept_change = True
            self.drift_count += 1
            key = min(self.actuals.keys(), key=lambda x:abs(x-n_global))
            if key != 0 and self.actuals[key] == 0 and abs(key-n_global) <= 500:
                print("A true positive detected at " + str(n_global))
                print("GLobal ratio:" + str(self.global_ratio))
                print("TP: " + str(key))
                self.drift_ts.append(n_global)
                self.actuals[key] += 1
                self.detect_TP()
            else:
                print("A false positive detected at " + str(n_global))
                print("GLobal ratio:" + str(self.global_ratio))
                print("Pr:" + str(self.pr))
                print("Key: " + str(key))
                self.detect_FP()
            self.drift_detected = True

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
        #return math.sqrt(pr) / (0.05 * self.global_ratio + math.sqrt(pr))
        #return (math.sqrt(pr)/((1-self.global_ratio) + math.sqrt(pr)) + (1-self.global_ratio))
        return math.sqrt(pr) / (1 + math.sqrt(pr))

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

    def detect_TP(self):
        self.TP_detected = True
        self.FP_detected = False

    def detect_FP(self):
        self.FP_detected = True
        self.TP_detected = False

    def get_TP(self):
        return self.TP_detected

    def get_FP(self):
        return self.FP_detected
