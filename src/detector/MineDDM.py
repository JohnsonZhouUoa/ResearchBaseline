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
                 default_prob=0.5, predict_size=100):
        super().__init__()
        self.sample_count = 1
        self.global_sample_count = 1
        self.global_ratio = 1.0
        self.drift_count = 0
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
        self.predict_size = predict_size
        self.learn_ts = False
        self.predict_result = 0.01
        self.drift_ts = []
        self.reset()

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
        self.learn_ts = True

    def add_element(self, prediction):

        if self.in_concept_change:
            self.reset()

        self.miss_prob = self.miss_prob + (prediction - self.miss_prob) / float(self.sample_count)
        self.miss_std = np.sqrt(self.miss_prob * (1 - self.miss_prob) / float(self.sample_count))
        self.sample_count += 1
        self.global_sample_count += 1

        self.estimation = self.miss_prob
        self.in_concept_change = False
        self.in_warning_zone = False
        self.delay = 0

        if (self.learn_ts):
            if (len(self.drift_ts) > 10):
                temp_df = pd.DataFrame({'timestamp': pd.date_range('2000-01-01', periods=len(self.drift_ts),freq='Q'), 'ts': self.drift_ts})
                temp_df.set_index('timestamp', inplace=True)
                hw_model = ExponentialSmoothing(temp_df, trend='add', seasonal='add').fit(optimized=True)
                pred = np.array(hw_model.predict(len(self.drift_ts), len(self.drift_ts)))
                self.predict_result = pred[0]
                self.learn_ts = False


        if self.drift_count > 0:
            ratio = self.global_sample_count / self.predict_result
            if (ratio < 1):
                self.global_ratio = ratio

        self.global_prob = self.calculate_pr(self.global_sample_count, self.drift_count)
        self.local_prob = self.calculate_pr(self.sample_count, 1)
        pr = self.sigmoid_transformation(self.global_ratio * self.global_prob + (1 - self.global_ratio) * self.local_prob)

        std = np.sqrt(pr * (1 - pr) / float(self.sample_count))

        if self.sample_count < self.min_instances:
            return

        if self.miss_prob + self.miss_std <= self.miss_prob_sd_min:
            self.miss_prob_min = self.miss_prob
            self.miss_sd_min = self.miss_std
            self.miss_prob_sd_min = self.miss_prob + self.miss_std

        if pr + std <= self.miss_prob_sd_min:
            self.miss_prob_min = pr
            self.miss_sd_min = std
            self.miss_prob_sd_min = pr + std

        if self.miss_prob + self.miss_std > self.miss_prob_min + self.out_control_level * self.miss_sd_min:
            self.in_concept_change = True
            self.drift_count += 1
            self.drift_ts.append(self.global_sample_count)

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

    def sigmoid_transformation(self, pr):
        #TODO
        return 1-math.exp(-pr)
