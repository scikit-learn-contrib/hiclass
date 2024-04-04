import numpy as np
from sklearn.model_selection import StratifiedKFold
from hiclass._calibration.BinaryCalibrator import _BinaryCalibrator
from scipy.stats import gmean
from hiclass._calibration.calibration_utils import _one_vs_rest_split
from collections import defaultdict
from sklearn.utils.validation import check_is_fitted


class _InductiveVennAbersCalibrator(_BinaryCalibrator):
    name = "InductiveVennAbersCalibrator"

    def __init__(self):
        self._is_fitted = False

    def fit(self, y, scores, X=None):
        positive_label = 1
        unique_labels = np.unique(y)
        assert len(unique_labels) <= 2

        y = np.where(y == positive_label, 1, 0)
        y = y.reshape(-1)  # make sure it's a 1D array
        # sort all scores s1, ..., sk in increasing order

        order_idx = np.lexsort([y, scores])
        ordered_calibration_scores, ordered_calibration_labels = scores[order_idx], y[order_idx]
        # remove duplicates
        unique_elements, unique_idx, unique_element_counts = np.unique(ordered_calibration_scores, return_index=True, return_counts=True)
        ordered_unique_calibration_scores, _ = ordered_calibration_scores[unique_idx], ordered_calibration_labels[unique_idx]

        self.k_distinct = len(unique_idx)

        def compute_csd(un_el, un_el_counts, ocs, ocl, oucs):

            # Count the frequencies of each s'j
            w = dict(zip(un_el, un_el_counts))

            y = np.zeros(self.k_distinct)
            csd = np.zeros((self.k_distinct + 1, 2))

            for j in range(self.k_distinct):
                s_j = oucs[j]
                matching_idx = np.where(ocs == s_j)
                matching_labels = ocl[matching_idx]
                y[j] = np.sum(matching_labels) / w[un_el[j]]

            csd[1:, 0] = np.cumsum(un_el_counts)
            csd[1:, 1] = np.cumsum(y * un_el_counts)

            return list(csd)

        def slope(top, next_to_top):
            return (next_to_top[1] - top[1]) / (next_to_top[0] - top[0])

        def at_or_above(p, cur_slope, top, next_to_top):
            intersection_point = (p[0], top[1] + cur_slope * (p[0] - top[0]))
            return p[1] >= intersection_point[1]

        def non_left_angle_turn(next_to_top, top, p_i):
            next_to_top = np.array(next_to_top)
            top = np.array(top)
            p_i = np.array(p_i)
            res = np.cross((top - next_to_top), (p_i - top))
            return res <= 0

        def non_right_angle_turn(next_to_top, top, p_i):
            next_to_top = np.array(next_to_top)
            top = np.array(top)
            p_i = np.array(p_i)
            res = np.cross((top - next_to_top), (p_i - top))
            return res >= 0

        def initialize_f1_corners(csd):
            stack = []
            # append P_{-1} and P_0
            stack.append(csd[0])
            stack.append(csd[1])

            for i in range(2, len(csd)):
                while len(stack) > 1 and non_left_angle_turn(next_to_top=stack[-2], top=stack[-1], p_i=csd[i]):
                    stack.pop()
                stack.append(csd[i])

            return stack

        def initialize_f0_corners(csd):
            stack = []
            # append p_{k'+1}, p_{k'}
            stack.append(csd[-1])
            stack.append(csd[-2])

            for i in range(len(csd) - 3, -1, -1):
                while len(stack) > 1 and non_right_angle_turn(next_to_top=stack[-2], top=stack[-1], p_i=csd[i]):
                    stack.pop()
                stack.append(csd[i])
            return stack

        point_addition = lambda p1, p2: tuple((p1[0] + p2[0], p1[1] + p2[1]))
        point_subtraction = lambda p1, p2: tuple((p1[0] - p2[0], p1[1] - p2[1]))

        def compute_f1(prev_stack, csd):
            F1 = np.zeros(self.k_distinct + 1)
            stack = []
            while prev_stack:
                stack.append(prev_stack.pop())

            for i in range(2, self.k_distinct + 2):
                F1[i - 1] = slope(top=stack[-1], next_to_top=stack[-2])
                # p_{i-1}
                csd[i - 1] = point_subtraction(point_addition(csd[i - 2], csd[i]), csd[i - 1])
                p_temp = csd[i - 1]

                if at_or_above(p_temp, F1[i - 1], top=stack[-1], next_to_top=stack[-2]):
                    continue

                stack.pop()
                while len(stack) > 1 and non_left_angle_turn(p_temp, stack[-1], stack[-2]):
                    stack.pop()
                stack.append(p_temp)
            return F1

        def compute_f0(prev_stack, csd):
            F0 = np.zeros(self.k_distinct + 1)
            stack = []
            while prev_stack:
                stack.append(prev_stack.pop())

            for i in range(self.k_distinct, 0, -1):
                F0[i] = slope(top=stack[-1], next_to_top=stack[-2])
                csd[i] = point_subtraction(point_addition(csd[i - 1], csd[i + 1]), csd[i])

                if at_or_above(csd[i], F0[i], top=stack[-1], next_to_top=stack[-2]):
                    continue
                stack.pop()
                while len(stack) > 1 and non_right_angle_turn(csd[i], stack[-1], stack[-2]):
                    stack.pop()
                stack.append(csd[i])
            return F0

        csd_1 = compute_csd(
            unique_elements,
            unique_element_counts,
            ordered_calibration_scores,
            ordered_calibration_labels,
            ordered_unique_calibration_scores
        )
        csd_0 = csd_1.copy()
        csd_0.append((csd_0[-1][0] + 1, csd_0[-1][1] + 0))
        csd_1.insert(0, (-1, -1))

        f1_stack = initialize_f1_corners(csd_1)
        f0_stack = initialize_f0_corners(csd_0)

        self.F1 = compute_f1(f1_stack, csd_1)
        self.F0 = compute_f0(f0_stack, csd_0)
        self.unique_elements = unique_elements
        self._is_fitted = True

        return self

    def predict_proba(self, scores, X=None):
        check_is_fitted(self)
        lower = np.searchsorted(self.unique_elements, scores, side="left")
        upper = np.searchsorted(self.unique_elements[:-1], scores, side="right") + 1

        p0 = self.F0[lower]
        p1 = self.F1[upper]

        return p1 / (1 - p0 + p1)

    def predict_intervall(self, scores):
        lower = np.searchsorted(self.unique_elements, scores, side="left")
        upper = np.searchsorted(self.unique_elements[:-1], scores, side="right") + 1
        p0 = self.F0[lower]
        p1 = self.F1[upper]

        return np.array(list(zip(p0, p1)))


class _CrossVennAbersCalibrator(_BinaryCalibrator):
    name = "CrossVennAbersCalibrator"

    def __init__(self, estimator, n_folds=5) -> None:
        self._is_fitted = False
        self.n_folds = n_folds
        self.estimator_type = type(estimator)
        self.estimator_params = estimator.get_params()
        self.estimator = estimator
        self.multiclass = False
        self.use_estimator_fallback = False
        self.used_cv = True

    def fit(self, y, scores, X):
        unique_labels = np.unique(y)
        assert len(unique_labels) >= 2
        self.ivaps = []

        try:
            splitter = StratifiedKFold(n_splits=self.n_folds)
            splits_x = []
            splits_y = []
            for train_index, cal_index in splitter.split(X, y):
                splits_x.append((X[train_index], X[cal_index]))
                splits_y.append((y[train_index], y[cal_index]))
        except ValueError:
                splits_x, splits_y = [], []

        # don't use cross validation
        if len(splits_x) == 0 or any([(len(np.unique(y_train)) < 2 or len(np.unique(y_cal)) < 2) for y_train, y_cal in splits_y]):
            self.used_cv = False
            print("skip cv split due to lack of positive samples!")

            if len(unique_labels) > 2:
                # use one vs rest
                score_splits, label_splits = _one_vs_rest_split(y, scores, self.estimator) # TODO use only original calibration samples
                for i in range(len(score_splits)):
                    # create a calibrator for each split
                    calibrator = _InductiveVennAbersCalibrator()
                    calibrator.fit(label_splits[i], score_splits[i])
                    self.ivaps.append(calibrator)
            elif len(unique_labels) == 2 and scores.ndim == 1:
                calibrator = _InductiveVennAbersCalibrator()
                calibrator.fit(y, scores) # TODO use only original calibration samples
                self.ivaps.append(calibrator)
            else:
                print("no fitted ivaps!")
                self.use_estimator_fallback = True

        else:
            self.ovr_ivaps = defaultdict(list)
            for i in range(self.n_folds):
                X_train, X_cal = splits_x[i][0], splits_x[i][1]
                y_train, y_cal = splits_y[i][0], splits_y[i][1]

                # train underlying model with x_train and y_train
                model = self.estimator_type()
                model.set_params(**self.estimator_params)
                model.fit(X_train, y_train)

                # calibrate IVAP with left out dataset
                calibration_scores = model.predict_proba(X_cal)
                
                if calibration_scores.shape[1] > 2:
                    self.multiclass = True
                    # one vs rest calibration
                    score_splits, label_splits = _one_vs_rest_split(y_cal, calibration_scores, model)
                    for idx in range(len(score_splits)):
                        # create a calibrator for each split
                        calibrator = _InductiveVennAbersCalibrator()
                        calibrator.fit(label_splits[idx], score_splits[idx])
                        self.ovr_ivaps[idx].append(calibrator)

                elif calibration_scores.shape[1] == 2 and len(np.unique(y_cal)) == 2:
                    calibrator = _InductiveVennAbersCalibrator()
                    calibrator.fit(y_cal, calibration_scores[:, 1])
                    self.ivaps.append(calibrator)

            if len(self.ivaps) == 0 and len(self.ovr_ivaps) == 0:
                print("no fitted ivaps!")
                self.use_estimator_fallback = True

        self._is_fitted = True

        return self

    def predict_proba(self, scores):
        check_is_fitted(self)

        if self.use_estimator_fallback:
            return scores

        if self.multiclass:

            score_splits = [scores[:, i] for i in range(scores.shape[1])]
            probabilities = np.zeros((scores.shape[0], scores.shape[1]))

            if self.used_cv:

                #score_splits = [scores[:, i] for i in range(scores.shape[1])]
                #probabilities = np.zeros((scores.shape[0], scores.shape[1]))

                for idx, scores in enumerate(score_splits):
                    res = []

                    if not self.ovr_ivaps[idx]:
                        continue

                    for calibrator in self.ovr_ivaps[idx]:
                        res.append(calibrator.predict_intervall(scores))
                    
                    res = np.array(res)
                    
                    p0 = res[:, :, 0]
                    p1 = res[:, :, 1]

                    p1_gm = gmean(p1)
                    probabilities[:, idx] = p1_gm / (gmean(1 - p0) + p1_gm)
                
                # normalize
                #probabilities /= probabilities.sum(axis=1, keepdims=True)
                #return probabilities
            
            else:
                #score_splits = [scores[:, i] for i in range(scores.shape[1])]
                #probabilities = np.zeros((scores.shape[0], scores.shape[1]))
                for idx, scores in enumerate(score_splits):
                    probabilities[:, idx] = self.ivaps[idx].predict_proba(scores)
                
                # normalize
                #probabilities /= probabilities.sum(axis=1, keepdims=True)
                #return probabilities
                    
            # normalize       
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            return probabilities

        else:
            res = []
            for calibrator in self.ivaps:
                res.append(calibrator.predict_intervall(scores))
            
            res = np.array(res)
            p0 = res[:, :, 0]
            p1 = res[:, :, 1]

            p1_gm = gmean(p1)
            return p1_gm / (gmean(1 - p0) + p1_gm)
