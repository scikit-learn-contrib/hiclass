from sklearn.preprocessing import LabelBinarizer

def _one_vs_rest_split(y, scores, estimator):
        # binarize multiclass labels
        label_binarizer = LabelBinarizer()
        label_binarizer.fit(estimator.classes_)
        binary_labels = label_binarizer.transform(y).T

        # split scores into k one vs rest splits
        score_splits = [scores[:, i] for i in range(scores.shape[1])]
        label_splits = [binary_labels[i] for i in range(len(score_splits))]

        return score_splits, label_splits