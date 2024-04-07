import pandas as pd
from sklearn import metrics


class Scores:
    def __init__(self):
        self.scoring_methods = [
            metrics.balanced_accuracy_score,
            metrics.adjusted_rand_score,
            metrics.f1_score,
            metrics.recall_score,
            metrics.matthews_corrcoef,
            metrics.jaccard_score,
            metrics.hamming_loss,
            metrics.precision_score,
        ]
        self.names = [
            "Balanced_accuracy_",
            "ARI_",
            "F1_",
            "Recall_",
            "Matthews_",
            "Jaccard_",
            "Hamming_",
            "Precision_",
            "FDR_",
        ]
        self.scores = [[] for _ in self.names]

    def get_classification_scores(self, y_true, y_pred):
        for i, scoring in enumerate(self.scoring_methods):
            res = scoring(y_true, y_pred)
            self.scores[i].append(res)
        self.scores[-1].append(1 - res)

    def save_confusion_matrix(
        self, y_true, y_pred, resfolder, plottype, scorename, gs_name
    ):
        mx = metrics.confusion_matrix(y_true, y_pred)
        df = pd.DataFrame(mx)
        df.index.name = "true label"
        df.columns.name = "predicted label"
        df.to_csv(
            resfolder
            + "confusion_matrix_"
            + plottype
            + "/"
            + gs_name
            + "_"
            + scorename
            + ".csv"
        )
