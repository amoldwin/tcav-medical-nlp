#From https://github.com/h4ste/clue_benchmark/blob/master/clue_metric.py

import numpy as np

import datasets

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, \
    balanced_accuracy_score, matthews_corrcoef, roc_auc_score, average_precision_score, precision_score, \
    recall_score, f1_score, brier_score_loss


_CITATION = """TBD."""

_DESCRIPTION = """TBD."""

_KWARGS_DESCRIPTION = """TBD."""


def mean_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray):
    true_directions = np.sign(y_true[1:], y_true[:-1])
    pred_directions = np.sign(y_pred[1:], y_pred[:-1])
    return np.mean(true_directions == pred_directions)


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class ClueMetric(datasets.Metric):
    def _info(self):
        if self.config_name not in {
            'staging_aki',
            'staging_pi',
            'staging_anemia',
            'phenotyping',
            'mortality',
            'length_of_stay'
        }:
            raise KeyError(
                "You should supply a configuration name from {"
                "'staging_aki', 'staging_pi', 'staging_anemia'"
                ", 'phenotyping', 'mortality', 'length_of_stay'}"
            )

        features = {}

        if self.config_name.startswith("staging"):
            features["predictions"] = datasets.Value("int32")
            features["references"] = datasets.Value("int32")
        elif self.config_name == "phenotyping":
            features["predictions"] = datasets.Value("bool")
            features["scores"] = datasets.Value("float32")
            features["references"] = datasets.Value("bool")
        elif self.config_name == "mortality":
            features["predictions"] = datasets.Value("int32")
            features["references"] = datasets.Value("int32")
        elif self.config_name == "length_of_stay":
            features["predictions"] = datasets.Value("int32")
            features["references"] = datasets.Value("int32")
        else:
            # This should be impossible to reach since we checked the configurations above
            raise AssertionError("Invalid configuration")

        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(features),
            codebase_urls=[],
            reference_urls=[],
            format="numpy",
        )

    def _compute(self, predictions, references, scores=None):
        print('computing metric', flush=True)
        if self.config_name == "phenotyping":
            assert False
            return {
                'Prec.': precision_score(y_true=references, y_pred=predictions),
                'Rec.': recall_score(y_true=references, y_pred=predictions),
                'F1-macro': f1_score(y_true=references, y_pred=predictions, average='macro'),
                'F1-micro': f1_score(y_true=references, y_pred=predictions, average='micro'),
                'Acc.': balanced_accuracy_score(y_true=references, y_pred=predictions),
                'ROC': roc_auc_score(y_true=references, y_score=scores),
                'PRC': average_precision_score(y_true=references, y_score=scores),
                'MCC': matthews_corrcoef(y_true=references, y_pred=predictions),
                'Brier': brier_score_loss(y_true=references, y_prob=scores)
            }
        else:
            return {
                'MAE': mean_absolute_error(y_true=references, y_pred=predictions),
                'MDA': mean_directional_accuracy(y_true=references, y_pred=predictions),
                'R2': r2_score(y_true=references, y_pred=predictions),
                'MSE': mean_squared_error(y_true=references, y_pred=predictions)
            }