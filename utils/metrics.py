"""
Metrics utility module for evaluating classification performance.
"""
import logging
import math
from sklearn import metrics
import numpy as np


def roc(y_true, y_pred, y_record_ids, c=0, calculation_method='subsample_level'):
    """
    Calculate the Receiver Operating Characteristic (ROC) values.

    Args:
        y_true (list): True labels.
        y_pred (list): Predicted probabilities.
        y_record_ids (list): Record identifiers.
        c (int, optional): The class index to evaluate. Defaults to 0.
        calculation_method (str, optional): Method for calculation ('subsample_level' or 'sample_level'). Defaults to 'subsample_level'.

    Returns:
        dict: A dictionary containing 'TPR', 'FPR', and 'Threshold' lists.
    """
    thresholds = [i / 100 for i in range(100)]
    rocvalues = {'TPR': [0.0], 'FPR': [0.0], 'Threshold': [1.0]}

    for threshold in thresholds:
        tp, fp, tn, fn, _ = confusionmatrix(y_true=y_true, y_pred=y_pred, y_record_ids=y_record_ids, c=c, threshold=threshold, calculation_method=calculation_method)

        tpr = truepositiverate(tp, fn)
        fpr = falsepositiverate(tn, fp)

        rocvalues['TPR'].append(tpr)
        rocvalues['FPR'].append(fpr)
        rocvalues['Threshold'].append(threshold)

    rocvalues['TPR'].append(1.0)
    rocvalues['FPR'].append(1.0)
    rocvalues['Threshold'].append(0.0)

    rocvalues['TPR'] = sorted(rocvalues['TPR'])
    rocvalues['FPR'] = sorted(rocvalues['FPR'])
    rocvalues['Threshold'] = sorted(rocvalues['Threshold'])

    return rocvalues


def auc(rocvalues):
    """
    Calculate the Area Under the ROC Curve (AUC).

    Args:
        rocvalues (dict): Dictionary with 'TPR' and 'FPR' lists.

    Returns:
        float: The computed AUC value.
    """
    tpr_values = rocvalues['TPR']
    fpr_values = rocvalues['FPR']

    return metrics.auc(fpr_values, tpr_values)


def confusionmatrix(y_true, y_pred, y_record_ids, c=0, threshold=0.5, calculation_method='subsample_level'):
    """
    Calculate the confusion matrix parameters (TP, FP, TN, FN) and record classifications.

    Args:
        y_true (list): True labels.
        y_pred (list): Predicted probabilities.
        y_record_ids (list): Record identifiers.
        c (int, optional): The class index. Defaults to 0.
        threshold (float, optional): Classification threshold. Defaults to 0.5.
        calculation_method (str, optional): Supported calculation methods are 'sample_level' and 'subsample_level'. Defaults to 'subsample_level'.

    Returns:
        tuple: True Positives (tp), False Positives (fp), True Negatives (tn), False Negatives (fn), and a dictionary of classifications (cls).
    """
    assert calculation_method in ['sample_level', 'subsample_level']

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    cls = {'TP': [], 'TN': [], 'FP': [], 'FN': []}

    # If calculation on sample level is required, all predictions are aggregated in advance
    if calculation_method == 'sample_level':
        sample_ids = [rec_id.rsplit('_', maxsplit=1)[0] for rec_id in y_record_ids]
        y_pred_s = {s_id: [] for s_id in sample_ids}
        y_true_s = {s_id: y_t for s_id, y_t in zip(sample_ids, y_true)}
        y_true_s_final = [y_true_s[s_id] for s_id in y_true_s]
        sample_ids_final = list(y_pred_s.keys())

        for y_p, s_id in zip(y_pred, sample_ids):
            y_pred_s[s_id].append(y_p)

        y_width = len(y_true[0])
        y_pred_s_mean = {}

        for s_id in y_pred_s:
            for i in range(y_width):
                y_pred_s_mean[s_id] = list(np.mean(y_pred_s[s_id], axis=0))

        y_pred_s_final = [y_pred_s_mean[s_id] for s_id in y_pred_s_mean]

        y_true = y_true_s_final
        y_pred = y_pred_s_final
        y_record_ids = sample_ids_final

    # Calculation of TP, FP, TN, FN
    for y_t, y_p, y_r in zip(y_true, y_pred, y_record_ids):
        # Positive
        if y_t[c] == 1.0:

            # Prediction positive
            if y_p[c] > threshold:
                tp += 1
                cls['TP'].append(y_r)

            # Prediction negative
            else:
                fn += 1
                cls['FN'].append(y_r)

        # Negative
        elif y_t[c] == 0.0:

            # Prediction positive
            if y_p[c] > threshold:
                fp += 1
                cls['FP'].append(y_r)

            # Prediction negative
            else:
                tn += 1
                cls['TN'].append(y_r)

        else:
            raise ValueError('True label for class {} has to be 0.0 or 1.0, was {}'.format(c, y_t[c]))

    return tp, fp, tn, fn, cls


def sensitivity(tp, fn):
    """Calculate sensitivity (true positive rate)."""
    return tp / (tp + fn)  # divisor can only be zero when no positives are in the set


def specificity(fp, tn):
    """Calculate specificity (true negative rate)."""
    return tn / (tn + fp)  # divisor can only be zero when no negatives are in the set


def truepositiverate(tp, fn):
    """Calculate the true positive rate (sensitivity)."""
    return sensitivity(tp, fn)


def falsepositiverate(tn, fp):
    """Calculate the false positive rate."""
    return fp / (fp + tn)  # divisor can only be zero when no negatives are in the set


def accuracy(tp, fp, tn, fn):
    """Calculate overall accuracy."""
    return (tp + tn) / (tp + tn + fp + fn)  # divisor can never be zero


def f1score(tp, fp, fn):
    """Calculate the F1 score."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    if (precision + recall) == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def youdensjstatistic(tp, fp, tn, fn):
    """Calculate Youden's J statistic."""
    return sensitivity(tp, fn) + specificity(fp, tn) - 1


def diagnosticoddsratio(tp, fp, tn, fn):
    """
    Calculate the Diagnostic Odds Ratio (DOR).
    Reference: Glas et al. 2003, DOI: 10.1016/S0895-4356(03)00177-X
    """

    sens = sensitivity(tp, fn)
    spec = specificity(fp, tn)

    # cases needed to prevent zero division
    if spec == 0.0:
        spec = 0.00001

    if spec == 1.0:
        spec = 0.99999

    if sens == 1.0:
        sens = 0.99999

    return (sens * spec) / ((1-sens) * (1-spec))


def positivepredictivevalue(tp, fp):
    """Calculate the Positive Predictive Value (PPV) or Precision."""
    if tp + fp == 0:
        return 0
    else:
        return tp / (tp + fp)  # divisor can be zero when all records are labeled negative


def negativepredictivevalue(tn, fn):
    """Calculate the Negative Predictive Value (NPV)."""
    if tn + fn == 0:
        return 0
    else:
        return tn / (tn + fn)  # divisor can be zero when all records are labeled positive


def matthewscorrelationcoefficient(tp, tn, fp, fn):
    """
    Calculate the Matthews Correlation Coefficient (MCC).
    Reference: https://doi.org/10.1016%2F0005-2795%2875%2990109-9
    """
    s1 = (tp + fp)
    s2 = (tp + fn)
    s3 = (tn + fp)
    s4 = (tn + fn)
    v = s1 * s2 * s3 * s4

    if (s1 == 0 and s2 != 0 and s3 != 0 and s4 != 0) or (s1 != 0 and s2 == 0 and s3 != 0 and s4 != 0) or (s1 != 0 and s2 != 0 and s3 == 0 and s4 != 0) or (s1 != 0 and s2 != 0 and s3 != 0 and s4 == 0):
        v = 1
    elif v == 0:
        return np.nan

    return (tp * tn - fp * fn) / math.sqrt(v)


def add_classification_metric_to_dictionary(metric_name, dictionary, tp, tn, fp, fn):
    """
    Calculate and add a specific classification metric to the given dictionary.

    Args:
        metric_name (str): Name of the metric to compute.
        dictionary (dict): Dictionary to append the metric to.
        tp (int): True Positives.
        tn (int): True Negatives.
        fp (int): False Positives.
        fn (int): False Negatives.
    """
    if metric_name == 'sensitivity':
        dictionary[metric_name] = sensitivity(tp, fn)
    elif metric_name == 'specificity':
        dictionary[metric_name] = specificity(fp, tn)
    elif metric_name == 'truepositiverate':
        dictionary[metric_name] = truepositiverate(tp, fn)
    elif metric_name == 'falsepositiverate':
        dictionary[metric_name] = falsepositiverate(tn, fp)
    elif metric_name == 'accuracy':
        dictionary[metric_name] = accuracy(tp, fp, tn, fn)
    elif metric_name == 'f1score':
        dictionary[metric_name] = f1score(tp, fp, fn)
    elif metric_name == 'youdensjstatistic':
        dictionary[metric_name] = youdensjstatistic(tp, fp, tn, fn)
    elif metric_name == 'DOR':
        dictionary[metric_name] = diagnosticoddsratio(tp, fp, tn, fn)
    elif metric_name == 'AUC':
        pass  # AUC included by default
    elif metric_name == 'PPV':
        dictionary[metric_name] = positivepredictivevalue(tp, fp)
    elif metric_name == 'NPV':
        dictionary[metric_name] = negativepredictivevalue(tn, fn)
    elif metric_name == 'MCC':
        dictionary[metric_name] = matthewscorrelationcoefficient(tp, tn, fp, fn)
    elif metric_name == 'SSB':
        raise Exception('Implement!')
    else:
        logging.warning('Unknown metric "{}". Skipping this one. Please check spelling.'.format(metric_name))


def is_low_value_good_for_metric(metric_name):
    """
    Determine if a lower value for a given metric indicates better performance.

    Args:
        metric_name (str): The metric name to check.

    Returns:
        bool: True if lower is better, False otherwise.
    """
    if metric_name == 'sensitivity':
        return False
    elif metric_name == 'specificity':
        return False
    elif metric_name == 'SSB':
        return False
    elif metric_name == 'truepositiverate':
        return False
    elif metric_name == 'falsepositiverate':
        return True
    elif metric_name == 'accuracy':
        return False
    elif metric_name == 'f1score':
        return False
    elif metric_name == 'youdensjstatistic':
        return False
    elif metric_name == 'DOR':
        return False
    elif metric_name == 'AUC':
        return False
    elif metric_name == 'PPV':
        return False
    elif metric_name == 'NPV':
        return False
    elif metric_name == 'MSE':
        return True
    elif metric_name == 'RMSE':
        return True
    elif metric_name == 'MCC':
        return False
    else:
        raise Exception('Unknown metric "{}"'.format(metric_name))


def calculate_metrics_for_classification(y_true, y_pred, y_classes, y_record_ids, metrics_to_calculate, calculation_methods, use_c=None):
    """
    Calculate multiple metrics for classification performance across requested classes.

    Args:
        y_true (list): True labels.
        y_pred (list): Predicted probabilities.
        y_classes (list): Class names corresponding to indices.
        y_record_ids (list): Record identifiers.
        metrics_to_calculate (list): List of metric names to evaluate.
        calculation_methods (list): List of calculation methods ('subsample_level', 'sample_level').
        use_c (int, optional): If provided, calculates metrics only for this specific class index. Defaults to None.

    Returns:
        dict: Evaluation results structured by class name and calculation method.
    """
    result = {}

    for c in range(len(y_classes)):
        if use_c is not None:
            if c != use_c:
                continue

        y_class = y_classes[c]

        result[y_class] = {}

        for calculation_method in calculation_methods:
            tp, fp, tn, fn, cls = confusionmatrix(y_true, y_pred, y_record_ids, c=c, calculation_method=calculation_method)
            rocvalues = roc(y_true, y_pred, y_record_ids, c=c, calculation_method=calculation_method)
            area_under_roc = auc(rocvalues)

            met = {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn, 'AUC': area_under_roc, 'ROC': rocvalues}
            logging.debug('{} - TP: {}, TN: {}, FP: {}, FN: {}, AUC: {}'.format(calculation_method, tp, tn, fp, fn, area_under_roc))

            for m in metrics_to_calculate:
                add_classification_metric_to_dictionary(m, met, tp=tp, tn=tn, fp=fp, fn=fn)

            result[y_class][calculation_method] = {'metrics': met, 'classification': cls}

    return result
