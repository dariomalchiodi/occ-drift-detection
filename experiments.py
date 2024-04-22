"""Experiments for scored-based drift detection"""

import json
import os
import random

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV

from svocc import SVOCC

SEEDS = [12345, 54321, 67890, 98760, 34567]

def set_random_seed(seed):
    """Set the system and numpy random seed."""

    np.random.seed(seed)
    random.seed(seed)

# use scored occ drift detection, analyze one element at a time

def_classifier = GridSearchCV(RandomForestClassifier(),
                              {'n_estimators': [5, 10, 20, 50, 100],
                               'max_depth': [10, 20, None],
                               'max_features': ['sqrt', None]},
                               cv=2)

defaults = {'window_size': 250,
            'p_value_threshold': 0.01,
            'c': 2,
            'classifier': def_classifier,
            'outlier_freq_threshold': 0.3}

# pylint: disable-next=invalid-name
def scored_drift_train(X, y,
                       window_size=defaults['window_size'],
                       p_value_threshold=defaults['p_value_threshold'],
                       keep_fraction=defaults['outlier_freq_threshold'],
                       c=defaults['c'],
                       classifier = defaults['classifier'],
                       use_score=True,
                       verbose=False):
    """Run the scored-based one-class drift detection experiment."""

    # pylint: disable=invalid-name
    drift_detector = SVOCC(c=c)

    X_train = X.iloc[:window_size].values
    y_train = y.iloc[:window_size].values
    drift_detector.fit(X_train)
    classifier.fit(X_train, y_train)
    nodrift_sample = drift_detector.predict(X_train, scored=use_score)
    curr_sample = nodrift_sample.copy()
    num_detections = 0
    predictions = []
    targets = []

    X_curr = X_train
    y_curr = y_train

    for x_new, y_new in zip(X.iloc[window_size:].values,
                            y.iloc[window_size:].values):
        predictions.append(classifier.predict([x_new]))
        targets.append(y_new)
        if len(X_curr) < window_size:
            X_curr = np.vstack([X_curr, [x_new]])
            y_curr = np.hstack([y_curr, [y_new]])
            curr_sample = np.hstack([curr_sample,
                                    drift_detector.predict([x_new],
                                                           scored=use_score)])
        else:
            p_value = mannwhitneyu(curr_sample, nodrift_sample).pvalue
            if p_value < p_value_threshold:
                perc = len(predictions) / (len(X) - window_size) * 100
                msg = f'{perc:.1f}% complete, '
                acc = metrics.accuracy_score(targets, predictions)
                msg += f' accuracy {acc:.3f} '
                msg += f' drift detected (p-value {p_value:.3f})'
                if verbose:
                    print(msg)
                num_detections += 1
                new_size = int(window_size * (1 - keep_fraction))
                X_curr = np.vstack([X_curr[new_size:], [x_new]])
                y_curr = np.hstack([y_curr[new_size:], [y_new]])
                drift_detector.fit(X_curr)
                classifier.fit(X_curr, y_curr)
                nodrift_sample = drift_detector.predict(X_curr,
                                                        scored=use_score)
            else:
                X_curr = np.vstack([X_curr[1:], [x_new]])
                y_curr = np.hstack([y_curr[1:], [y_new]])
                curr_sample = np.hstack([curr_sample[1:],
                                        drift_detector.predict([x_new],
                                                            scored=use_score)])

    acc = metrics.accuracy_score(targets, predictions)
    if verbose:
        print(f'Detected {num_detections} drifts.')
        print(f'Final accuracy {acc:.3f} ')
    return {'num_drift': num_detections, 'accuracy': acc}

# pylint: disable-next=invalid-name
def hybrid_drift_train(X, y,
                       window_size=defaults['window_size'],
                       p_value_threshold=defaults['p_value_threshold'],
                       keep_fraction=defaults['outlier_freq_threshold'],
                       c=defaults['c'],
                       classifier=defaults['classifier'],
                       verbose=False):
    """Run the hybrid scored-based one-class drift detection experiment."""

    return scored_drift_train(X, y,
                       window_size=window_size,
                       p_value_threshold=p_value_threshold,
                       keep_fraction=keep_fraction,
                       c=c,
                       classifier=classifier,
                       use_score=False,
                       verbose=verbose)

# use binary occ drift detection, analyze one element at a time

def binary_drift_train(X, y,
                       window_size=defaults['window_size'],
                       outlier_freq_threshold=defaults['outlier_freq_threshold'],
                       c=defaults['c'],
                       classifier=defaults['classifier'],
                       verbose=False):
    """Run one binary-based one-class drift detection experiment."""

    # pylint: disable=invalid-name
    drift_detector = SVOCC(c=c)
    X_train = X.iloc[:window_size].values
    y_train = y.iloc[:window_size].values
    drift_detector.fit(X_train)
    classifier.fit(X_train, y_train)
    outlier_pred = drift_detector.predict(X_train)

    num_detections = 0
    predictions = []
    targets = []

    X_curr = X_train
    y_curr = y_train

    for x_new, y_new in zip(X.iloc[window_size:].values,
                            y.iloc[window_size:].values):
        predictions.append(classifier.predict([x_new]))
        targets.append(y_new)
        if len(X_curr) < window_size:
            X_curr = np.vstack([X_curr, [x_new]])
            y_curr = np.hstack([y_curr, [y_new]])
            outlier_pred = np.hstack([outlier_pred,
                                    drift_detector.predict([x_new])])
        else:
            outlier_freq = np.mean(outlier_pred)
            if outlier_freq >= outlier_freq_threshold:
                perc = len(predictions) / (len(X) - window_size) * 100
                msg = f'{perc:.1f}% complete, '
                acc = metrics.accuracy_score(targets, predictions)
                msg += f' accuracy {acc:.3f} '
                msg += f' drift detected (outlier freq. {outlier_freq:.3f})'
                if verbose:
                    print(msg)
                num_detections += 1
                new_size = int(window_size * (1 - outlier_freq_threshold))
                X_curr = np.vstack([X_curr[new_size:], [x_new]])
                y_curr = np.hstack([y_curr[new_size:], [y_new]])
                drift_detector.fit(X_curr)
                classifier.fit(X_curr, y_curr)
                outlier_pred = np.hstack([outlier_pred[new_size:],
                                        drift_detector.predict([x_new])])
            else:
                X_curr = np.vstack([X_curr[1:], [x_new]])
                y_curr = np.hstack([y_curr[1:], [y_new]])
                outlier_pred = np.hstack([outlier_pred[1:],
                                        drift_detector.predict([x_new])])
    acc = metrics.accuracy_score(targets, predictions)
    if verbose:
        print(f'Detected {num_detections} drifts.')
        print(f'Final accuracy {acc:.3f} ')
    return {'num_drift': num_detections, 'accuracy': acc}

# pylint: disable-next=invalid-name
def evaluate_method(dataset_name, X, y, method_name, method):
    """Evaluate the performance of a specific drift detection method
    on a dataset for different seeds of the pseudorandom generator."""

    results = []

    for seed in SEEDS:
        set_random_seed(seed)
        classifier = GridSearchCV(RandomForestClassifier(random_state=seed),
                                {'n_estimators': [5, 10, 20, 50, 100],
                                'max_depth': [10, 20, None],
                                'max_features': ['sqrt', None],
                                'n_jobs': [-1]},
                                cv=2)
        results.append(method(X, y, classifier=classifier))

    num_drift = [r['num_drift'] for r in results]
    acc = [r['accuracy'] for r in results]

    result = {'dataset': dataset_name,
              'method': method_name,
              'drift_detected_mean': np.mean(num_drift),
              'drift_detected_std': np.std(num_drift),
              'accuracy_mean':np.mean(acc),
              'accuracy_std':np.std(acc)}

    return result

def analyze_dataset(data_name, data_path):
    """Evaluate the performance all drift detection methods on a dataset."""

    df = pd.read_csv(data_path)
    # pylint: disable-next=invalid-name
    X = df.drop(['target'], axis=1)
    y = df['target']

    method_names = ['SDD', 'HDD', 'BDD']
    methods = [scored_drift_train, hybrid_drift_train, binary_drift_train]

    return [evaluate_method(data_name, X, y, method_name, method)
            for method_name, method in tqdm(zip(method_names, methods))]

def run_experiment(path, name):
    """Run a batch of experiments for all datasets in a directory."""

    result = []
    for file in tqdm(os.listdir(path)):
        data_name = file[:-4]
        data_path = path + file
        result.extend(analyze_dataset(data_name, data_path))

    with open(f'result-{name}.json', 'w', encoding='utf-8') as f:
        json.dump(result, f)

if __name__ == '__main__':
    for category in ['real-world', 'artificial']:
        run_experiment(f'data/{category}/', category)
