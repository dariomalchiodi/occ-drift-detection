"""Experiments for scored-based drift detection"""

import argparse
from collections import namedtuple
import json
import os
from pathlib import Path
import random

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats import weightstats
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV

from svocc import SVOCC
from svocc.kernel import GaussianKernel

# TODO
# - conteggiare uno solo tra i tre tipi di drift quando vengono rilevati
#   assieme in una finestra
# - verificare come vengono salvati i p-value, perché in covtype sembra che
#   ci sia una differenza nel numero di label drift ma i conteggi sembrano
#   uguali
# - pensare dei dataset artificiali che inducono un drift specifico (sia in
#   termini di covariate, label e congiunto, sia nella modalità subitanea,
#   graduale, ecc.) valutando anche se questo non sia già stato fatto nei
#   dataset sintetici.
# - nei grafici del p-value indicare con delle barre i momenti nei quali
#   sappiamo che c'è stato un drift (sicuramente per i dati sintetici, forse
#   anche per quelli reali).
# - valutare se è veramente necessario ragionare sul drift di x e y insieme.
# - usare il test Z per valutare il drift delle etichette nei casi di
#   classificazione binaria.


SEED = 3759914
TestResult = namedtuple('TestResult', ['stat', 'pvalue'])

def set_random_seed(seed):
    """Set the system and numpy random seed."""

    np.random.seed(seed)
    random.seed(seed)

def ztest(x1, x2):
    """Adapter for the Z test returning a named tuple."""

    if set(x1) == {1} and (x1 == x2).all():
        # test result with equal samples, both having all components = 1
        result = TestResult(-1, 0.31731)
    elif set(x1) == {0} and (x1 == x2).all():
        # test result with equal samples, both having all components = 0
        result = TestResult(1, 0.31731)
    else:
        result = TestResult(*weightstats.ztest(x1, x2))

    return result


# pylint: disable-next=invalid-name
def clean_dataset(X, y, cv=2):
    """Remove cases from dataset having less than two examples."""
    y = pd.Series(y)
    target_freq = y.value_counts()
    invalid_targets = [i for i, f in target_freq.items() if f<cv]
    X = X[~y.isin(invalid_targets)]
    y = y[~y.isin(invalid_targets)]
    return X, y

def_classifier = GridSearchCV(RandomForestClassifier(random_state=SEED),
                              {'n_estimators': [5, 10, 20, 50, 100],
                               'max_depth': [10, 20, None],
                               'max_features': ['sqrt', None]},
                               cv=2,
                               n_jobs=-1)

defaults = {'window_size': 250,
            'p_value_threshold': 0.01,
            'c': 2,
            'classifier': def_classifier,
            'outlier_freq_threshold': 0.3}

# pylint: disable-next=invalid-name
def scored_drift_train(X, y, pbar,
                       window_size=defaults['window_size'],
                       p_value_threshold=defaults['p_value_threshold'],
                       keep_fraction=defaults['outlier_freq_threshold'],
                       c=defaults['c'],
                       classifier = defaults['classifier'],
                       test = mannwhitneyu,
                       use_score=True,
                       verbose=False):
    """Run the scored-based one-class drift detection experiment."""

    # pylint: disable=invalid-name
    cov_drift_detector = SVOCC(c=c, k=GaussianKernel(0.1))
    dis_drift_detector = SVOCC(c=c, k=GaussianKernel(0.1))
    lab_drift_detector = SVOCC(c=c, k=GaussianKernel(0.1))

    X_train = X.iloc[:window_size].values
    y_train = y.iloc[:window_size].values

    X_train, y_train = clean_dataset(X_train, y_train)
    y_train_expanded = np.expand_dims(y_train, axis=1)
    Xy_train = np.hstack([X_train, y_train_expanded])

    cov_drift_detector.fit(X_train)
    dis_drift_detector.fit(Xy_train)
    lab_drift_detector.fit(y_train_expanded)
    classifier.fit(X_train, y_train)

    cov_nodrift_sample = cov_drift_detector.predict(X_train, scored=use_score)
    cov_curr_sample = cov_nodrift_sample.copy()
    dis_nodrift_sample = dis_drift_detector.predict(Xy_train,
                                                    scored=use_score)
    dis_curr_sample = dis_nodrift_sample.copy()
    lab_nodrift_sample = lab_drift_detector.predict(y_train_expanded,
                                                    scored=use_score)
    lab_curr_sample = lab_nodrift_sample.copy()

    cov_detections = dis_detections = lab_detections = 0
    predictions = []
    targets = []
    cov_p = []
    dis_p = []
    lab_p = []

    X_curr = X_train
    y_curr = y_train
    y_curr_expanded = np.expand_dims(y_curr, axis=1)
    Xy_curr = np.hstack([X_curr, y_curr_expanded])

    for x_new, y_new in zip(X.iloc[window_size:].values,
                            y.iloc[window_size:].values):
        pbar.update()
        predictions.append(classifier.predict([x_new]))
        targets.append(y_new)
        xy_new = np.hstack([x_new, [y_new]])
        if len(X_curr) < window_size:
            X_curr = np.vstack([X_curr, [x_new]])
            y_curr = np.hstack([y_curr, [y_new]])
            y_curr_expanded = np.expand_dims(y_curr, axis=1)
            Xy_curr = np.hstack([X_curr, y_curr_expanded])

            cov_curr_sample = np.hstack([cov_curr_sample,
                                cov_drift_detector.predict([x_new],
                                                           scored=use_score)])
            dis_curr_sample = np.hstack([dis_curr_sample,
                                dis_drift_detector.predict([xy_new],
                                                           scored=use_score)])
            lab_curr_sample = np.hstack([lab_curr_sample,
                                lab_drift_detector.predict([y_curr],
                                                           scored=use_score)])
        else:
            drift_detected = False

            cov_p_value = test(cov_curr_sample, cov_nodrift_sample).pvalue
            cov_p.append(cov_p_value)
            if cov_p_value < p_value_threshold:
                cov_detections += 1
                drift_detected = True

            dis_p_value = test(dis_curr_sample, dis_nodrift_sample).pvalue
            dis_p.append(dis_p_value)
            if dis_p_value < p_value_threshold:
                dis_detections += 1
                drift_detected = True

            lab_p_value = test(lab_curr_sample, lab_nodrift_sample).pvalue
            lab_p.append(lab_p_value)
            if lab_p_value < p_value_threshold:
                lab_detections += 1
                drift_detected = True

            if drift_detected:
                new_size = int(window_size * (1 - keep_fraction))
                X_curr = np.vstack([X_curr[new_size:], [x_new]])
                y_curr = np.hstack([y_curr[new_size:], [y_new]])
                X_curr, y_curr = clean_dataset(X_curr, y_curr)
                y_curr_expanded = np.expand_dims(y_curr, axis=1)
                Xy_curr = np.hstack([X_curr, y_curr_expanded])

                cov_drift_detector.fit(X_curr)
                dis_drift_detector.fit(Xy_curr)
                lab_drift_detector.fit(y_curr_expanded)

                classifier.fit(X_curr, y_curr)

                cov_nodrift_sample = cov_drift_detector.predict(X_curr,
                                                        scored=use_score)
                dis_nodrift_sample = dis_drift_detector.predict(Xy_curr,
                                                    scored=use_score)
                lab_nodrift_sample = lab_drift_detector.predict(y_curr_expanded,
                                                    scored=use_score)
            else:
                X_curr = np.vstack([X_curr[1:], [x_new]])
                y_curr = np.hstack([y_curr[1:], [y_new]])
                Xy_curr = np.vstack([Xy_curr[1:], [xy_new]])

                cov_curr_sample = np.hstack([cov_curr_sample[1:],
                                        cov_drift_detector.predict([x_new],
                                                            scored=use_score)])
                dis_curr_sample = np.hstack([dis_curr_sample[1:],
                                        dis_drift_detector.predict([xy_new],
                                                            scored=use_score)])
                lab_curr_sample = np.hstack([lab_curr_sample[1:],
                                        lab_drift_detector.predict([[y_new]],
                                                            scored=use_score)])

    acc = metrics.accuracy_score(targets, predictions)
    if verbose:
        print('Detected:')
        print(f'{cov_detections} covariate drifts.')
        print(f'{dis_detections} distribution drifts.')
        print(f'{lab_detections} label drifts.')
        print(f'Final accuracy {acc:.3f} ')
    return {'cov_detections': cov_detections,
            'dis_detections': dis_detections,
            'lab_detections': lab_detections,
            'accuracy': acc,
            'cov_p_values': cov_p,
            'dis_p_values': dis_p,
            'lab_p_values': lab_p}

# pylint: disable-next=invalid-name
def hybrid_drift_train(X, y, pbar,
                       window_size=defaults['window_size'],
                       p_value_threshold=defaults['p_value_threshold'],
                       keep_fraction=defaults['outlier_freq_threshold'],
                       c=defaults['c'],
                       classifier=defaults['classifier'],
                       verbose=False):
    """Run the hybrid scored-based one-class drift detection experiment."""

    return scored_drift_train(X, y, pbar,
                       window_size=window_size,
                       p_value_threshold=p_value_threshold,
                       keep_fraction=keep_fraction,
                       c=c,
                       classifier=classifier,
                       test=ztest,
                       use_score=False,
                       verbose=verbose)

# use binary occ drift detection, analyze one element at a time

def binary_drift_train(X, y, pbar,
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
    X_train, y_train = clean_dataset(X_train, y_train)
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
        pbar.update()
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
                X_curr, y_curr = clean_dataset(X_curr, y_curr)
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
    return {'cov_detections': num_detections,
            'dis_detections': 0,
            'lab_detections': 0,
            'accuracy': acc,
            'cov_p_values': [],
            'dis_p_values': [],
            'lab_p_values': []}

# pylint: disable-next=invalid-name
def evaluate_method(dataset_name, X, y, method_name, method, pbar):
    """Evaluate the performance of a specific drift detection method
    on a dataset."""

    result = method(X, y, pbar)

    result = {'dataset': dataset_name,
              'method': method_name,
              'cov_drift_detected': result['cov_detections'],
              'dis_drift_detected': result['dis_detections'],
              'lab_drift_detected': result['lab_detections'],
              'cov_p_values': result['cov_p_values'],
              'dis_p_values': result['dis_p_values'],
              'lab_p_values': result['lab_p_values'],
              'accuracy': result['accuracy']}

    return result

def analyze_dataset(data_name, data_path):
    """Evaluate the performance all drift detection methods on a dataset."""

    df = pd.read_csv(data_path)
    # pylint: disable-next=invalid-name
    X = df.drop(['target'], axis=1)
    y = df['target']

    pbar = tqdm(total=(3 * (len(y) - defaults['window_size'])))

    method_names = ['SDD', 'HDD', 'BDD']
    methods = [scored_drift_train, hybrid_drift_train, binary_drift_train]

    result = [evaluate_method(data_name, X, y, method_name, method, pbar)
              for method_name, method in zip(method_names, methods)]
    pbar.close()
    return result

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='process arguments')
    parser.add_argument('dataset')
    parser.add_argument('-o', dest='output', default='.')

    args = parser.parse_args()
    dataset = args.dataset
    output_dir = Path(args.output)

    set_random_seed(SEED)

    name = dataset[dataset.rfind('/')+1:-4]

    output_file = output_dir / Path(name + '.json')
    if os.path.exists(output_file):
        print(f'{output_file} already exists: skipping.')
    else:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analyze_dataset(name, dataset), f)
