import os
import pandas as pd
import numpy as np
import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedKFold
from ..telstra_data import TelstraData, multiclass_log_loss
from sacred import Experiment
from sacred.observers import MongoObserver
from scipy.optimize import minimize

ex = Experiment('kNN')
ex.observers.append(MongoObserver.create(db_name = "telstra"))

@ex.config
def my_config():
    series = "kNN"
    n_folds = 10
    featureparams = {"location_min_count": 0,
                     "n_common_events":20,
                     "n_common_log_features":100,
                     "n_common_resources":10,
                     "n_label_encoded_events":0,
                     "n_label_encoded_log_features":0}
    aggregateparams = {"loc_agg_prior_weight":3.0}
    include = []
    exclude = ['location','sevtype','logfeatle_min',]
    clfparams = {'n_neighbors': 10,
    'weights':'distance',
    'p':2,
    'metric':'minkowski',
    }
    save_oob_predictions =  False
    save_test_predictions = False
    skip_cross_validation = False

@ex.automain
def knn(series, n_folds, clfparams, featureparams, aggregateparams, include, exclude,
        save_test_predictions, save_oob_predictions, skip_cross_validation, _run):
    data = TelstraData(include = include, exclude = exclude, **featureparams)
    data.features_to_scale.append('location')
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pred_cols = ['predict_{}'.format(i) for i in range(3)]
    best_eps = 1e-15
    if skip_cross_validation:
        loss = 999.
    else:
        y = data.get_y()
        kf = StratifiedKFold(y.values, n_folds=n_folds, shuffle=True)
        pred = pd.DataFrame(0., index = y.index, columns = pred_cols)
        i = 1
        _run.info['loss'] = []
        _run.info['trainloss'] = []
        feature_importances_ = 0
        for itrain, itest in kf:
            Xtr, ytr, Xte, yte = data.get_train_test_features(itrain, itest, **aggregateparams)

            clf = KNeighborsClassifier(**clfparams)
            clf.fit(Xtr, ytr)#, weights)
            pred.iloc[itest, :] = clf.predict_proba(Xte)
            i += 1
        def obj(x):
            return multiclass_log_loss(y.values, pred.values, eps = 10.**x)
        res = minimize(obj, -2.)
        best_eps = 10**(res.x[0])
        loss = multiclass_log_loss(y, pred.values, eps=best_eps)
        _run.info['best_eps'] = best_eps
        _run.info['features'] = list(Xtr.columns)
        # Optionally save oob predictions
        if save_oob_predictions:
            filename = '{}_{}.csv'.format(series, time)
            pred.to_csv(filename, index_label='id')
    # Optionally generate test predictions
    if save_test_predictions:
        filename = '{}_test_{}.csv'.format(series, time)
        Xtr, ytr, Xte, yte = data.get_train_test_features(**aggregateparams)

        clf = KNeighborsClassifier(**clfparams)
        clf.fit(Xtr, ytr)#,weights)
        predtest = pd.DataFrame(clf.predict_proba(Xte),
                                index = yte.index, columns = pred_cols)
        predtest = np.clip(y_pred, best_eps, 1 - best_eps)
        predtest /= predtest.values.sum(axis=1)[:, np.newaxis]
        predtest.to_csv(filename, index_label='id')
    return loss
