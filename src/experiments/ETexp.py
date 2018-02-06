import os
import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import ExtraTreesClassifier as ET
from sklearn.cross_validation import StratifiedKFold
from src.telstra_data import TelstraData, multiclass_log_loss
from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment('extra_trees')
ex.observers.append(MongoObserver.create(db_name = "telstra"))

@ex.config
def my_config():
    series = "ET"
    n_folds = 10
    featureparams = {"location_min_count": 0,
                     "n_common_events":20,
                     "n_common_log_features":40,
                     "n_common_resources":5}
    aggregateparams = {"loc_agg_prior_weight":3.0}
    include = []
    exclude = []
    clfparams = {'n_estimators': 200,
                   'n_jobs': -1,
                'criterion': "entropy",
                'min_samples_leaf': 2,
                'bootstrap':False}
    save_oob_predictions =  False
    save_test_predictions = False
    skip_cross_validation = False

@ex.automain
def et(series, n_folds, clfparams, featureparams, aggregateparams, include, exclude,
        save_test_predictions, save_oob_predictions, skip_cross_validation, _run):
    data = TelstraData(include = include, exclude = exclude, **featureparams)
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    pred_cols = ['predict_{}'.format(i) for i in range(3)]
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

            clf = ET(**clfparams)
            clf.fit(Xtr, ytr)
            pred.iloc[itest, :] = clf.predict_proba(Xte)
            trainloss = multiclass_log_loss(ytr, clf.predict_proba(Xtr))
            _run.info['trainloss'].append(trainloss)
            loss = multiclass_log_loss(yte, pred.iloc[itest].values)
            _run.info['loss'].append(loss)
            if i == 1:
                feature_importances_ = clf.feature_importances_/n_folds
            else:
                feature_importances_ += clf.feature_importances_/n_folds
            i += 1
        loss = multiclass_log_loss(y, pred.values)

        _run.info['features'] = list(Xtr.columns)
        _run.info['feature_importances'] = list(feature_importances_)
        # Optionally save oob predictions
        if save_oob_predictions:
            filename = '{}_{}.csv'.format(series, time)
            pred.to_csv(filename, index_label='id')
    # Optionally generate test predictions
    if save_test_predictions:
        filename = '{}_test_{}.csv'.format(series, time)
        Xtr, ytr, Xte, yte = data.get_train_test_features(**aggregateparams)
        clf = ET(**clfparams)
        clf.fit(Xtr, ytr)
        predtest = pd.DataFrame(clf.predict_proba(Xte),
                                index = yte.index, columns = pred_cols)
        predtest.to_csv(filename, index_label='id')
    return loss
