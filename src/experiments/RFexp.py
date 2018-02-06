import os
import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.cross_validation import StratifiedKFold
from src.telstra_data import TelstraData, multiclass_log_loss
from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment('random_forest')
ex.observers.append(MongoObserver.create(db_name = "telstra"))

@ex.config
def my_config():
    series = "RF"
    n_folds = 10
    featureparams = {"location_min_count": 0,
                     "n_common_events":20,
                     "n_common_log_features":40,
                     "n_common_resources":5,
                     "n_label_encoded_events":0,
                     "n_label_encoded_log_features":4}
    aggregateparams = {"loc_agg_prior_weight":3.0}
    include = []
    exclude = []
    clfparams = {'n_estimators': 200,
    'criterion': 'gini',
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 4,
    'min_weight_fraction_leaf': 0.0,
    'max_features': 'auto',
    'max_leaf_nodes': None,
    'bootstrap': True,
    'oob_score': False,
    'n_jobs': -1,
    'verbose': 0,
    'warm_start': False,
    'class_weight': None}
    save_oob_predictions =  False
    save_test_predictions = False
    skip_cross_validation = False

@ex.automain
def rf(series, n_folds, clfparams, featureparams, aggregateparams, include, exclude,
        save_test_predictions, save_oob_predictions, skip_cross_validation, _run):
    data = TelstraData(include = include, exclude = exclude, **featureparams)
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
            clf = RF(**clfparams)
            clf.fit(Xtr, ytr)#, weights)
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
        clf = RF(**clfparams)
        clf.fit(Xtr, ytr)#,weights)
        predtest = pd.DataFrame(clf.predict_proba(Xte),
                                index = yte.index, columns = pred_cols)
        predtest.to_csv(filename, index_label='id')
    return loss
