import os
import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import RandomForestClassifier as RF
from src.refined_rf import RefinedRandomForest as RRF
from sklearn.cross_validation import StratifiedKFold
from src.telstra_data import TelstraData, multiclass_log_loss
from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment('refined_random_forest')
ex.observers.append(MongoObserver.create(db_name = "telstra"))

@ex.config
def my_config():
    series = "RRF"
    n_folds = 10
    featureparams = {"location_min_count": 0,
                     "n_common_events":20,
                     "n_common_log_features":40,
                     "n_common_resources":5,
                     "n_label_encoded_events":4,
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
    refineparams = {'C':0.01,
                     'n_prunings': 10,
                     'prune_pct': 0.1,
                     'criterion':'sumnorm'}
    save_oob_predictions =  False
    save_test_predictions = False
    skip_cross_validation = False

@ex.automain
def rrf(series, n_folds, clfparams, featureparams, aggregateparams, refineparams, include, exclude,
        save_test_predictions, save_oob_predictions, skip_cross_validation, _run):
    data = TelstraData(include = include, exclude = exclude, **featureparams)
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pred_cols = ['predict_{}'.format(i) for i in range(3)]
    best_pruning = refineparams['n_prunings']
    if skip_cross_validation:
        loss = 999.
    else:
        y = data.get_y()
        kf = StratifiedKFold(y.values, n_folds=n_folds, shuffle=True)
        pred = pd.DataFrame(0., index = y.index, columns = pred_cols)
        i = 1
        _run.info['loss'] = []
        _run.info['trainloss'] = []
        for itrain, itest in kf:
            Xtr, ytr, Xte, yte = data.get_train_test_features(itrain, itest, **aggregateparams)
            clf = RF(**clfparams)
            clf.fit(Xtr, ytr)
            rrf = RRF(clf, **refineparams)
            rrf.fit(Xtr, ytr)
            loss2tr = multiclass_log_loss(ytr.values, rrf.predict_proba(Xtr))
            loss2te = multiclass_log_loss(yte.values, rrf.predict_proba(Xte))
            _run.info['loss'].append(loss2te)
            _run.info['trainloss'].append(loss2tr)
            print("Fold {} mlogloss train: {:.4f}, test: {:.4f}".format(i, loss2tr, loss2te))
            pred.iloc[itest,:] = rrf.predict_proba(Xte)
            i+=1
        loss = multiclass_log_loss(y.values, pred.values)
        _run.info['features'] = list(Xtr.columns)
        # Optionally save oob predictions
        if save_oob_predictions:
            filename = '{}_{}.csv'.format(series, time)
            pred.to_csv(filename, index_label='id')
    # Optionally generate test predictions
    if save_test_predictions:
        filename = '{}_test_{}.csv'.format(series, time)
        Xtr, ytr, Xte, yte = data.get_train_test_features(**aggregateparams)
        #
        # weights = np.concatenate((np.ones(ytr.shape[0]),0.3*np.ones(semilabels.shape[0])))
        # Xtr = pd.concat((Xtr, Xtest), axis=0)
        # ytr = pd.concat((ytr, semilabels))
        clf = RF(**clfparams)
        clf.fit(Xtr, ytr)#,weights)
        rrf = RRF(clf, **refineparams)
        rrf.fit(Xtr, ytr)
        predtest = pd.DataFrame(rrf.predict_proba(Xte),
                                index = yte.index, columns = pred_cols)
        predtest.to_csv(filename, index_label='id')
    return loss
