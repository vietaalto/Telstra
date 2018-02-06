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
                     'n_prunings': 20,
                     'prune_pct': 0.1,
                     'criterion':'sumnorm'}
    save_oob_predictions =  False
    save_test_predictions = False
    skip_cross_validation = False

@ex.automain
def rf(series, n_folds, clfparams, featureparams, aggregateparams, refineparams, include, exclude,
        save_test_predictions, save_oob_predictions, skip_cross_validation, _run):
    data = TelstraData(include = include, exclude = exclude, **featureparams)
    X, y, Xtest, ytest = data.get_train_test_features(**aggregateparams)
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pred_cols = ['predict_{}'.format(i) for i in range(3)]
    rrfparams = dict(refineparams)
    rrfparams['n_prunings'] = 1
    kf = StratifiedKFold(y.values, n_folds=n_folds, shuffle=True)
    pred = pd.DataFrame(0., index = y.index, columns = pred_cols)
    testpreds = [] # list for storing  test predictions
    i = 1
    _run.info['loss'] = []
    _run.info['trainloss'] = []
    _run.info['best_pruning'] = []
    for itrain, itest in kf:
        Xtr, ytr, Xte, yte = data.get_train_test_features(itrain, itest, **aggregateparams)
        clf = RF(**clfparams)
        clf.fit(Xtr, ytr)
        rrf = RRF(clf, **rrfparams)
        best_loss = 1000.
        train_loss = 1000.
        no_improvement_in = 0
        for k in range(refineparams['n_prunings']):
            try:
                rrf.fit(Xtr, ytr) # fit and do 1 pruning
            except IndexError as e:
                print('IndexError')
                break
            loss2tr = multiclass_log_loss(ytr.values, rrf.predict_proba(Xtr))
            loss2te = multiclass_log_loss(yte.values, rrf.predict_proba(Xte))
            print("Fold {} Pruning {} mlogloss train: {:.4f}, test: {:.4f}".format(i,k, loss2tr, loss2te))
            if loss2te < best_loss: # performance is better
                no_improvement_in = 0
                best_loss = loss2te + 0. # save new best loss
                # predict oof samples with new model
                pred.iloc[itest,:] = rrf.predict_proba(Xte)
                # predict test with new model
                testpred = rrf.predict_proba(Xtest)
                # record current train loss
                train_loss = loss2tr + 0.
            else:
                no_improvement_in += 1
            if no_improvement_in >= 5:
                break
        # Append current testpred to testpreds list
        testpreds.append(pd.DataFrame(testpred,
                                index = ytest.index, columns = pred_cols))
        # Save loss and train loss from current fold
        _run.info['loss'].append(best_loss)
        _run.info['trainloss'].append(train_loss)
        _run.info['best_pruning'].append(k + 1)
        print("Fold {} mlogloss train: {:.4f}, test: {:.4f}".format(i, train_loss, best_loss))
        i+=1
    loss = multiclass_log_loss(y.values, pred.values)
    _run.info['features'] = list(Xtr.columns)

    filename = '{}_{}_cv_{:.4f}.csv'.format(series, time, loss)
    pred.to_csv(filename, index_label='id')
    # Optionally generate test predictions
    testpred = sum(testpreds) / len(testpreds)
    filename = '{}_test_{}_cv_{:.4f}.csv'.format(series, time, loss)
    testpred.to_csv(filename, index_label='id')
    return loss
