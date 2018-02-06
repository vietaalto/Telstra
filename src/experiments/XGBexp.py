import os
import pandas as pd
import numpy as np
import xgboost as xgb
import datetime
from sklearn.cross_validation import StratifiedKFold
from sacred import Experiment
from src.telstra_data import TelstraData, multiclass_log_loss
from sacred.observers import MongoObserver

ex = Experiment('gradient_boosting')
ex.observers.append(MongoObserver.create(db_name = "telstra"))

@ex.config
def my_config():
    series = "XGB"
    n_folds = 10
    num_trees = 2000
    early_stopping_rounds = 50
    verbose_eval = 0
    featureparams = {"location_min_count": 0,
                     "n_common_events":20,
                     "n_common_log_features":60,
                     "n_common_resources":10,
                     "n_label_encoded_log_features":4}
    aggregateparams = {"loc_agg_prior_weight":3.0}
    include = []
    exclude = []
    clfparams = {"objective": "multi:softprob",
                 "eval_metric": 'mlogloss',
                 "num_class":3,
                 "max_delta_step": 0,
                 "eta": 0.3,
                 "gamma": 0.,
                 "max_depth": 6,
                 "subsample": 1.,
                 "colsample_bytree": 1.,
                 "lambda": 1.,
                 "alpha": 0.,
                 "silent": 1}
    save_oob_predictions =  False
    save_test_predictions = False
    skip_cross_validation = False

@ex.automain
def xgbrun(series, n_folds, clfparams, featureparams, num_trees,
early_stopping_rounds, verbose_eval, _seed, _run, aggregateparams, include, exclude,
        save_test_predictions, save_oob_predictions, skip_cross_validation):
    clfparams['seed'] = _seed
    data = TelstraData(include = include, exclude = exclude, **featureparams)
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pred_cols = ['predict_{}'.format(i) for i in range(3)]
    num_rounds = num_trees + 0
    params = {"verbose_eval":verbose_eval}
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
            dtrain = xgb.DMatrix(Xtr, ytr)
            dvalid = xgb.DMatrix(Xte, yte)
            params = {"verbose_eval":verbose_eval}
            if (i == 1) and (early_stopping_rounds > 0) :
                watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
                params["evals"] = watchlist
                params["early_stopping_rounds"] = early_stopping_rounds

            gbm = xgb.train(clfparams, dtrain, num_rounds, **params)

            if i == 1:
                num_rounds = gbm.best_ntree_limit
                _run.info['num_rounds'] = num_rounds

            pred.iloc[itest, :] = gbm.predict(dvalid,ntree_limit=num_rounds).reshape(yte.shape[0],len(pred_cols))
            predtrain = gbm.predict(dtrain,ntree_limit=num_rounds).reshape(ytr.shape[0],len(pred_cols))
            loss = multiclass_log_loss(yte, pred.iloc[itest].values)
            trainloss = multiclass_log_loss(ytr, predtrain)
            #print("Fold {:02d}: trainloss = {:.4f}, testloss = {:.4f}".format(i,trainloss, loss))
            _run.info['loss'].append(loss)
            _run.info['trainloss'].append(trainloss)
            i += 1
        loss = multiclass_log_loss(y, pred.values)

        _run.info['features'] = list(Xtr.columns)
        # Optionally save oob predictions
        if save_oob_predictions:
            filename = '{}_{}.csv'.format(series, time)
            pred.to_csv(filename, index_label='id')

    # Optionally generate test predictions
    if save_test_predictions:
        filename = '{}_test_{}.csv'.format(series, time)
        Xtr, ytr, Xte, yte = data.get_train_test_features(**aggregateparams)
        dtrain = xgb.DMatrix(Xtr, ytr)
        dtest = xgb.DMatrix(Xte)
        gbm = xgb.train(clfparams, dtrain, num_rounds, **params)
        predtest = pd.DataFrame(gbm.predict(dtest).reshape(yte.shape[0],len(pred_cols)),
                                index = yte.index, columns = pred_cols)
        predtest.to_csv(filename, index_label='id')
    return loss
