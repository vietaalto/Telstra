import os
import pandas as pd
import numpy as np
import datetime
from sklearn.cross_validation import StratifiedKFold
from sacred import Experiment
from sacred.observers import MongoObserver
from src.NNmodel import NN_with_CategoryEmbedding as NN
from src.telstra_data import TelstraData, multiclass_log_loss

ex = Experiment('neural_network')
ex.observers.append(MongoObserver.create(db_name = "telstra"))

@ex.config
def my_config():
    series = "NN"
    n_folds = 8
    nb_epoch = 100
    early_stopping_rounds = 10
    featureparams = {"location_min_count": 0,
                     "n_common_events":-1, # use all
                     "n_common_log_features":320,
                     "n_common_resources":-1}
    aggregateparams = {"loc_agg_prior_weight":1.0}
    include = []
    exclude = []
    modelparams = {'dense_1_size': 139,
                  'dense_2_size': 85,
                  'dense_3_size':0,
                  'dropout_1': 0.33518161422102916,
                  'dropout_2': 0.11022698073986464,
                  'dropout_3': 0.10,
                  'events_model_l1': 0.0828912773683751,
                  'events_model_l2': 0.09058440269409214,
                  'events_model_size': 36,
                  'location_embedding_l1': 152.047625356857,
                  'location_embedding_l2': 0.005514020843033711,
                  'location_embedding_size': 989,
                  'log_model_l1': 0.0009942575290641173,
                  'log_model_l2': 0.0065254811643931775,
                  'log_model_size': 209,
                  'resource_model_l1': 0.17048437747361556,
                  'resource_model_l2': 0.0018463316856817326,
                  'resource_model_size': 6,
                  'rest_model_l1': 0.00013833756522212644,
                  'rest_model_l2': 0.00032397510167688485,
                  'rest_model_size': 16,
                  'severity_embedding_l1': 0.0004075447695903401,
                  'severity_embedding_l2': 0.006030296208917552,
                  'severity_embedding_size': 2}
    save_oob_predictions =  False
    save_test_predictions = False
    skip_cross_validation = False

@ex.automain
def nnrun(_run, modelparams, nb_epoch, early_stopping_rounds, series, n_folds,
            featureparams, aggregateparams, include, exclude,
            save_test_predictions, save_oob_predictions, skip_cross_validation):
    data = TelstraData(include = include, exclude = exclude, **featureparams)
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pred_cols = ['predict_{}'.format(i) for i in range(3)]
    best_epoch = nb_epoch
    if skip_cross_validation:
        loss = 999.
    else:
        y = data.get_y()
        kf = StratifiedKFold(y.values, n_folds=n_folds, shuffle=True)
        pred = pd.DataFrame(0., index = y.index, columns = pred_cols)
        i = 1
        _run.info['history'] = []
        for itrain, itest in kf:
            Xtr, ytr, Xte, yte = data.get_train_test_features(itrain, itest, **aggregateparams)
            # find best nb_epoch by early stopping in the first fold
            params = {"nb_epoch":best_epoch}
            if i == 1:
                params["early_stopping_rounds"] = early_stopping_rounds
            model = NN(modelparams)

            history = model.fit_val(Xtr, ytr, Xte, yte, **params)
            _run.info['history'].append(history)
            pred.iloc[itest, :] = model.predict_proba(Xte)
            if i == 1:
                best_epoch = len(history['loss']) - early_stopping_rounds
                _run.info['best_epoch'] = best_epoch
            loss = multiclass_log_loss(yte, pred.iloc[itest].values)
            print('Fold {:02d}: Loss = {:.5f}'.format(i, loss))
            i += 1
            break
        loss = multiclass_log_loss(y.values, pred.values)
        # Optionally save oob predictions
        if save_oob_predictions:
            filename = '{}_{}.csv'.format(series, time)
            pred.to_csv(filename, index_label='id')
    # Optionally generate test predictions
    if save_test_predictions:
        filename = '{}_test_{}.csv'.format(series, time)
        Xtr, ytr, Xte, yte = data.get_train_test_features(**aggregateparams)
        model = NN(modelparams)
        history = model.fit(Xtr, ytr, nb_epoch = best_epoch)
        predtest = pd.DataFrame(model.predict_proba(Xte),
                                index = yte.index, columns = pred_cols)
        predtest.to_csv(filename, index_label='id')
    return loss
