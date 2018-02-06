import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class SparseRobustMaxScaler(object):
    # for scaling log feature volume variables
    def __init__(self, percentile = 75):
        self.percentile_ = percentile

    def fit(self, X,y=None):
        z = np.copy(X)
        z[z==0] = np.NaN
        p = np.nanpercentile(np.abs(z),self.percentile_,axis=0)
        p[np.isnan(p)] = 1.
        self.scale_ = p

    def transform(self, X):
        return X / self.scale_

class TelstraData(object):
    def __init__(self, datadir = '../data', include = [], exclude = [], **featureparams):
        self.data = self.load_data(datadir)
        self.build_features(**featureparams)
        self.include = include
        self.exclude = exclude
    def load_data(self, datadir):
        def loc_converter(location_string):
            return int(location_string.split(" ")[1])
        train = pd.read_csv(os.path.join(datadir, 'train.csv'), index_col='id',
                    converters = {'location':loc_converter})
        test = pd.read_csv(os.path.join(datadir, 'test.csv'), index_col='id',
                    converters = {'location':loc_converter})
        events = pd.read_csv(os.path.join(datadir,'event_type.csv'),
                    converters = {'event_type':loc_converter})
        log = pd.read_csv(os.path.join(datadir,'log_feature.csv'),
                    converters = {'log_feature':loc_converter})
        resource = pd.read_csv(os.path.join(datadir,'resource_type.csv'),
                    converters = {'resource_type':loc_converter})
        sev = pd.read_csv(os.path.join(datadir,'severity_type.csv'), index_col='id',
                    converters = {'severity_type':loc_converter})
        loc = pd.concat((train[['location']],test[['location']]),axis=0)
        testpreds = pd.read_csv(os.path.join(datadir,'test_preds.csv'), index_col='id')
        oofpreds = pd.read_csv(os.path.join(datadir,'oof_preds.csv'), index_col='id')
        preds = pd.concat((testpreds,oofpreds),axis=0)
        return dict(    train = train, test = test,
                events = events, log = log,
                resource=resource, sev = sev, loc=loc, preds = preds)

    def build_features(self,
                       location_min_count = 0,
                       event_min_count = 0,
                       log_feature_min_count = 0,
                       n_common_events = 20, # -1 to use all
                       n_common_log_features = 40,
                       n_common_resources = 5,
                       n_label_encoded_events = 0,
                       n_label_encoded_log_features = 4,
                       ):
        loc = self.data['loc']
        log = self.data['log']
        sev = self.data['sev']
        resource = self.data['resource']
        events = self.data['events']
        preds = self.data['preds']

        features = []
        features_to_scale = []
        X = pd.DataFrame(0, index=sev.index, columns = [])
        X['fault_severity'] = self.data['train'].fault_severity
        #######################
        # severity types data #
        #######################
        X['sevtype'] = sev.severity_type
        features.append('sevtype')
        ##################
        # locations data #
        ##################
        X['location'] = loc['location']
        features.append('location')
        # magic feature here
        X['num'] = X.groupby('location')['sevtype'].transform(lambda x: np.arange(x.shape[0])+1)
        # three different types of normalizing
        X['numsh'] = X.groupby('location')['num'].transform(lambda x: x/(x.shape[0]+1))
        X['numsh0'] = X.groupby('location')['num'].transform(lambda x: (x-1)/(x.shape[0]))
        X['numsh1'] = X.groupby('location')['num'].transform(lambda x: x/(x.shape[0]))
        features.extend(['num','numsh','numsh0','numsh1'])
        features_to_scale.extend(['num','numsh','numsh0','numsh1'])
        # location counts
        lc = pd.DataFrame(loc['location'].value_counts()).rename(columns={'location':'loc_count'})
        X = pd.merge(X, lc, how='left', left_on='location',right_index=True).fillna(0)
        features.append('loc_count')
        features_to_scale.append('loc_count')
        # optionally bin low count locations together
        if location_min_count > 0:
            to_replace = lc.loc_count[lc.loc_count <= location_min_count].index
            value = to_replace.max()
            X['location'] = X['location'].replace(to_replace, value)
        ###############
        # events data #
        ###############
        # number of events per id
        nevents = pd.DataFrame(events['id'].value_counts()).rename(columns={'id':'nevents'})
        X = pd.merge(X, nevents, right_index=True, left_index=True, how='left').fillna(0)
        features.append('nevents')
        features_to_scale.append('nevents')
        # binary features for common events
        evtypes = events.event_type.value_counts()
        common_events = evtypes.index
        if n_common_events != -1:
            common_events = common_events[:n_common_events]
        ohevents = events.loc[events.event_type.isin(common_events)].groupby(['id','event_type'])['id'].count()
        ohevents = ohevents.unstack().fillna(0).add_prefix('event_')
        X = pd.merge(X, ohevents, right_index=True, left_index=True, how='left').fillna(0)
        features.extend(ohevents.columns)
        # Label-encoded events
        if n_label_encoded_events > 0:
            events['num'] = events.groupby('id')['id'].cumcount()+1
            filtered = events[events['num']<=n_label_encoded_events]
            evle = filtered.set_index(['id','num']).unstack().fillna(0)
            evle.columns = evle.columns.get_level_values(1)
            evle = evle.add_prefix('evle_')
            X[evle.columns] = evle
            features.extend(evle.columns)
        ####################
        # log feature data #
        ####################
        # Volume aggregates on log scale
        log['logvolume'] = np.log(log.volume + 1)
        X['volsumlog'] = np.log1p(log.groupby('id')['volume'].agg('sum'))
        features.append('volsumlog')
        features_to_scale.append('volsumlog')
        logvol = log.groupby('id')['logvolume'].agg(['count','min','mean','max','std','sum']).fillna(0).add_prefix('logvolume_')
        X = pd.merge(X, logvol, how='left', right_index=True, left_index=True).fillna(0)
        features.extend(logvol.columns)
        features_to_scale.extend(logvol.columns)
        # Volume variables for common features
        common_features = log.log_feature.value_counts().index
        if n_common_log_features != -1:
            common_features = common_features[:n_common_log_features]
        ohlog = log.loc[log.log_feature.isin(common_features)].groupby(['id','log_feature'])['logvolume'].mean()
        ohlog = ohlog.unstack().fillna(0).add_prefix('logfeatvol_')
        X = pd.merge(X, ohlog, how='left', left_index=True, right_index=True).fillna(0)
        features.extend(ohlog.columns)
        features_to_sparse_scale = list(ohlog.columns)
        # Label-encoded features for max volume, min Volume
        def label_encoded_log_features(df, n):
            ind = df['volume'].values.argsort()
            if len(ind) > n:
                ind = ind[:n]
            lef = np.zeros(n,dtype=np.int32)
            lef[np.arange(len(ind))] = df['log_feature'].values[ind]
            return pd.Series(lef,np.arange(n)+1)
        if n_label_encoded_log_features > 0:
            logfeatle = log.groupby('id').apply(label_encoded_log_features,n_label_encoded_log_features).add_prefix('logfeatle_')
            X[logfeatle.columns] = logfeatle[logfeatle.columns]
            features.extend(logfeatle.columns)
        X['logfeatle_min'] = log.groupby('id').apply(lambda df:df['log_feature'].values[df['volume'].values.argmin()])
        features.extend(['logfeatle_min'])
        ## volsumlog relative to moving average
        #X['rcount'] = X.groupby('location')['num'].transform(lambda x: pd.rolling_count(x,window=3,center=True))
        rmean = lambda x: pd.rolling_mean(x,window=9,min_periods=1,center=True)
        X['logvolume_sum_ma9'] = X.groupby('location')['logvolume_sum'].transform(rmean)
        X['logvolume_sum_ma9_diff'] = X['logvolume_sum'] - X['logvolume_sum_ma9']
        X['volsumlog_ma9'] = X.groupby('location')['volsumlog'].transform(rmean)
        X['volsumlog_ma9_diff'] = X['volsumlog'] - X['volsumlog_ma9']
        ma = X.groupby('location')['logfeatvol_203'].transform(rmean)
        X['logfeatvol_203_ma9_diff'] = X['logfeatvol_203'] - ma
        features.extend(['logvolume_sum_ma9_diff','volsumlog_ma9_diff','logfeatvol_203_ma9_diff'])
        features_to_scale.extend(['logvolume_sum_ma9_diff','volsumlog_ma9_diff','logfeatvol_203_ma9_diff'])
        #######################
        # resource types data #
        #######################
        nresources = pd.DataFrame(resource['id'].value_counts()).rename(columns={'id':'nresources'})
        X = pd.merge(X, nresources, right_index=True, left_index=True, how='left').fillna(0)
        features.append('nresources')
        features_to_scale.append('nresources')
        # one-hot common resources
        restypes = resource.resource_type.value_counts()
        common_resources = restypes.index
        if n_common_resources != -1:
            common_resources = common_resources[:n_common_resources]
        ohres = resource.loc[resource.resource_type.isin(common_resources)].groupby(['id','resource_type'])['resource_type'].count()
        ohres = ohres.unstack().fillna(0.).add_prefix('restype_')
        X = pd.merge(X, ohres, how='left', left_index=True, right_index=True).fillna(0.)
        features.extend(ohres.columns)

        # Probability features should be scaled too
        features_to_scale.extend(['loc_prob_{}'.format(i) for i in range(3)])

        self.features = features
        self.X = X
        self.features_to_scale = features_to_scale
        self.features_to_sparse_scale = features_to_sparse_scale

    def get_y(self):
        return self.data['train'].fault_severity

    def get_train_test_features(self, itrain = None, itest = None,
                            # aggregate features parameters
                            loc_agg_prior_weight = 3.):
        train = self.data['train']
        test = self.data['test']
        if itrain is None:
            # No indices into train, so return all train data
            itrain = train.index
        else:
            itrain = train.index[itrain]
        if itest is None:
            itest = test.index
        else:
            itest = train.index[itest]
        self.build_lagged_fault_features(itrain, itest)
        Xtr = self.X.loc[itrain, self.features]
        Xte = self.X.loc[itest, self.features]
        ytr = self.X.loc[itrain, 'fault_severity']
        yte = self.X.loc[itest,'fault_severity']
        aggXtr, aggXte = self.build_location_based_loo_aggregates(itrain,
                              itest, prior_weight = loc_agg_prior_weight)
        Xtr = pd.concat((Xtr, aggXtr), axis=1)
        Xte = pd.concat((Xte, aggXte), axis=1)

        # Include or exclude features
        if len(self.include) > 0:
            usefeatures = list(self.include)
        else:
            usefeatures = list(Xtr.columns)
        usefeatures = [f for f in usefeatures if f not in self.exclude]

        # Scaling
        to_scale = [f for f in usefeatures if f in self.features_to_scale]
        X = pd.concat((Xtr, Xte),axis=0)
        scaler = StandardScaler()
        scaler.fit(X[to_scale])
        Xtr[to_scale] = scaler.transform(Xtr[to_scale])
        Xte[to_scale] = scaler.transform(Xte[to_scale])
        to_sparse_scale = [f for f in usefeatures if f in self.features_to_sparse_scale]
        scaler = SparseRobustMaxScaler()
        scaler.fit(X[to_sparse_scale])
        Xtr[to_sparse_scale] = scaler.transform(Xtr[to_sparse_scale])
        Xte[to_sparse_scale] = scaler.transform(Xte[to_sparse_scale])
        return Xtr[usefeatures], ytr, Xte[usefeatures], yte

    def build_location_based_loo_aggregates(self, itrain, itest, prior_weight = 3.):
        """
        predict probability of response given location
        itrain - indices into train data
        itest - indices into test or out-of-fold data
        returns features df for train, df for test
        """
        train = self.X.loc[itrain,['location','fault_severity']]
        test = self.X.loc[itest,['location']]
        pred_cols = ['loc_prob_{}'.format(i) for i in range(3)]
        counts = train.groupby(['location','fault_severity'])['fault_severity'].count().unstack().fillna(0)
        counts.columns = pred_cols
        train_counts = pd.merge(train, counts, how='left',left_on = 'location', right_index = True)
        train_counts.loc[train_counts.fault_severity==0,'loc_prob_0'] -= 1
        train_counts.loc[train_counts.fault_severity==1,'loc_prob_1'] -= 1
        train_counts.loc[train_counts.fault_severity==2,'loc_prob_2'] -= 1
        prior = train.groupby('fault_severity')['fault_severity'].count().values.astype(np.float32)
        prior /= np.sum(prior)
        feats_tr = train_counts[pred_cols].add(prior * prior_weight,axis=1)
        feats_tr = feats_tr.div(feats_tr.sum(axis=1), axis=0)
        feats_tr.loc[feats_tr.loc_prob_0.isnull(),pred_cols] = prior

        test_counts = pd.merge(test, counts, how='left',left_on = 'location', right_index = True)
        feats_te = test_counts[pred_cols].add(prior * prior_weight,axis=1)
        feats_te = feats_te.div(feats_te.sum(axis=1), axis=0)
        feats_te.loc[feats_te.loc_prob_0.isnull(),pred_cols] = prior
        return feats_tr[pred_cols], feats_te[pred_cols]

    def build_lagged_fault_features(self, itrain, itest):
        self.X.fault_severity = self.data['train'].fault_severity
        X = self.X
        X['lastknown'] = self.data['train'].fault_severity.loc[itrain]
        def ewma(fs, halflife):
            return lambda x: [0]+pd.ewma(x==fs,halflife = halflife,min_periods=1)[:-1].tolist()
        X['ewma02'] = X.groupby('location')['lastknown'].transform(ewma(0,2))
        X['ewma12'] = X.groupby('location')['lastknown'].transform(ewma(1,2))
        X['ewma22'] = X.groupby('location')['lastknown'].transform(ewma(2,2))

        X['lastknown'] = X.groupby('location')['lastknown'].fillna(method='ffill').fillna(0)
        X['lastknown'] = X.groupby('location')['lastknown'].transform(lambda x: [0]+x.iloc[:-1].tolist())
        X['nextknown'] = self.data['train'].fault_severity.loc[itrain]
        X['nextknown'] = X.groupby('location')['nextknown'].fillna(method='bfill').fillna(0)
        X['nextknown'] = X.groupby('location')['nextknown'].transform(lambda x: x.iloc[1:].tolist()+[0])
        if 'lastknown' not in self.features:
            self.features.extend(['lastknown','nextknown'])
            self.features.extend(['ewma02','ewma12','ewma22'])

def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        true class, intergers in [0, n_classes - 1)
    y_pred : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    n_samples = actual.shape[0]
    actual[np.arange(n_samples), y_true.astype(int)] = 1
    vectsum = np.sum(actual * np.log(predictions))
    loss = -1.0 / n_samples * vectsum
    return loss
