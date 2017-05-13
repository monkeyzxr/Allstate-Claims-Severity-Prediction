import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.cross_validation import train_test_split as tts
from sklearn.metrics import mean_absolute_error as mae
from sklearn import linear_model as lm
from sklearn import ensemble as en
import xgboost as xgb

location = '/Users/monkeyzxr/Desktop/CS 6375.002 - Machine Learning/Project/dataset from kaggle'
train = pd.read_csv(os.path.join(location,'train.csv'))
# test = pd.read_csv(os.path.join(location,'test.csv'))
del train['id']
##ADD:#########################
del train['cont12']
del train['cont9']
#################################

df_train, df_test = tts(train,
									test_size = 0.2,
									random_state = 123)

shift = 1500 
# shift = 50   MAE is : 1280.57149767
# shift = 100  MAE is : 1277.98974664
# shift = 150  MAE is : 1275.82563592
# shift = 200  MAE is : 1273.97832532
# shift = 250  MAE is : 1272.38697447
# shift = 300  MAE is : 1271.00173961
# shift = 1000 MAE is : 1262.74242893
# shift = 1500 MAE is : 1262.39066045 ==> best
# shift = 1750 MAE is : 1262.80001391
# shift = 2000 MAE is : 1263.38130528
# shift = 5000 MAE is : 1275.38232042

train_Y = np.log10(df_train['loss'] + shift)
# test_Y = np.log10(df_test['loss'] + shift)

features = df_train.columns
cat_feature = list(features[0:116])
##cont_feature = list(features[117:130])   ADD:##########
cont_feature = list(features[117:128])
#################################################

for i in cat_feature:
    df_train[i] = pd.factorize(df_train[i], sort = True)[0]
    df_test[i] = pd.factorize(df_test[i], sort = True)[0]
    # test[i] = pd.factorize(test[i], sort = True)[0]

#train_X = df_train.iloc[:, 0:130]
#test_X = df_test.iloc[:, 0:130]  ADD:
###########################################
train_X = df_train.iloc[:, 0:128]
test_X = df_test.iloc[:, 0:128]
##################################################

lr = lm.LinearRegression(fit_intercept=True, normalize=True)
model = lr.fit(train_X, train_Y)
Prediction = np.power(10, (model.predict(test_X))) - shift
MAE = mae(df_test['loss'], Prediction)
print('MAE is : ' + str(MAE))
# MAE is : 1262.39066045
##After delete 2 attributes, MAE is 1262.90203841. !! increase 0.511378

ridge = lm.Ridge(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None)
model = ridge.fit(train_X, train_Y)
Prediction = np.power(10, (model.predict(test_X))) - shift
MAE = mae(df_test['loss'], Prediction)
print('MAE is : ' + str(MAE))
# MAE is : 1262.3032256
##After delete 2 attributes, MAE is 1262.76513713. !! increase  0.4619115

lasso = lm.LassoLarsCV(fit_intercept=True, verbose=False, max_iter=500, normalize=True, precompute='auto', cv=None, max_n_alphas=1000, n_jobs=1, eps=2.2204460492503131e-16, copy_X=True, positive=False)
model = lasso.fit(train_X, train_Y)
Prediction = np.power(10, (model.predict(test_X))) - shift
MAE = mae(df_test['loss'], Prediction)
print('MAE is : ' + str(MAE))
# MAE is : 1257.91678327
##After delete 2 attributes, MAE is 1257.90438593. !! Only decrease  0.01239734

ada = en.AdaBoostRegressor(base_estimator=None, n_estimators=50, learning_rate=1.0, loss='linear', random_state=None)
model = ada.fit(train_X, train_Y)
Prediction = np.power(10, (model.predict(test_X))) - shift
MAE = mae(df_test['loss'], Prediction)
print('MAE is : ' + str(MAE))
# MAE is : 1705.48186145
##After delete 2 attributes, MAE is 1724.64523341 !! increase 19.16337, very weird

ada = en.AdaBoostRegressor(base_estimator=None, n_estimators=50, learning_rate=0.1, loss='linear', random_state=None)
model = ada.fit(train_X, train_Y)
Prediction = np.power(10, (model.predict(test_X))) - shift
MAE = mae(df_test['loss'], Prediction)
print('MAE is : ' + str(MAE))
# MAE is : 1463.55713854
##After delete 2 attributes, MAE is 1464.40019611 !! increase 0.84305757

ada = en.AdaBoostRegressor(base_estimator=None, n_estimators=100, learning_rate=0.1, loss='linear', random_state=None)
model = ada.fit(train_X, train_Y)
Prediction = np.power(10, (model.predict(test_X))) - shift
MAE = mae(df_test['loss'], Prediction)
print('MAE is : ' + str(MAE))
# MAE is : 1491.76377706
##After delete 2 attributes, MAE is 1493.29752146 !! increase 1.5337444

ada = en.AdaBoostRegressor(base_estimator=None, n_estimators=50, learning_rate=0.1, loss='square', random_state=None)
model = ada.fit(train_X, train_Y)
Prediction = np.power(10, (model.predict(test_X))) - shift
MAE = mae(df_test['loss'], Prediction)
print('MAE is : ' + str(MAE))
# MAE is : 1489.18724058
##After delete 2 attributes, MAE is 1482.03027309 !! decrease 7.15696749

ada = en.AdaBoostRegressor(base_estimator=None, n_estimators=10, learning_rate=0.1, loss='linear', random_state=None)
model = ada.fit(train_X, train_Y)
Prediction = np.power(10, (model.predict(test_X))) - shift
MAE = mae(df_test['loss'], Prediction)
print('MAE is : ' + str(MAE))
# MAE is : 1454.44577686
##After delete 2 attributes, MAE is 1450.23754455 !! decrease 4.20823231


#rf = en.RandomForestRegressor(n_estimators=10, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)
rf = en.RandomForestRegressor(n_estimators=10, criterion='mse', max_depth=None,  min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,  bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)
model = rf.fit(train_X, train_Y)
Prediction = np.power(10, (model.predict(test_X))) - shift
MAE = mae(df_test['loss'], Prediction)
print('MAE is : ' + str(MAE))
# MAE is : 1276.87548015
##After delete 2 attributes, MAE is 1284.46139457 !! increase 7.58591442

rf = en.RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)
model = rf.fit(train_X, train_Y)
Prediction = np.power(10, (model.predict(test_X))) - shift
MAE = mae(df_test['loss'], Prediction)
print('MAE is : ' + str(MAE))
# MAE is : 1221.81499823

rf = en.RandomForestRegressor(n_estimators=10, criterion='mae', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)
model = rf.fit(train_X, train_Y)
Prediction = np.power(10, (model.predict(test_X))) - shift
MAE = mae(df_test['loss'], Prediction)
print('MAE is : ' + str(MAE))
# did not work

rf = en.RandomForestRegressor(n_estimators=10, criterion='mse', max_depth=5, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)
model = rf.fit(train_X, train_Y)
Prediction = np.power(10, (model.predict(test_X))) - shift
MAE = mae(df_test['loss'], Prediction)
print('MAE is : ' + str(MAE))
# MAE is : 1381.23997869

bag = en.BaggingRegressor(base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)
model = bag.fit(train_X, train_Y)
Prediction = np.power(10, (model.predict(test_X))) - shift
MAE = mae(df_test['loss'], Prediction)
print('MAE is : ' + str(MAE))
# MAE is : 1276.92930777
##After delete 2 attributes, MAE is 1278.2624438 !! increase 1.33313603

bag = en.BaggingRegressor(base_estimator=None, n_estimators=100, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)
model = bag.fit(train_X, train_Y)
Prediction = np.power(10, (model.predict(test_X))) - shift
MAE = mae(df_test['loss'], Prediction)
print('MAE is : ' + str(MAE))
# MAE is : 1222.12816712
##After delete 2 attributes, MAE is 1222.20585524 !! increase 0.007768528

#gbr = en.GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=10, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_split=1e-07, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')
gbr = en.GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=10, subsample=1.0,  min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3,  init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')
model = gbr.fit(train_X, train_Y)
Prediction = np.power(10, (model.predict(test_X))) - shift
MAE = mae(df_test['loss'], Prediction)
print('MAE is : ' + str(MAE))
# MAE is : 1173.85455634

gbr = en.GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_split=1e-07, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')
model = gbr.fit(train_X, train_Y)
Prediction = np.power(10, (model.predict(test_X))) - shift
MAE = mae(df_test['loss'], Prediction)
print('MAE is : ' + str(MAE))
# MAE is : 1215.56216263

params = {}
params['booster'] = 'gbtree'
params['objective'] = "reg:linear"
params['eval_metric'] = 'mae'
params['eta'] = 1
params['gamma'] = 1
params['min_child_weight'] = 1
params['colsample_bytree'] = 1
params['subsample'] = 1
params['max_depth'] = 1
params['max_delta_step'] = 0
params['silent'] = 1
params['random_state'] = 123

xgtrain = xgb.DMatrix(train_X, label = train_Y)
xgtest = xgb.DMatrix(test_X)
model = xgb.train(params, xgtrain, 1000)
Prediction = np.power(10, (model.predict(xgtest))) - shift
MAE = mae(df_test['loss'], Prediction)
print('MAE is : ' + str(MAE))
# MAE is : 1237.0680729

params = {}
params['booster'] = 'gblinear'
params['objective'] = "reg:linear"
params['eval_metric'] = 'mae'
params['eta'] = 1
params['gamma'] = 1
params['min_child_weight'] = 1
params['colsample_bytree'] = 1
params['subsample'] = 1
params['max_depth'] = 1
params['max_delta_step'] = 0
params['silent'] = 1
params['random_state'] = 123

xgtrain = xgb.DMatrix(train_X, label = train_Y)
xgtest = xgb.DMatrix(test_X)
model = xgb.train(params, xgtrain, 1000)
Prediction = np.power(10, (model.predict(xgtest))) - shift
MAE = mae(df_test['loss'], Prediction)
print('MAE is : ' + str(MAE))
# MAE is : 1262.18196267

params = {}
params['booster'] = 'dart'
params['objective'] = "reg:linear"
params['eval_metric'] = 'mae'
params['eta'] = 1
params['gamma'] = 1
params['min_child_weight'] = 1
params['colsample_bytree'] = 1
params['subsample'] = 1
params['max_depth'] = 1
params['max_delta_step'] = 0
params['silent'] = 1
params['random_state'] = 123

xgtrain = xgb.DMatrix(train_X, label = train_Y)
xgtest = xgb.DMatrix(test_X)
model = xgb.train(params, xgtrain, 1000)
Prediction = np.power(10, (model.predict(xgtest))) - shift
MAE = mae(df_test['loss'], Prediction)
print('MAE is : ' + str(MAE))
# MAE is : 1237.06804753

params = {}
params['booster'] = 'gbtree'
params['objective'] = "reg:linear"
params['eval_metric'] = 'mae'
params['eta'] = 0.5
params['gamma'] = 1
params['min_child_weight'] = 1
params['colsample_bytree'] = 1
params['subsample'] = 1
params['max_depth'] = 1
params['max_delta_step'] = 0
params['silent'] = 1
params['random_state'] = 123

xgtrain = xgb.DMatrix(train_X, label = train_Y)
xgtest = xgb.DMatrix(test_X)
model = xgb.train(params, xgtrain, 1000)
Prediction = np.power(10, (model.predict(xgtest))) - shift
MAE = mae(df_test['loss'], Prediction)
print('MAE is : ' + str(MAE))
# MAE is : 1235.43971988

params = {}
params['booster'] = 'gbtree'
params['objective'] = "reg:linear"
params['eval_metric'] = 'mae'
params['eta'] = 0.1
params['gamma'] = 1
params['min_child_weight'] = 1
params['colsample_bytree'] = 1
params['subsample'] = 1
params['max_depth'] = 1
params['max_delta_step'] = 0
params['silent'] = 1
params['random_state'] = 123

xgtrain = xgb.DMatrix(train_X, label = train_Y)
xgtest = xgb.DMatrix(test_X)
model = xgb.train(params, xgtrain, 1000)
Prediction = np.power(10, (model.predict(xgtest))) - shift
MAE = mae(df_test['loss'], Prediction)
print('MAE is : ' + str(MAE))
# MAE is : 1239.43181345

params = {}
params['booster'] = 'gbtree'
params['objective'] = "reg:linear"
params['eval_metric'] = 'mae'
params['eta'] = 0.2
params['gamma'] = 1
params['min_child_weight'] = 1
params['colsample_bytree'] = 1
params['subsample'] = 1
params['max_depth'] = 1
params['max_delta_step'] = 0
params['silent'] = 1
params['random_state'] = 123

xgtrain = xgb.DMatrix(train_X, label = train_Y)
xgtest = xgb.DMatrix(test_X)
model = xgb.train(params, xgtrain, 1000)
Prediction = np.power(10, (model.predict(xgtest))) - shift
MAE = mae(df_test['loss'], Prediction)
print('MAE is : ' + str(MAE))
# MAE is : 1238.13441131

params = {}
params['booster'] = 'gbtree'
params['objective'] = "reg:linear"
params['eval_metric'] = 'mae'
params['eta'] = 0.01
params['gamma'] = 1
params['min_child_weight'] = 1
params['colsample_bytree'] = 1
params['subsample'] = 1
params['max_depth'] = 1
params['max_delta_step'] = 0
params['silent'] = 1
params['random_state'] = 123

xgtrain = xgb.DMatrix(train_X, label = train_Y)
xgtest = xgb.DMatrix(test_X)
model = xgb.train(params, xgtrain, 1000)
Prediction = np.power(10, (model.predict(xgtest))) - shift
MAE = mae(df_test['loss'], Prediction)
print('MAE is : ' + str(MAE))
# MAE is : 1363.33451984

params = {}
params['booster'] = 'gbtree'
params['objective'] = "reg:linear"
params['eval_metric'] = 'mae'
params['eta'] = 0.1
params['gamma'] = 1
params['min_child_weight'] = 1
params['colsample_bytree'] = 1
params['subsample'] = 1
params['max_depth'] = 1
params['max_delta_step'] = 0
params['silent'] = 1
params['random_state'] = 123

xgtrain = xgb.DMatrix(train_X, label = train_Y)
xgtest = xgb.DMatrix(test_X)
model = xgb.train(params, xgtrain, 10000)
Prediction = np.power(10, (model.predict(xgtest))) - shift
MAE = mae(df_test['loss'], Prediction)
print('MAE is : ' + str(MAE))
# MAE is : 1239.43181345

params = {}
params['booster'] = 'gbtree'
params['objective'] = "reg:linear"
params['eval_metric'] = 'mae'
params['eta'] = 0.1
params['gamma'] = 0.5
params['min_child_weight'] = 1
params['colsample_bytree'] = 1
params['subsample'] = 1
params['max_depth'] = 1
params['max_delta_step'] = 0
params['silent'] = 1
params['random_state'] = 123

xgtrain = xgb.DMatrix(train_X, label = train_Y)
xgtest = xgb.DMatrix(test_X)
model = xgb.train(params, xgtrain, 10000)
Prediction = np.power(10, (model.predict(xgtest))) - shift
MAE = mae(df_test['loss'], Prediction)
print('MAE is : ' + str(MAE))
# MAE is : 1231.97155938

params = {}
params['booster'] = 'gbtree'
params['objective'] = "reg:linear"
params['eval_metric'] = 'mae'
params['eta'] = 0.1
params['gamma'] = 0.25
params['min_child_weight'] = 1
params['colsample_bytree'] = 1
params['subsample'] = 1
params['max_depth'] = 1
params['max_delta_step'] = 0
params['silent'] = 1
params['random_state'] = 123

xgtrain = xgb.DMatrix(train_X, label = train_Y)
xgtest = xgb.DMatrix(test_X)
model = xgb.train(params, xgtrain, 10000)
Prediction = np.power(10, (model.predict(xgtest))) - shift
MAE = mae(df_test['loss'], Prediction)
print('MAE is : ' + str(MAE))
# MAE is : 1226.8768663

params = {}
params['booster'] = 'gbtree'
params['objective'] = "reg:linear"
params['eval_metric'] = 'mae'
params['eta'] = 0.1
params['gamma'] = 0.125
params['min_child_weight'] = 1
params['colsample_bytree'] = 1
params['subsample'] = 1
params['max_depth'] = 1
params['max_delta_step'] = 0
params['silent'] = 1
params['random_state'] = 123

xgtrain = xgb.DMatrix(train_X, label = train_Y)
xgtest = xgb.DMatrix(test_X)
model = xgb.train(params, xgtrain, 10000)
Prediction = np.power(10, (model.predict(xgtest))) - shift
MAE = mae(df_test['loss'], Prediction)
print('MAE is : ' + str(MAE))
# MAE is : 1223.45410247

params = {}
params['booster'] = 'gbtree'
params['objective'] = "reg:linear"
params['eval_metric'] = 'mae'
params['eta'] = 0.1
params['gamma'] = 0.01
params['min_child_weight'] = 1
params['colsample_bytree'] = 1
params['subsample'] = 1
params['max_depth'] = 1
params['max_delta_step'] = 0
params['silent'] = 1
params['random_state'] = 123

xgtrain = xgb.DMatrix(train_X, label = train_Y)
xgtest = xgb.DMatrix(test_X)
model = xgb.train(params, xgtrain, 10000)
Prediction = np.power(10, (model.predict(xgtest))) - shift
MAE = mae(df_test['loss'], Prediction)
print('MAE is : ' + str(MAE))
# MAE is : 1220.75410897

params = {}
params['booster'] = 'gbtree'
params['objective'] = "reg:linear"
params['eval_metric'] = 'mae'
params['eta'] = 0.1
params['gamma'] = 0
params['min_child_weight'] = 1
params['colsample_bytree'] = 1
params['subsample'] = 1
params['max_depth'] = 1
params['max_delta_step'] = 0
params['silent'] = 1
params['random_state'] = 123

xgtrain = xgb.DMatrix(train_X, label = train_Y)
xgtest = xgb.DMatrix(test_X)
model = xgb.train(params, xgtrain, 10000)
Prediction = np.power(10, (model.predict(xgtest))) - shift
MAE = mae(df_test['loss'], Prediction)
print('MAE is : ' + str(MAE))
# MAE is : 1220.16999408

params = {}
params['booster'] = 'gbtree'
params['objective'] = "reg:linear"
params['eval_metric'] = 'mae'
params['eta'] = 0.1
params['gamma'] = 0
params['min_child_weight'] = 1
params['colsample_bytree'] = 1
params['subsample'] = 1
params['max_depth'] = 6
params['max_delta_step'] = 0
params['silent'] = 1
params['random_state'] = 123

xgtrain = xgb.DMatrix(train_X, label = train_Y)
xgtest = xgb.DMatrix(test_X)
model = xgb.train(params, xgtrain, 10000)
Prediction = np.power(10, (model.predict(xgtest))) - shift
MAE = mae(df_test['loss'], Prediction)
print('MAE is : ' + str(MAE))
# MAE is : 1255.37754604

params = {}
params['booster'] = 'gbtree'
params['objective'] = "reg:linear"
params['eval_metric'] = 'mae'
params['eta'] = 0.1
params['gamma'] = 0
params['min_child_weight'] = 1
params['colsample_bytree'] = 1
params['subsample'] = 1
params['max_depth'] = 5
params['max_delta_step'] = 0
params['silent'] = 1
params['random_state'] = 123

xgtrain = xgb.DMatrix(train_X, label = train_Y)
xgtest = xgb.DMatrix(test_X)
model = xgb.train(params, xgtrain, 10000)
Prediction = np.power(10, (model.predict(xgtest))) - shift
MAE = mae(df_test['loss'], Prediction)
print('MAE is : ' + str(MAE))
# MAE is : 1209.7856336

params = {}
params['booster'] = 'gbtree'
params['objective'] = "reg:linear"
params['eval_metric'] = 'mae'
params['eta'] = 0.1
params['gamma'] = 0
params['min_child_weight'] = 1
params['colsample_bytree'] = 1
params['subsample'] = 1
params['max_depth'] = 4
params['max_delta_step'] = 0
params['silent'] = 1
params['random_state'] = 123

xgtrain = xgb.DMatrix(train_X, label = train_Y)
xgtest = xgb.DMatrix(test_X)
model = xgb.train(params, xgtrain, 10000)
Prediction = np.power(10, (model.predict(xgtest))) - shift
MAE = mae(df_test['loss'], Prediction)
print('MAE is : ' + str(MAE))
# MAE is : 1225.5653299

params = {}
params['booster'] = 'gbtree'
params['objective'] = "reg:linear"
params['eval_metric'] = 'mae'
params['eta'] = 0.1
params['gamma'] = 0
params['min_child_weight'] = 1
params['colsample_bytree'] = 0.5
params['subsample'] = 1
params['max_depth'] = 5
params['max_delta_step'] = 0
params['silent'] = 1
params['random_state'] = 123

xgtrain = xgb.DMatrix(train_X, label = train_Y)
xgtest = xgb.DMatrix(test_X)
model = xgb.train(params, xgtrain, 10000)
Prediction = np.power(10, (model.predict(xgtest))) - shift
MAE = mae(df_test['loss'], Prediction)
print('MAE is : ' + str(MAE))
# MAE is : 1204.71512827

params = {}
params['booster'] = 'gbtree'
params['objective'] = "reg:linear"
params['eval_metric'] = 'mae'
params['eta'] = 0.1
params['gamma'] = 0
params['min_child_weight'] = 1
params['colsample_bytree'] = 0.25
params['subsample'] = 1
params['max_depth'] = 5
params['max_delta_step'] = 0
params['silent'] = 1
params['random_state'] = 123

xgtrain = xgb.DMatrix(train_X, label = train_Y)
xgtest = xgb.DMatrix(test_X)
model = xgb.train(params, xgtrain, 5000)
Prediction = np.power(10, (model.predict(xgtest))) - shift
MAE = mae(df_test['loss'], Prediction)
print('MAE is : ' + str(MAE))
# MAE is : 1179.26364201

params = {}
params['booster'] = 'gbtree'
params['objective'] = "reg:linear"
params['eval_metric'] = 'mae'
params['eta'] = 0.1
params['gamma'] = 0
params['min_child_weight'] = 1
params['colsample_bytree'] = 0.25
params['subsample'] = 1
params['max_depth'] = 5
params['max_delta_step'] = 0
params['silent'] = 1
params['random_state'] = 123

xgtrain = xgb.DMatrix(train_X, label = train_Y)
xgtest = xgb.DMatrix(test_X)
model = xgb.train(params, xgtrain, 2500)
Prediction = np.power(10, (model.predict(xgtest))) - shift
MAE = mae(df_test['loss'], Prediction)
print('MAE is : ' + str(MAE))
# MAE is : 1167.47685787

params = {}
params['booster'] = 'gbtree'
params['objective'] = "reg:linear"
params['eval_metric'] = 'mae'
params['eta'] = 0.1
params['gamma'] = 0
params['min_child_weight'] = 1
params['colsample_bytree'] = 0.25
params['subsample'] = 1
params['max_depth'] = 5
params['max_delta_step'] = 0
params['silent'] = 1
params['random_state'] = 123

xgtrain = xgb.DMatrix(train_X, label = train_Y)
xgtest = xgb.DMatrix(test_X)
model = xgb.train(params, xgtrain, 2000)
Prediction = np.power(10, (model.predict(xgtest))) - shift
MAE = mae(df_test['loss'], Prediction)
print('MAE is : ' + str(MAE))
# MAE is : 1164.0802791

params = {}
params['booster'] = 'gbtree'
params['objective'] = "reg:linear"
params['eval_metric'] = 'mae'
params['eta'] = 0.1
params['gamma'] = 0
params['min_child_weight'] = 1
params['colsample_bytree'] = 0.25
params['subsample'] = 1
params['max_depth'] = 5
params['max_delta_step'] = 0
params['silent'] = 1
params['random_state'] = 123

xgtrain = xgb.DMatrix(train_X, label = train_Y)
xgtest = xgb.DMatrix(test_X)
model = xgb.train(params, xgtrain, 1000)
Prediction = np.power(10, (model.predict(xgtest))) - shift
MAE = mae(df_test['loss'], Prediction)
print('MAE is : ' + str(MAE))
# MAE is : 1158.73970776

params = {}
params['booster'] = 'gbtree'
params['objective'] = "reg:linear"
params['eval_metric'] = 'mae'
params['eta'] = 0.1
params['gamma'] = 0
params['min_child_weight'] = 1
params['colsample_bytree'] = 0.5
params['subsample'] = 1
params['max_depth'] = 5
params['max_delta_step'] = 0
params['silent'] = 1
params['random_state'] = 123

xgtrain = xgb.DMatrix(train_X, label = train_Y)
xgtest = xgb.DMatrix(test_X)
model = xgb.train(params, xgtrain, 1000)
Prediction = np.power(10, (model.predict(xgtest))) - shift
MAE = mae(df_test['loss'], Prediction)
print('MAE is : ' + str(MAE))
# MAE is : 1159.56401091

params = {}
params['booster'] = 'gbtree'
params['objective'] = "reg:linear"
params['eval_metric'] = 'mae'
params['eta'] = 0.1
params['gamma'] = 0
params['min_child_weight'] = 1
params['colsample_bytree'] = 0.1
params['subsample'] = 1
params['max_depth'] = 5
params['max_delta_step'] = 0
params['silent'] = 1
params['random_state'] = 123

xgtrain = xgb.DMatrix(train_X, label = train_Y)
xgtest = xgb.DMatrix(test_X)
model = xgb.train(params, xgtrain, 1000)
Prediction = np.power(10, (model.predict(xgtest))) - shift
MAE = mae(df_test['loss'], Prediction)
print('MAE is : ' + str(MAE))
# MAE is : 1155.43585165

params = {}
params['booster'] = 'gbtree'
params['objective'] = "reg:linear"
params['eval_metric'] = 'mae'
params['eta'] = 0.1
params['gamma'] = 0
params['min_child_weight'] = 1
params['colsample_bytree'] = 0.1
params['subsample'] = 0.7
params['max_depth'] = 5
params['max_delta_step'] = 0
params['silent'] = 1
params['random_state'] = 123

xgtrain = xgb.DMatrix(train_X, label = train_Y)
xgtest = xgb.DMatrix(test_X)
model = xgb.train(params, xgtrain, 1000)
Prediction = np.power(10, (model.predict(xgtest))) - shift
MAE = mae(df_test['loss'], Prediction)
print('MAE is : ' + str(MAE))
# MAE is : 1180.59057663

params = {}
params['booster'] = 'gbtree'
params['objective'] = "reg:linear"
params['eval_metric'] = 'mae'
params['eta'] = 0.1
params['gamma'] = 0
params['min_child_weight'] = 2
params['colsample_bytree'] = 0.1
params['subsample'] = 1
params['max_depth'] = 5
params['max_delta_step'] = 0
params['silent'] = 1
params['random_state'] = 123

xgtrain = xgb.DMatrix(train_X, label = train_Y)
xgtest = xgb.DMatrix(test_X)
model = xgb.train(params, xgtrain, 1000)
Prediction = np.power(10, (model.predict(xgtest))) - shift
MAE = mae(df_test['loss'], Prediction)
print('MAE is : ' + str(MAE))
# MAE is : 1160.69545693
