import numpy as np
import pandas as pd
import xgboost as xgb

with_pd = True

print("xgboost version: ", xgb.__version__)
xgb_reg = xgb.XGBRegressor(n_estimators=100, max_depth=10, eta=0.1, subsample=0.7, colsample_bytree=0.8)
x_train = np.random.sample([1000,2])
x_test = np.random.sample([20,2])
y_train = x_train[:,1]
y_test = x_test[:,1]
if with_pd:
    x_train = pd.DataFrame(data=x_train, columns=['f1','f2'])
    y_train = pd.DataFrame(data=y_train, columns=['target'])
    x_test = pd.DataFrame(data=x_test, columns=['f1','f2'])
    y_test = pd.DataFrame(data=y_test, columns=['target'])
xgb_reg.fit(x_train, y_train)

y_train_pred = xgb_reg.predict(x_train)
print("train target mean square: ", np.mean(np.power(y_train, 2)))
print("train MSE: ", np.mean(np.power(y_train_pred-np.array(y_train).flatten(), 2)))
y_test_pred = xgb_reg.predict(x_test)
print("test target mean square: ", np.mean(np.power(y_test, 2)))
print("test MSE: ", np.mean(np.power(y_test_pred-np.array(y_test).flatten(), 2)))

print("importance scores")
print("feature_importances_: ", xgb_reg.feature_importances_)
for impt_type in ['weight', 'cover', 'gain', 'total_gain', 'total_cover']:
    res = xgb_reg.get_booster().get_score(importance_type=impt_type)
    print(f"booster {impt_type}: ", res)
    if impt_type=='gain':
        print("gain normalized: ", res['f1']/(res['f1']+res['f2']), 1-res['f1']/(res['f1']+res['f2']))