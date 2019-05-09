import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import catboost as cat

import featureEngineering

def model(train,test):
	Kfolder = KFoldValidation(train)
	lgbmodel = lgb.LGBMRegressor(n_estimators=10000,
	                             objective='regression',
	                             metric='rmse',
	                             max_depth=5,
	                             num_leaves=30,
	                             min_child_samples=100,
	                             learning_rate=0.01,
	                             boosting='gbdt',
	                             min_data_in_leaf=10,
	                             feature_fraction=0.9,
	                             bagging_freq=1,
	                             bagging_fraction=0.9,
	                             importance_type='gain',
	                             lambda_l1=0.2,
	                             bagging_seed=2019,
	                             subsample=.8,
	                             colsample_bytree=.9,
	                             use_best_model=True)

	catmodel = cat.CatBoostRegressor(iterations=10000,
	                                 learning_rate=0.01,
	                                 depth=5,
	                                 eval_metric='RMSE',
	                                 colsample_bylevel=0.8,
	                                 bagging_temperature=0.2,
	                                 metric_period=None,
	                                 early_stopping_rounds=200,
	                                 random_seed=100)

	features = list(train.columns)
	features = [i for i in features if (i != 'id' and i != 'revenue')]
	Kfolder.validate(train, test, features, lgbmodel, name="lgbfinal", prepare_stacking=True)
	#Kfolder.validate(train, test, features, catmodel, name='catfinal', prepare_stacking=True,
	#                fit_params={"use_best_model": True, "verbose": 100})

	#test['revenue'] = np.expm1(test['lgbfinal']*0.5 + test['catfinal']*0.5)
	#test['id'] = test['id'].apply(lambda x:int(x))
	#test[['id', 'revenue']].to_csv('./data/submission.csv', index=False)


# data中的revenue应该是正确revenue
# y是预测的revenue，是log后的
def score(data, y):
	# 用一个dataframe保存id和对应的正确revenue以及预测revenue
	validation_res = pd.DataFrame(
		{'id': data['id'].values,
		 'transaction_revenue': data['revenue'].values,
		 'predict_revenue':np.expm1(y)}
	)

	validation_res = validation_res.groupby('id')['transaction_revenue','predict_revenue'].sum().reset_index()
	return np.sqrt(mean_squared_error(np.log1p(validation_res['transaction_revenue'].values),
	                                  np.log1p(validation_res['predict_revenue'].values)))

class KFoldValidation():
	# 类初始化
	def __init__(self, data, n_splits=5):
		# 得到唯一的id数组
		unique_vis = np.array(sorted(data['id'].astype(str).unique()))
		folds = GroupKFold(n_splits)
		ids = np.arange(data.shape[0])

		# 用来记录每一折的训练id和验证id
		self.fold_ids = []

		for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
			self.fold_ids.append([
				ids[data['id'].astype(str).isin(unique_vis[trn_vis])],
				ids[data['id'].astype(str).isin(unique_vis[val_vis])]
			])

	# 验证
	def validate(self,train,test,features, model, name='', prepare_stacking=False,
	             fit_params={"early_stopping_rounds":500, "verbose": 100, "eval_metric": "rmse"}):

		full_score = 0

		if prepare_stacking:
			test[name] = 0
			train[name] = np.NAN

		for fold_id, (trn, val) in enumerate(self.fold_ids):
			# 构造训练特征和值以及验证特征和值
			devel = train[features].iloc[trn]
			y_devel = np.log1p(train['revenue'].iloc[trn])
			valid = train[features].iloc[val]
			y_valid = np.log1p(train['revenue'].iloc[val])

			print('Fold ', fold_id, ':')
			model.fit(devel, y_devel, eval_set = [(valid, y_valid)], **fit_params)

			#if(len(model.feature_importance_) == len(features)):
			#	model.FI['fold' + str(fold_id)] = model.feature_importances_ / model.feature_importances_.sum()

			predictions = model.predict(valid)
			predictions[predictions < 0] = 0
			print("Fold ", fold_id, " error: ", mean_squared_error(y_valid, predictions) ** 0.5)

			fold_score =score(train.iloc[val], predictions)
			full_score += fold_score / len(self.fold_ids)
			print("Fold ", fold_id, " score: ", fold_score)

			if(prepare_stacking):
				train[name].iloc[val] = predictions
				test_predictions = model.predict(test[features])
				test_predictions[test_predictions < 0] = 0
				test[name] += test_predictions/len(self.fold_ids)

		print('Final Score:',full_score)
		return full_score

def main():
	train,test = featureEngineering.processData()
	model(train,test)

	#test.info()
	#train.info()

if __name__ == '__main__':
	main()
