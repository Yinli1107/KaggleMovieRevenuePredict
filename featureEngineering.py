import prepareData
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
import ast

def feature_engineering(df):
	# 把year抽出来
	df['release_year'] = df['release_date'].apply(lambda x: 20 if pd.isna(x) else int(x.split('/')[2]))
	df['release_year'] = df['release_year'].apply(lambda x:
	                                                    x + 2000 if (x <= 19 and x < 100) else x)
	df['release_year'] = df['release_year'].apply(lambda x:
	                                                    x + 1900 if (x > 19 and x < 100) else x)

	min_year= df['release_year'].min()
	print(min_year)
	df['release_year'] = df['release_year'].apply(lambda x: x-min_year)

	valid_budget_num = df.shape[0] - (df['budget'] == 0).astype(int).sum()
	print(valid_budget_num)

	median_budget = df['budget'].sum()/valid_budget_num
	print(median_budget)

	df['budget'] = df['budget'].apply(lambda x: median_budget if x==0 or pd.isna(x) else x)
	# 把budget取log，构造新特征
	df['budget'] = np.log1p(df['budget'])
	df['_budget_year_ratio'] = df['budget']/(df['release_year'])

	# 是否有collection信息
	df['has_collection'] = df['belongs_to_collection'].apply(lambda x:0 if pd.isna(x) else 1)

	# 电影的类别数量
	df['genres'] = df['genres'].apply(lambda x:{} if pd.isna(x) else ast.literal_eval(x))
	df['genres_num'] = df['genres'].apply(lambda x:len(x) if x!={} else 0)

	df['has_homepage'] = df['homepage'].apply(lambda x:0 if pd.isna(x) else 1)

	# original_language是否为英语
	df['is_english'] = df['original_language'].apply(lambda x: 1 if x=='en' else 0)
	# keywords的数目
	df['Keywords'] = df['Keywords'].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x))
	df['keywords_num'] = df['Keywords'].apply(lambda x:len(x) if x!={} else 0)

	df['_rating_per_votenum'] = df['rating'] / df['totalVotes']
	df['_rating_times_popularity'] = df['rating'] * df['popularity']
	df['_votes_per_budget'] = df['totalVotes']/df['budget']

	# 去掉没用的特征
	df = df.drop(['belongs_to_collection','genres','homepage','imdb_id','original_language',
	              'original_title','overview','poster_path','production_companies','production_countries',
	              'release_date','spoken_languages','status','tagline','title','Keywords','cast','crew'],axis=1)

	# 用中值填充缺失值，也就是popularity2、rating和totalVotes
	imputer = Imputer(strategy='median')
	imputer.fit(df)
	X = imputer.transform(df)
	df = pd.DataFrame(X, columns=df.columns)

	return df

def processData():
	train,test = prepareData.getData()
	train = feature_engineering(train)
	test = feature_engineering(test)
	return train,test
