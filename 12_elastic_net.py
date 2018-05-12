# ElasticNet regression
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import ElasticNet

filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
'B', 'LSTAT', 'MEDV']
data = read_csv(filename, delim_whitespace=True, names=names)
array = data.values
X = array[:, :13]
y = array[:, 13]
kfold = KFold(n_splits=10, random_state=7)
model = ElasticNet()
results = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
print(results.mean())