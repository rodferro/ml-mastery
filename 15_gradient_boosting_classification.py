# Stochastic Gradient Boosting classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'] 
data = read_csv(filename, names=names)
array = data.values
X = array[:, :8]
y = array[:, 8]
seed = 7
kfold = KFold(n_splits=10, random_state=seed)
model = GradientBoostingClassifier(n_estimators=100, random_state=seed)
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())