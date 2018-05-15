# Extra Trees classification
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier

filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'] 
data = read_csv(filename, names=names)
array = data.values
X = array[:, :8]
y = array[:, 8]
kfold = KFold(n_splits=10, random_state=7)
model = ExtraTreesClassifier(n_estimators=100, max_features=7)
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())