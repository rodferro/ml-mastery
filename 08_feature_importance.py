# Feature importance with Extra Trees Classifier
from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier

filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'] 
data = read_csv(filename, names=names)
array = data.values
X = array[:, :8]
y = array[:, 8]

# feature extraction
model = ExtraTreesClassifier()
model.fit(X, y)
print(model.feature_importances_)