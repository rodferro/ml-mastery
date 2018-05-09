# Evaluate using Leave One Out Cross Validation
from pandas import read_csv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
array = data.values
X = array[:, :8]
y = array[:, 8]
loocv = LeaveOneOut()
model = LogisticRegression()
results = cross_val_score(model, X, y, cv=loocv)
print('Accuracy: %.3f%% (%.3f%%)' % (results.mean() * 100.0, results.std() * 100.0))