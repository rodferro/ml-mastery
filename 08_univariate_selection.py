# Feature extraction with univariate statistical tests (chi-squared for classification)
from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
array = data.values
X = array[:, 0:8]
y = array[:, 8]

# feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, y)

# summarize scores
set_printoptions(precision=3)
print(fit.scores_)

# summarize selected features
features = fit.transform(X)
print(features[0:5, :])