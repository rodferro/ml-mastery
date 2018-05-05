# Feature extraction with RFE
from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'] 
data = read_csv(filename, names=names)
array = data.values
X = array[:,0:8]
y = array[:,8]

# feature extraction
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, y)

print('Num Features: %d' % (fit.n_features_))
print('Selected Features: %s' % (fit.support_))
print('Feature Ranking: %s' % (fit.ranking_))