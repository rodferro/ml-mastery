# Feature extraction with PCA
from pandas import read_csv
from sklearn.decomposition import PCA

filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
array = data.values
X = array[:, 0:8]
y = array[:, 8]

# feature extraction
pca = PCA(n_components=3)
fit = pca.fit(X)

# summarize components
print('Explained variance: %s' % (fit.explained_variance_ratio_))
print(fit.components_)