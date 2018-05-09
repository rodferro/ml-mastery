# Normalize data (length of 1)
from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import Normalizer

filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
array = data.values

# separate array into input and output components
X = array[:, :8]
y = array[:, 8]
scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)

# summarize transformed data
set_printoptions(precision=3)
print(normalizedX[:5, :])