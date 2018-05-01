# Pairwise Pearson's correlation
from pandas import read_csv
from pandas import set_option

filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
set_option('display.width', 80)
set_option('precision', 3)
print(data.corr(method='pearson'))