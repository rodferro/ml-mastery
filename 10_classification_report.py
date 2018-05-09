# Cross validation classification report
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
array = data.values
X = array[:, :8]
y = array[:, 8]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)
model = LogisticRegression()
model.fit(X_train, y_train)
predicted = model.predict(X_test)
print(classification_report(y_test, predicted))