# Save model using joblib
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.externals.joblib import dump
from sklearn.externals.joblib import load

filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
array = data.values
X = array[:, :8]
y = array[:, 8]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=7)
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model to disk
filename = 'finalized_model.sav'
dump(model, filename)

# Some time later...

# Load the model from disk
loaded_model = load(filename)
result = loaded_model.score(X_test, y_test)
print(result)