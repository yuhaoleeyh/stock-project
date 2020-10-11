from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

# import sklearn
# print(sklearn.__version__)

dataset = datasets.load_iris()

model = LogisticRegression()

model.fit(dataset.data, dataset.target)

print(model)

expected = dataset.target
predicted = model.predict(dataset.data)

print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

