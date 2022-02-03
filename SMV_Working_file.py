import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


cancer = datasets.load_breast_cancer()

x = cancer.data  # hard split [:100] first 100 is testing data rest is training data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

classes = ['malignant', 'benign']

# clf = svm.SVC(kernel="poly", degree=2)  # support vector specification
clf1 = svm.SVC(kernel="linear", C=2)  # support vector specification
clf1.fit(x_train, y_train)

y_pred1 = clf1.predict(x_test)

acc1 = metrics.accuracy_score(y_test, y_pred1)  # order doesnt matter
print(acc1)

clf2 = KNeighborsClassifier(n_neighbors=13)
clf2.fit(x_train, y_train)

y_pred2 = clf2.predict(x_test)

acc2 = metrics.accuracy_score(y_test, y_pred2)  # order doesnt matter
print(acc2)
#  typically knn doesnt work well in high dimensions
