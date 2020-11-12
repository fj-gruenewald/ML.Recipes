
#Determine what Fruit it is given the Weight
from sklearn import tree

features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = [0, 0, 1, 1]

#Simple Tree Classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

#Make an Example --> 0 for apple, 1 for orange
print(clf.predict([[160, 0]]))