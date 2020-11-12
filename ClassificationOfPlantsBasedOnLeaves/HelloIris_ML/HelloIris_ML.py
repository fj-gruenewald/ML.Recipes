
# Determine Flowers by the lengh of their leafs
from sklearn.datasets import load_iris

# load the iris dataset
iris = load_iris()

# print out an iris example dataset
# features a plant can have
print(iris.feature_names)

#targetes species
print(iris.target_names)

# example length data
print(iris.data[0])


print("\n Starting the real Work")

# decision tree classifier
from sklearn import tree

# 
X, y = load_iris(return_X_y=True)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)


# visulaize the decision tree
import graphviz 
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("iris") 

dot_data = tree.export_graphviz(clf, out_file=None, 
                     feature_names=iris.feature_names,  
                     class_names=iris.target_names,  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 

