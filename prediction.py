import pandas as pd
import numpy as np
from sklearn import tree

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
#print(train.describe())
pd.options.mode.chained_assignment = None
train["Age"] = train["Age"].fillna(train["Age"].median())
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
train["Embarked"] = train["Embarked"].fillna("S")
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2
#print(train["Sex"])
#print(train["Embarked"])
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1
test["Age"] = test["Age"].fillna(test["Age"].median())
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2

train["Child"] = 0
train["Child"][train["Age"] < 18] = 1
target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)
#print(my_tree_one.feature_importances_)
#print(my_tree_one.score(features_one, target))
test.Fare[152] = test["Fare"].median()
test_features = test[["Pclass", "Sex", "Age","Fare"]].values
my_prediction = my_tree_one.predict(test_features)
print(my_prediction)
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
print(my_solution)
print(my_solution.shape)
my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])


features_two = train[["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]].values
my_tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)
my_tree_two = my_tree_two.fit(features_two, target)
print(my_tree_two.score(features_two, target))
test_features_two = test[["Pclass", "Sex", "Age","Fare", "SibSp", "Parch", "Embarked"]].values
my_prediction_two = my_tree_two.predict(test_features_two)
print(my_prediction)
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
print(my_solution)
print(my_solution.shape)
my_solution.to_csv("my_solution_two.csv", index_label = ["PassengerId"])

