import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
combine = [train, test]
print(train.columns.values)
print(test.columns.values)
print(train.head(10))
print(train.tail(10))

'''train.info()
print('_'*30)
test.info()
print(train.describe())
print(train.describe(include = ['O']))
print(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))
g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
grid = sns.FacetGrid(train, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
grid = sns.FacetGrid(train, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
plt.show()'''

train = train.drop(['Ticket', 'Cabin'], axis=1)
test = test.drop(['Ticket', 'Cabin'], axis=1)
combine = [train, test]
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

for dataset in combine:
    dataset['Title'] = dataset['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})
    dataset['Title'] = dataset['Title'].fillna(0)
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
print(train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values(by='Survived', ascending=False))
train = train.drop(["Name", "PassengerId"], axis=1)
print(train.head(11))

guess_age = np.zeros((2, 3))
for i in range(0, 2):
    for j in range(0, 3):
        guess = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
        age_guess = guess.median()
        guess_age[i, j] = age_guess            
for i in range(0, 2):
    for j in range(0, 3):
        train.loc[ (train.Age.isnull()) & (train.Sex == i) & (train.Pclass == j+1),'Age'] = guess_age[i,j]
        test.loc[ (test.Age.isnull()) & (test.Sex == i) & (test.Pclass == j+1),'Age'] = guess_age[i,j]
train['Age'] = train['Age'].astype(int)
test['Age'] = test['Age'].astype(int)

train['AgeBand'] = pd.cut(train['Age'], 5)
test['AgeBand'] = pd.cut(test['Age'], 5)
print(train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True))
combine = [train, test]
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, ['Age']] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), ['Age']] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), ['Age']] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), ['Age']] = 3
    dataset.loc[ dataset['Age'] > 64, ['Age']] = 4
train = train.drop(['AgeBand'], axis=1)
test = test.drop(['AgeBand'], axis=1)
combine = [train, test]
print(train.head())
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
print(train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False))
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
print(train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())
train = train.drop(['Parch', 'SibSp'], axis=1)
test = test.drop(['Parch', 'SibSp'], axis=1)
combine = [train, test]
print(train.head())
print(test.head())
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass
print(train.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10))
freq_port = train.Embarked.dropna().mode()[0]
print(freq_port)
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
print(train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)
train['FareBand'] = pd.cut(train['Fare'], 4)
test['FareBand'] = pd.cut(test['Fare'], 4)
print(train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True))
combine = [test, train]
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
train = train.drop(['FareBand'], axis=1)
test = test.drop(['FareBand'], axis=1)
combine = [train, test]    
print(train.head(10))
print(test.head(10))
test = test.drop(['Name'], axis = 1)
test.info() 

X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test  = test.drop("PassengerId", axis=1).copy()
print(X_train.shape, Y_train.shape, X_test.shape)


linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
print((linear_svc.score(X_train, Y_train)))
my_solution = pd.DataFrame({"PassengerId": test["PassengerId"],"Survived": Y_pred})
print(my_solution.shape)
my_solution.to_csv("linear_svc.csv", index = False)

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
print(decision_tree.score(X_train, Y_train))
my_solution = pd.DataFrame({"PassengerId": test["PassengerId"],"Survived": Y_pred})
print(my_solution.shape)
my_solution.to_csv("decision_tree.csv", index = False)

random_forest = RandomForestClassifier(n_estimators=100, min_samples_split = 5, random_state = 1)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
print(random_forest.score(X_train, Y_train))
my_solution = pd.DataFrame({"PassengerId": test["PassengerId"],"Survived": Y_pred})
print(my_solution.shape)
my_solution.to_csv("random_forest.csv", index = False)

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
print(svc.score(X_train, Y_train))
my_solution = pd.DataFrame({"PassengerId": test["PassengerId"],"Survived": Y_pred})
print(my_solution.shape)
my_solution.to_csv("support_vector_machines.csv", index = False)

