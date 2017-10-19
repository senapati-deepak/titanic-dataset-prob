
# coding: utf-8

# # **_Titanic Dataset Solution_**
# ***

# <br><br><br>To solve the titanic dataset we need to follow the following steps:-
#     * Import the required Libraries.
#     * Load the the training and test datasets.
#     * Visualise the data with the help of graphs.
#     * Carry out data analysis and data cleaning.
#     * Make predictions.

# ## _Importing the required libraries:-_<br>

# In[565]:


import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier



# <br>
# ## _Loading the training and test datasets. And combine them into a single dataset- combine._ <br>

# In[566]:


train = pd.read_csv("datasets/train.csv")
test = pd.read_csv("datasets/test.csv")
combine = [train, test]
train.columns.values


# In[567]:


test.columns.values


# <br>
# ## _Analysing the Data_<br>

# Printing the first 10 fields of train.....

# In[568]:


train.head(10)


# Printing the last 10 fields of train....

# In[569]:


train.tail(10)


# In[570]:


train.info()
print("_"*100+"\n")
test.info()


# In[571]:


train.describe()


# In[572]:


train.describe(include = ['O'])


# In[573]:


train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[574]:


train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[575]:


train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[576]:


train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# <br>
# ## _Data Visualisation_<br>

# In[577]:


g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)


# In[578]:


grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


# In[579]:


grid = sns.FacetGrid(train, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()


# In[580]:


grid = sns.FacetGrid(train, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()


# <br>
# ## _Data Cleaning_<br>

# In[581]:


train = train.drop(['Ticket', 'Cabin'], axis=1)
test = test.drop(['Ticket', 'Cabin'], axis=1)
combine = [train, test]


# In[582]:


for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')


# In[583]:


for dataset in combine:
    dataset['Title'] = dataset['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})
    dataset['Title'] = dataset['Title'].fillna(0)
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


# In[584]:


train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[585]:


train = train.drop(["Name", "PassengerId"], axis=1)
train.head(11)


# In[586]:


guess_age = np.zeros((2, 3))
for i in range(0, 2):
    for j in range(0, 3):
        guess = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
        age_guess = guess.median()
        guess_age[i, j] = age_guess  
guess_age


# In[587]:


for i in range(0, 2):
    for j in range(0, 3):
        train.loc[ (train.Age.isnull()) & (train.Sex == i) & (train.Pclass == j+1),'Age'] = guess_age[i,j]
        test.loc[ (test.Age.isnull()) & (test.Sex == i) & (test.Pclass == j+1),'Age'] = guess_age[i,j]
train['Age'] = train['Age'].astype(int)
test['Age'] = test['Age'].astype(int)


# In[588]:


train['Age'].head()


# In[589]:


test['Age'].head()


# In[590]:


train['AgeBand'] = pd.cut(train['Age'], 5)
test['AgeBand'] = pd.cut(test['Age'], 5)
train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


# In[591]:


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
train.head()


# In[592]:


for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[593]:


for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()


# In[594]:


train = train.drop(['Parch', 'SibSp'], axis=1)
test = test.drop(['Parch', 'SibSp'], axis=1)
combine = [train, test]


# In[595]:


train.head()


# In[596]:


test.head()


# In[597]:


combine = [train, test]
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass
train.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)


# In[598]:


freq_port = train.Embarked.dropna().mode()[0]
freq_port


# In[599]:


train['Embarked'] = train['Embarked'].fillna(freq_port)
test['Embarked'] = test['Embarked'].fillna(freq_port)
train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[600]:


test['Embarked'].head(11)


# In[601]:


test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)
train['FareBand'] = pd.cut(train['Fare'], 4)
test['FareBand'] = pd.cut(test['Fare'], 4)
train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)


# In[602]:


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


# In[603]:


for dataset in combine:
    dataset.loc[ dataset['Embarked'] == 'S', 'Embarked'] = 0
    dataset.loc[ dataset['Embarked'] == 'C', 'Embarked'] = 1
    dataset.loc[ dataset['Embarked'] == 'Q', 'Embarked'] = 2
    dataset['Embarked'] = dataset['Embarked'].astype(int)
combine = [train, test]


# In[604]:


train.head(10)


# In[605]:


test.head(10)


# In[606]:


test = test.drop(['Name'], axis = 1)
test = test.drop(['Age*Class'], axis = 1)
train = train.drop(['Age*Class'], axis = 1)
test = test.drop(['Fare'], axis = 1)
train = train.drop(['Fare'], axis = 1)
test = test.drop(['IsAlone'], axis = 1)
train = train.drop(['IsAlone'], axis = 1)
test.info() 


# In[607]:


train.info()


# <br>
# ## _Making Predictions_<br>

# In[608]:


X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
X_test  = test.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# <br>
# I have made the predictions in 5 diff methods:-
#     1. Linear SVC
#     2. Decision Tree
#     3. Random Forest
#     4. Support Vector Machine
#     5. logistic Regression.

# <br>**_1. Linear SVC_**

# In[609]:


linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)


# In[610]:


my_solution = pd.DataFrame({"PassengerId": test["PassengerId"],"Survived": Y_pred})
my_solution.to_csv("solutions/linear_svc.csv", index = False)
my_solution.shape


# In[611]:


linear_svc.score(X_train, Y_train)


# <br>
# **_2. Decision Tree_**

# In[612]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)


# In[613]:


my_solution = pd.DataFrame({"PassengerId": test["PassengerId"],"Survived": Y_pred})
my_solution.to_csv("solutions/decision_tree.csv", index = False)
my_solution.shape


# In[614]:


decision_tree.score(X_train, Y_train)


# <br>
# **_3. Random Forest_**

# In[615]:


random_forest = RandomForestClassifier(n_estimators=100, min_samples_split = 5, random_state = 1)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)


# In[616]:


my_solution = pd.DataFrame({"PassengerId": test["PassengerId"],"Survived": Y_pred})
my_solution.to_csv("solutions/random_forest.csv", index = False)
my_solution.shape


# In[617]:


random_forest.score(X_train, Y_train)


# <br>
# **_4. Support Vector Machine_**

# In[618]:


svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)


# In[619]:


my_solution = pd.DataFrame({"PassengerId": test["PassengerId"],"Survived": Y_pred})
my_solution.to_csv("solutions/support_vector_machines.csv", index = False)
my_solution.shape


# In[620]:


svc.score(X_train, Y_train)


# <br>
# **_5. Logistic Regression_**

# In[621]:


logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)


# In[622]:


my_solution = pd.DataFrame({"PassengerId": test["PassengerId"],"Survived": Y_pred})
my_solution.to_csv("solutions/LogisticRegression.csv", index = False)
my_solution.shape


# In[623]:


logreg.score(X_train, Y_train)

