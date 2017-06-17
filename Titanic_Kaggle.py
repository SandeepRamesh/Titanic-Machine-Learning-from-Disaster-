#Author:Sandeep Ramesh

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Importing the dataset
titanic_train = pd.read_csv('train.csv')
titanic_test= pd.read_csv('test.csv')

#To check for missing values
titanic_train.info()
titanic_test.info()

#plotting the scatter matrix first
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
scatter_matrix(titanic_train, figsize=(25,25))
plt.show()

titanic_train=titanic_train.drop(['PassengerId','Name','Ticket'],1)
titanic_test=titanic_test.drop(['Name','Ticket'],1)

#drop cabin because of too many NaN values
titanic_train= titanic_train.drop(['Cabin'],1)
titanic_test= titanic_test.drop(['Cabin'],1)

#To convert Sex from categorical to dummy variables and determine which gender to drop for dummy variable trap
sns.factorplot('Sex','Survived',data=titanic_train)
titanic_train=titanic_train.join(pd.get_dummies(titanic_train.Sex,prefix='Sex'))
titanic_train=titanic_train.drop(['Sex','Sex_male'],1)
titanic_test=titanic_test.join(pd.get_dummies(titanic_test.Sex,prefix='Sex'))
titanic_test=titanic_test.drop(['Sex','Sex_male'],1)


#converting Pclass categorical to dummy variables and determine which class to drop for dummy variable trap
#titanic_train['Pclass']=pd.get_dummies(titanic_train.Pclass)
sns.factorplot('Pclass','Survived',data=titanic_train)
plt.hist(titanic_train.Pclass)    #to determine class 2 or class 3 to drop
titanic_train=titanic_train.join(pd.get_dummies(titanic_train.Pclass,prefix='Pclass'))
titanic_train=titanic_train.drop(['Pclass','Pclass_3'],1)
titanic_test=titanic_test.join(pd.get_dummies(titanic_test.Pclass,prefix='Pclass'))
titanic_test=titanic_test.drop(['Pclass','Pclass_3'],1)


#filling the Embarked with most frequent column after getting a count
#sub=pd.DataFrame(titanic_train)
#sub.to_csv('newtrain.csv')
sns.factorplot('Embarked',kind='count',data=titanic_train)
titanic_train['Embarked'] = titanic_train['Embarked'].fillna("S")

#converting Embarked to dummy variable and dropping the extra column 'S' for dummy trap since S has low chance
sns.factorplot('Embarked','Survived',data=titanic_train)
titanic_train=titanic_train.join(pd.get_dummies(titanic_train.Embarked,prefix='Embarked'))
titanic_train=titanic_train.drop(['Embarked','Embarked_S'],1)
titanic_test=titanic_test.join(pd.get_dummies(titanic_test.Embarked,prefix='Embarked'))
titanic_test=titanic_test.drop(['Embarked','Embarked_S'],1)

#missing values in Age 
imputer = Imputer(missing_values='NaN',strategy='median',axis=0)
imputer = imputer.fit(titanic_train[['Age']])
titanic_train[['Age']]=imputer.transform(titanic_train[['Age']])
imputer = imputer.fit(titanic_test[['Age']])
titanic_test[['Age']]=imputer.transform(titanic_test[['Age']])

#missing values in fare
imputer = Imputer(missing_values='NaN',strategy='median',axis=0)
imputer = imputer.fit(titanic_train[['Fare']])
titanic_train[['Fare']]=imputer.transform(titanic_train[['Fare']])
imputer = imputer.fit(titanic_test[['Fare']])
titanic_test[['Fare']]=imputer.transform(titanic_test[['Fare']])

x_train=titanic_train.drop(['Survived'],1)
y_train=titanic_train['Survived']
x_test=titanic_test.drop(['PassengerId'],1)




lr=LogisticRegression(random_state=0)
lr=lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
score=lr.score(x_train,y_train)
print("Logistic Regression Classifier score:{}".format(score))

rfc=RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=0)
rfc=rfc.fit(x_train,y_train)
y_pred=rfc.predict(x_test)
score=rfc.score(x_train,y_train)
print("Random forest Classifier score:{}".format(score))


knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn=knn.fit(x_train, y_train)
y_pred=knn.predict(x_test)
score=knn.score(x_train,y_train)
print("K Nearest Neighbors Classifier score:{}".format(score))

nb = GaussianNB()
nb=nb.fit(x_train,y_train)
y_pred=nb.predict(x_test)
score=nb.score(x_train,y_train)
print("Naive Bayes Classifier score:{}".format(score))

svc = SVC(kernel = 'rbf', random_state = 0)
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
score=svc.score(x_train,y_train)
print("Kernel SVM Classifier score:{}".format(score))


xg = XGBClassifier(max_depth=3,n_estimators=100)
xg.fit(x_train, y_train)
y_pred = xg.predict(x_test)
score=xg.score(x_train,y_train)
print("XG Boost Classifier score:{}".format(score))

#Random Forest is the best classifier according to the score

results=pd.DataFrame({"PassengerId":titanic_test['PassengerId'],"Survived":y_pred})
results.to_csv('Final_Output.csv',index=False)

