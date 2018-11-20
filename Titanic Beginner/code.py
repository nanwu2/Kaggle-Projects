
"""
Created on Thu May  3 23:00:11 2018

@author: Nancy Wu
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def sex(x):
    if x == 'male':
        return 1
    else:
        return 2

def embarked(x):
    if x == 'S':
        return 1
    elif x == 'C':
        return 2
    elif x == 'Q':
        return 3
    else:
        return 4
    
def classification_error(y, yhat):
    return np.mean(y!=yhat)
    

titanic = pd.read_csv('C:\\Users\\Nancy Wu\\Kaggle Projects\\Titanic Beginner\\train.csv', header=0)

# not sure if we want to drop cabin... find out the frequency counts, maybe can group by 
# which letter cabin starts with or level the cabin is on?

X = titanic.drop(['Survived', 'Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1)

#### recode the categorical variables as numbers ####
# sex: 1 male, 2 female

X['Sex'] = X['Sex'].apply(sex)

# embarked: 1 S, 2 C, 3 Q, 0 nan
X.Embarked.unique()
X['Embarked'] = X['Embarked'].apply(embarked)

# recode nan values from the age variable
X.loc[np.isnan(X['Age']), 'Age'] = -1

#encode embarked as dummy
lb = preprocessing.LabelBinarizer()
lb.fit(X.iloc[:,6])

dummies = lb.transform(X.iloc[:,6])
X_dummies = X.iloc[:,:6]
X_train = np.concatenate((X_dummies,dummies), axis=1)

# get labels

y = titanic['Survived']

y = y.as_matrix()

# shuffle and split into train and validate

X_train, X_val, y_train, y_val = train_test_split(X,y, random_state=1)

y_train = np.squeeze(y_train)
y_val = np.squeeze(y_val)


# Fit baseline classifier that predicts the most common label
baseline = DummyClassifier(strategy = 'most_frequent',random_state=1)
baseline.fit(X_train,y_train)

yhat = baseline.predict(X_val)
classification_error(y_val, yhat) # 0.4260


# Try other models

### try logisitic regression ###

# with L2 regularization
log_l2 = LogisticRegression()

log_l2.fit(X_train,y_train)
log_l2_yhat = log_l2.predict(X_val)
classification_error(y_val, log_l2_yhat) #0.1973


# with L1 regularization
log_l1 = LogisticRegression(penalty='l1')
log_l1.fit(X_train,y_train)
log_l1_yhat = log_l1.predict(X_val)
classification_error(y_val, log_l1_yhat) #0.2063


# TODO try ROC curve to determine at which percentage to predict 


# try a random forest classifier

rf_model = RandomForestClassifier(n_estimators = 10, max_depth = 10)
rf_model.fit(X_train,y_train)
rf_yhat = rf_model.predict(X_val)
classification_error(y_val,rf_yhat) # 0.2242






