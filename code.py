
"""
Created on Thu May  3 23:00:11 2018

@author: Nancy Wu
"""

import pandas as pd

def sex(x):
    if x == 'male':
        return 1
    else:
        return 2

def embarked(x):
    if x == 'S':
        return 1

titanic = pd.read_csv('C:\\Users\\Nancy Wu\\Kaggle Projects\\Titanic Beginner\\train.csv', header=0)
X = titanic.drop(['Survived'], axis=1)

#### recode the categorical variables as numbers ####
# sex: 1 male, 2 female

X['Sex'] = X['Sex'].apply(sex)

# embarked: 1 S, 2 C, 3 Q, 0 nan
X.Embarked.unique()

len(X.Ticket.unique())

y = titanic['Survived']
