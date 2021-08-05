import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle
# Importing the dataset
df = pd.read_csv('churn_modelling.csv')
df = pd.get_dummies(df,drop_first=True)
X = df.drop(['Exited'],axis=1)
Y = df['Exited']
X = np.array(X)
x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size=0.2,random_state=42)

sc = StandardScaler()
sc.fit(x_train)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)


lr = LogisticRegression()
lr.fit(x_train,y_train)


#using pickel to dump the codde
file = open('model.pkl','wb')
pickle.dump(lr,file)
file.close()