import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle



df=pd.read_csv('Churn_Modelling.csv')


X=df.iloc[:,3:13]
Y=df.iloc[:,13] 


X=X.drop(['Geography','Gender'],axis=1)



x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=0)



sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)



lr = LogisticRegression()
lr.fit(x_train,y_train)

#using pickel to dump the codde
file = open('model.pkl','wb')
pickle.dump(lr,file)
file.close()

#using pickel to dump the codde
file1 = open('sc.pkl','wb')
pickle.dump(sc,file1)
file1.close()