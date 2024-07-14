# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

wine_dataset=pd.read_csv('winequality-red.csv')

# wine_dataset.head()

sns.catplot(x="quality",data=wine_dataset, kind="count")
# plt.show() #this function is mandataory in pycharm to see the plot

plt.figure(figsize=(6,6))
sns.barplot(x="quality",y="volatile acidity",data=wine_dataset)
# plt.show()

correlation=wine_dataset.corr()
plt.figure(figsize=(12,12))
sns.heatmap(correlation,cbar=True,square=True,fmt='.1f',annot_kws={"size":10},annot=True,cmap="Blues")
# plt.show()

"""Data Preprocessing"""

# Separate the data and label
X=wine_dataset.drop("quality",axis=1) # axis=1 for col and axis=0 for row
#X

Y=wine_dataset["quality"].apply(lambda y_val: 1 if y_val>=7 else (2 if 4<=y_val<=6 else 0))

# print(Y)

# Train and Test Split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2,random_state=3)

# print(Y.shape,Y_train.shape,Y_test.shape)

# Model training
model=RandomForestClassifier()
model.fit(X_train.values,Y_train.values)

X_test_prediction=model.predict(X_test.values)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)

# print(test_data_accuracy)

# Building the Predictive model
demo_test=(7.5,0.52,0.16,1.9,0.085,12.0,35.0,0.9968,3.38,0.62,9.5)
data_d=[]
q=input("Do you want to run your data set or demo set? write \"demo\" to run demo or \"test\" to run your own test case: ")
if(q=="demo"):
  ans=np.asarray(demo_test)
  ans_reshape=ans.reshape(1,-1) # one row and col accordingly to run for one instance
  prediction=model.predict(ans_reshape)
  if(prediction==1):
    print(f"The data set is: {ans}")
    print("Excllent Quality Wine")

else:
  lst=[]
  fa=input("Enter the value of Fixed Acidity: ")
  va=input("Enter the value of Volatile Acidity: ")
  ca=input("Enter the value of Citric Acid: ")
  rs=input("Enter the value of Residual Sugar: ")
  cl=input("Enter the value of Chlorides: ")
  so=input("Enter the value of Free Sulphur Dioxide: ")
  tso=input("Enter the value of Total Sulphur Dioxide: ")
  d=input("Enter the value of Density: ")
  ph=input("Enter the value of PH value: ")
  sh=input("Enter the value of Sulphates: ")
  al=input("Enter the value of Alcohol: ")


  lst.append(fa)
  lst.append(va)
  lst.append(ca)
  lst.append(rs)
  lst.append(cl)
  lst.append(so)
  lst.append(tso)
  lst.append(d)
  lst.append(ph)
  lst.append(sh)
  lst.append(al)

  ans=np.asarray(lst)
  ans_reshape=ans.reshape(1,-1)
  prediction=model.predict(ans_reshape)
  if(prediction==1):
    print("Excellent Quality Wine")
  elif (prediction==2):
    print("Mediocre Quality Wine")
  else:
    print("Poor Quality Wine")

