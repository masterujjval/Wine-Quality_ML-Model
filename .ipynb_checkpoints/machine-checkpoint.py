# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Loading dataset

wine_dataset=pd.read_csv('winequality-red.csv')
# to print number of rows and cols
#print(np.shape(wine_dataset))

# To get first five rows of the dataset
#print(wine_dataset.head())

# Checking for missing values
#print(wine_dataset.isnull().sum())

# Statistical measure of data
#print(wine_dataset.describe())


# Number of values for each quality

sns.catplot(x="quality",data=wine_dataset, kind="count")
plt.show() #this function is mandataory in pycharm to see the plot

# volatile acifity vs quality barplot
plt.figure(figsize=(6,6))
sns.barplot(x="quality",y="volatile acidity",data=wine_dataset)
plt.show()

correlation=wine_dataset.corr()
sns.heatmap(correlation,cbar=True,square=True,fmt='.1f',annot_kws={"size":10},annot=True,cmap="Blues")
plt.show()