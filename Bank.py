#!/usr/bin/env python
# coding: utf-8

# # importing libraries

# In[1]:


import os
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
from  matplotlib import pyplot 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
# from sklearn.datasets import make_classification


# # loading dataset

# In[2]:


df = pd.read_csv("bank-loan.csv")


# There are 700 data in training dataset, 150 in test dataset

# # Exploratory data analysis

# In[3]:


df.columns


# # High Level Analysis

# In[4]:


df.head()


# 1. Age: Age of each customer                      Numerical
# 2. Ed: Education categories                Categorical
# 3. Employ: Employment status -                Numerical
#     Corresponds to job
#     status and being
#     converted to numeric
#     format
# 4. Address: Geographic area -                     Numerical
#     Converted to numeric
#     values
# 5  Income: Gross Income of each                   Numerical
#     customer
# 6. debtinc: Individual’s debt                     Numerical
#     payment to his or her
#     gross income
# 7. creddebt: debt-to-credit ratio is a            Numerical
#     measurement of how
#     much you owe your
#     creditors as a
#     percentage of your
#     available credit (credit
#     limits)
# 8. othdebt Any other debts                       Numerical
# 
# 
# 

# In[5]:


df.info()


# In[6]:


df.describe()


# # Null Check

# In[7]:


# null values present=====> 150 values are missing
# [df["default"].isna()]
df.isnull().sum()


# # Missing value analysis

# In[8]:


#Create dataframe with missing percentage
missing_val = pd.DataFrame(df.isnull().sum())

#Reset index
missing_val = missing_val.reset_index()

#Rename variable
missing_val = missing_val.rename(columns = {'index': 'Variables', 0: 'Missing_percentage'})

#Calculate percentage
missing_val['Missing_percentage'] = (missing_val['Missing_percentage']/len(df))*100

#descending order
missing_val = missing_val.sort_values('Missing_percentage', ascending = False).reset_index(drop = True)

#save output results 
missing_val.to_csv("Mising_perc_python.csv", index = False)


# In[9]:


#Impute with median -unsure because cannot fill in missing values for them
# df['default'] = df['default'].fillna(df['default'].median())


# # Null correction

# In[10]:


df=df.dropna()
df.info()


# # Outlier Analysis

# In[40]:


#Plot boxplot to visualize Outliers
get_ipython().run_line_magic('matplotlib', 'inline')
l = df.columns.values
number_of_columns=9
number_of_rows = len(l)-1/number_of_columns
plt.figure(figsize=(2*number_of_columns,5*number_of_rows))
for i in range(0,len(l)):
    plt.subplot(number_of_rows + 1,number_of_columns,i+1)
    sns.set_style('whitegrid')
    sns.boxplot(df[l[i]],color='red',orient='h')
    plt.tight_layout()


# In[12]:


#To check distribution-Skewness
plt.figure(figsize=(2*number_of_columns,5*number_of_rows))
for i in range(0,len(l)):
    plt.subplot(number_of_rows + 1,number_of_columns,i+1)
    sns.distplot(df[l[i]],kde=True) 
#All independent variables are right skewed/positively skewed.


# In[13]:


#save numeric names
cnames =  ['age', 'ed', 'employ', 'address', 'income', 'debtinc', 'creddebt',
       'othdebt', 'default']


# # Feature Selection

# In[16]:


#default correlation matrix
k = 8 #number of variables for heatmap
cols = df.corr().nlargest(k, 'default')['default'].index
cm = df[cols].corr()
plt.figure(figsize=(10,6))
sns.heatmap(cm, annot=True, cmap = 'viridis')


# In[17]:


df.drop("address", axis="columns", inplace=True)


# .# dropping correlated variable

# # Separate X and Y

# In[18]:


X = df.drop(['default'], axis=1)
y = df['default']


# # Modeling 

# In[19]:


#dividing data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=200)


# In[20]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg =LogisticRegression(penalty='l2', C=0.1)
logreg.fit(X_train,y_train)
#predict new test cases
Log_Predictions = logreg.predict(X_test)
print(logreg.score(X_test,y_test))


# In[21]:


#build confusion matrix
# from sklearn.metrics import confusion_matrix 
# CM = confusion_matrix(y_test, y_pred)
CM = pd.crosstab(y_test, Log_Predictions)

#let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

#check accuracy of model
#accuracy_score(y_test, y_pred)*100
print(((TP+TN)*100)/(TP+TN+FP+FN))

#False Negative rate 
print((FN*100)/(FN+TP))

print("Defaulted", sum(Log_Predictions!=0))
print("Non-defaulted ", sum(Log_Predictions==0))
#Results
CM


# In[22]:


#ROC curve and AUC
prob=logreg.predict_proba(X_test)
# print(prob)
prob=prob[:,0]
auc=roc_auc_score(y_test,prob)
print(auc)
fpr,tpr,thresh=roc_curve(y_test,prob)
plt.plot([0,1],[1,0],linestyle='--')
plt.plot(fpr,tpr,marker='.',color='red',label="Logistic regression")


# In[23]:


#Decision Tree
from sklearn import tree
C50_model = tree.DecisionTreeClassifier(criterion='entropy').fit(X_train, y_train)

#predict new test cases
C50_Predictions = C50_model.predict(X_test)


# In[24]:


#build confusion matrix
# from sklearn.metrics import confusion_matrix 
# CM = confusion_matrix(y_test, y_pred)
CM = pd.crosstab(y_test, C50_Predictions)

#let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

#check accuracy of model
#accuracy_score(y_test, y_pred)*100
print(((TP+TN)*100)/(TP+TN+FP+FN))

#False Negative rate 
print((FN*100)/(FN+TP))

print("Defaulted", sum(C50_Predictions!=0))
print("Non-defaulted ", sum(C50_Predictions==0))
#Results
CM


# In[25]:


#ROC curve and AUC
prob=C50_model.predict_proba(X_test)
# print(prob)
prob=prob[:,0]
auc=roc_auc_score(y_test,prob)
print(auc)
fpr,tpr,thresh=roc_curve(y_test,prob)
plt.plot([0,1],[1,0],linestyle='--')
plt.plot(fpr,tpr,marker='.',color='red')


# ### it can be colcluded that Logistic Regression gave better results than Decision Tree

# In[26]:


#Random Forest
from sklearn.ensemble import RandomForestClassifier

RF_model = RandomForestClassifier(n_estimators = 200).fit(X_train, y_train)


# In[27]:


RF_Predictions = RF_model.predict(X_test)


# In[28]:


#build confusion matrix
# from sklearn.metrics import confusion_matrix 
# CM = confusion_matrix(y_test, y_pred)
CM = pd.crosstab(y_test, RF_Predictions)

#let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

#check accuracy of model
#accuracy_score(y_test, y_pred)*100
print(((TP+TN)*100)/(TP+TN+FP+FN))

#False Negative rate 
print((FN*100)/(FN+TP))

print("Defaulted", sum(RF_Predictions!=0))
print("Non-defaulted ", sum(RF_Predictions==0))
CM


# In[29]:


#ROC curve and AUC
prob=RF_model.predict_proba(X_test)
# print(prob)
prob=prob[:,0]
auc=roc_auc_score(y_test,prob)
print(auc)
fpr,tpr,thresh=roc_curve(y_test,prob)
plt.plot([0,1],[1,0],linestyle='--')
plt.plot(fpr,tpr,marker='.',color='red')


# In[30]:


#Naive Bayes
from sklearn.naive_bayes import GaussianNB

#Naive Bayes implementation
NB_model = GaussianNB().fit(X_train, y_train)


# In[31]:


#predict test cases
NB_Predictions = NB_model.predict(X_test)


# In[32]:


#Build confusion matrix
CM = pd.crosstab(y_test, NB_Predictions)

#let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

#check accuracy of model
# accuracy_score(y_test, y_pred)*100
print(((TP+TN)*100)/(TP+TN+FP+FN))

#False Negative rate 
print((FN*100)/(FN+TP))

print("Defaulted", sum(NB_Predictions!=0))
print("Non-defaulted ", sum(NB_Predictions==0))
CM


# In[33]:


#ROC curve and AUC
prob=NB_model.predict_proba(X_test)
# print(prob)
prob=prob[:,0]
auc=roc_auc_score(y_test,prob)
print(auc)
fpr,tpr,thresh=roc_curve(y_test,prob)
plt.plot([0,1],[1,0],linestyle='--')
plt.plot(fpr,tpr,marker='.',color='red')


# ### It seems like RF gives similar results as Log Reg

# In[44]:


#XGB classifier
from xgboost import XGBClassifier
XG_model = XGBClassifier(n_estimators = 200).fit(X_train, y_train)


# In[45]:


XG_Predictions = XG_model.predict(X_test)


# In[46]:


#build confusion matrix
# from sklearn.metrics import confusion_matrix 
# CM = confusion_matrix(y_test, y_pred)
CM = pd.crosstab(y_test, XG_Predictions)

#let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]

#check accuracy of model
#accuracy_score(y_test, y_pred)*100
print(((TP+TN)*100)/(TP+TN+FP+FN))

#False Negative rate 
print((FN*100)/(FN+TP))

print("Defaulted", sum(XG_Predictions!=0))
print("Non-defaulted ", sum(XG_Predictions==0))
CM


# In[ ]:


#ROC curve and AUC
prob=NB_model.predict_proba(X_test)
# print(prob)
prob=prob[:,0]
auc=roc_auc_score(y_test,prob)
print(auc)
fpr,tpr,thresh=roc_curve(y_test,prob)
plt.plot([0,1],[1,0],linestyle='--')
plt.plot(fpr,tpr,marker='.',color='red')


# # ROC and AUC curves

# # AUC =1 ROC touches (0,1)
# #     =0.5 ROC diagonal line
# #     =0 ROC touches (1,0) 
