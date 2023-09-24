#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd       #world happiness index
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('Happyness.csv')
df


# In[3]:


df.head(10)


# In[4]:


df.tail(10)


# In[5]:


df.shape


# In[6]:


print(f"The rows and columns in the dataset:{df.shape}")
print(f"\n The column headers in the dataset:{df.columns}")


# In[7]:


df.isnull().sum()


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


df.skew()


# In[11]:


plt.figure(figsize=(10,7))
sns.countplot(x ='Happiness Score',data = df)
plt.xlabel('STANDARD OF HAPPINESS SCORE')
plt.ylabel('Count of rows in the dataset')
plt.show()


# In[12]:


sns.pairplot(data=df,palette = "Dark2")


# In[13]:


index=0
labels = df['Happiness Score']
features = df.drop ('Happiness Score',axis=1)
for col in features.items():
    plt.figure(figsize=(10,5))
    sns.barplot(x=labels, y= col[index], data=df, color="green")
    plt.show()


# In[14]:


plt.figure(figsize=(26,14))
sns.heatmap(df.corr(),annot = True, fmt = '0.2f',linewidth = 0.2, linecolor ='black', cmap = 'Spectral')
plt.xlabel('Standard_Name',fontsize =14)
plt.ylabel('Features_Name',fontsize =14)
plt.title('Descriptive Graph',fontsize =20)
plt.show()


# In[15]:


df.head(7)


# In[16]:


df = df.drop('Dystopia Residual', axis =1)
df


# In[17]:


df.shape


# In[18]:


from scipy.stats import zscore


# In[19]:


x= df.drop ('Happiness Score', axis=1)
y= df['Happiness Score']


# In[20]:


y.value_counts()


# In[21]:


print("Feature Dimension=",x.shape)
print("Label Dimension=",y.shape)


# In[22]:


from sklearn.preprocessing import StandardScaler


# In[23]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


# In[24]:


print("Min.Happiness Score:", df["Happiness Score"].min())
print("Max.Happiness Score:", df["Happiness Score"].max())


# In[25]:


df.dtypes


# In[26]:


df.nunique().to_frame("No.of unique values")


# In[27]:


print("Total Duplicate Rows are",df.duplicated().sum())


# In[28]:


df['Region'].value_counts().sort_values()


# In[29]:


df.corr()


# In[30]:


plt.figure(figsize=(10,7))
sns.distplot(df['Happiness Score'])
plt.title('Plot for Happiness score')
plt.xlabel('Happiness score')
plt.ylabel('Standard_Scale')
plt.show()


# In[31]:


plt.figure(figsize=(10,7))
sns.distplot(df['Happiness Rank'])
plt.title('Plot for Happiness rank')
plt.xlabel('Happiness Rank')
plt.ylabel('Standard_Scale')
plt.show()


# In[32]:


maxRS=0


# In[33]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=maxRS)


# In[34]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.linear_model import Lasso,Ridge


# In[35]:


df.head(6)


# In[36]:


data={'Country': {1},  'Region': {9}}
df =df.copy()
df.replace(data, inplace=True)


# In[37]:


df.describe()


# In[38]:


from scipy.stats import zscore


# In[39]:


from sklearn.preprocessing import StandardScaler


# In[40]:


from sklearn.model_selection import GridSearchCV


# In[41]:


param = {'alpha':[1.0,.05,.4,2], 'fit_intercept':[True,False], 'solver':['auto', 'sud', ' cholesky', 'lsqr', 'sag', 'saga', 'lbfgs'], 'positive': [False,True],'random_state':[1,4,10,20]}


# In[85]:


import joblib


# In[86]:


from sklearn import metrics 
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# In[87]:


import pickle


# In[88]:


filename = 'world happiness index.pkl'
pickle.dump(model,open(filename, 'wb'))


# In[ ]:





# # world tiatenic

# In[ ]:





# In[42]:


df1 = pd.read_csv('taitenic.csv')
df1


# In[43]:


df1.head(17)


# In[44]:


df1.tail(17)


# In[45]:


df1.shape


# In[46]:


print(f"The rows and columns in the dataset:{df.shape}")
print(f"\n The column headers in the dataset:{df.columns}")


# In[47]:


df1.isnull().sum()


# In[48]:


df1.info()


# In[49]:


df1.describe()


# In[50]:


df1.skew()


# In[51]:


plt.figure(figsize=(10,7))
sns.countplot(x ='PassengerId',data = df1)
plt.xlabel('STANDARD OF PassengerId')
plt.ylabel('Count of rows in the dataset')
plt.show()


# In[52]:


sns.pairplot(data=df1,palette = "Dark2")


# In[53]:


index=0
labels = df1['Survived']
features = df1.drop ('Survived',axis=1)
for col in features.items():
    plt.figure(figsize=(10,5))
    sns.barplot(x=labels, y= col[index], data=df1, color="red")
    plt.show()


# In[54]:


plt.figure(figsize=(26,14))
sns.heatmap(df1.corr(),annot = True, fmt = '0.2f',linewidth = 0.2, linecolor ='black', cmap = 'Spectral')
plt.xlabel('Passenger_Id',fontsize =14)
plt.ylabel('Features_Name',fontsize =14)
plt.title('Descriptive Graph',fontsize =20)
plt.show()


# In[55]:


df1.head()


# In[56]:


df1 = df1.drop('PassengerId', axis =1)
df1


# df1.shape

# In[57]:


from scipy.stats import zscore


# In[58]:


print("Feature Dimension=",x.shape)
print("Label Dimension=",y.shape)


# In[59]:


from sklearn.preprocessing import StandardScaler


# In[60]:


df1.nunique().to_frame("No.of unique values")


# In[61]:


for i in df1.columns:
    print(df1[i].value_counts())
    print("\n")


# In[62]:


print("Total Duplicate Rows are",df1.duplicated().sum())


# In[63]:


df1.head(19)


# In[64]:


df1['Cabin'].value_counts().sort_values()


# In[65]:


df1['Embarked'].value_counts().sort_values()


# In[66]:


df1['Sex'].value_counts().sort_values()


# In[67]:


df1['Name'].value_counts().sort_values()


# In[68]:


df1.head(10)


# In[69]:


from sklearn.preprocessing import StandardScaler


# In[70]:


import joblib


# In[71]:


from sklearn import metrics 
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# In[72]:


import pickle


# In[74]:


model = Ridge (alpha=0.05,fit_intercept=True,positive=False,random_state=1,solver='auto')


# In[75]:


filename = 'world tiatenic.pkl'
pickle.dump(model,open(filename, 'wb'))


# In[ ]:




