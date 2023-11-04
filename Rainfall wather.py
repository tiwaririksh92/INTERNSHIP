#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('rain wather.csv')
df


# In[3]:


df.head(15)


# In[4]:


df.tail(15)


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
sns.countplot(x ='Rainfall',data = df)
plt.xlabel('Avrage in Rainfall')
plt.ylabel('Count of rows in the dataset')
plt.show()


# In[12]:


sns.pairplot(data=df,palette = "Dark2")


# In[13]:


index=0
labels = df['Rainfall']
features = df.drop ('Rainfall',axis=1)
for col in features.items():
    plt.figure(figsize=(10,5))
    sns.barplot(x=labels, y= col[index], data=df, color="green")
    plt.show()


# In[16]:


plt.figure(figsize=(26,14))
sns.heatmap(df.corr(),annot = True, fmt = '0.2f',linewidth = 0.2, linecolor ='black', cmap = 'Spectral')
plt.xlabel('Figure',fontsize =14)
plt.ylabel('Features_Rainfall',fontsize =14)
plt.title('Descriptive Graph',fontsize =20)
plt.show()


# In[17]:


df = df.drop('MaxTemp', axis =1)
df


# In[18]:


df.shape


# In[19]:


from scipy.stats import zscore


# In[22]:


x= df.drop ('MinTemp', axis=1)
y= df['MinTemp']


# In[23]:


y.value_counts()


# In[24]:


print("Feature Dimension=",x.shape)
print("Label Dimension=",y.shape)


# In[25]:


from sklearn.preprocessing import StandardScaler


# In[27]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


# In[30]:


from sklearn.model_selection import train_test_split


# In[32]:


from sklearn.svm import SVR


# In[33]:


svr=SVR()

svr.fit(x_train,y_train)
pred_svr=svr.predict(x_test)
pred_train=svr.predict(x_train)
print('R2_score:',r2_score(y_test,pred_svr))
print('R2_score on training Data:',r2_score(y_train,pred_train)*100)
print('Mean Absolute Error:',mean_absolute_error(y_test,pred_svr))
print('Mean Squared Error:',mean_squared_error(y_test,pred_svr))
print("Root Mean Squared Error:",np.sqrt(mean_squared_error(y_test,pred_svr)))


# In[ ]:




