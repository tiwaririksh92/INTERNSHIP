#!/usr/bin/env python
# coding: utf-8

# In[91]:


import pandas as pd       #glass identification
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('Glass.csv')
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


names = [ 'Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'glass_type']
df.columns = names
df.head()


# In[12]:


df = df.drop('Id', 1)
df.head(5)


# In[13]:


from scipy import stats
z = abs (stats.zscore(df))
df = df[(z<3).all(axis=1)]


# In[14]:


features = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
label = ['glass_type']
x = df[features]
y = df[label]


# In[15]:


x.shape


# In[16]:


type(x)


# In[17]:


x2 = x.values
from matplotlib import pyplot as plt
import seaborn as sns
for i in range(1,9):
    sns.distplot(x2[i])
    plt.xlabel(features[i])
    plt.show()


# In[18]:


x2 = pd.DataFrame(x)
plt.figure(figsize=(8,8))
sns.pairplot(data=x2)
plt.show()


# In[19]:


plt.figure(figsize=(26,14))
sns.heatmap(df.corr(),annot = True, fmt = '0.2f',linewidth = 0.2, linecolor ='black', cmap = 'Spectral')
plt.xlabel('Standard_Name',fontsize =14)
plt.ylabel('Features_Name',fontsize =14)
plt.title('Descriptive Graph',fontsize =20)
plt.show()


# In[20]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[21]:


x.head(5)


# In[22]:


y.head(4)


# In[23]:


from sklearn import preprocessing
x = preprocessing.scale(x)


# In[24]:


x2 = x
from matplotlib import pyplot as plt
import seaborn as sns
for i in range(1,9):
    sns.distplot(x2[i])
    plt.xlabel(features[i])
    plt.show()


# In[25]:


from sklearn.model_selection import train_test_split


# In[26]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=0,stratify=y)


# In[27]:


y_train = y_train.values.ravel()
y_test = y_test.values.ravel()


# In[28]:


print('Shape of x_train = ' + str(x_train.shape))
print('Shape of x_test = ' + str(x_test.shape))
print('Shape of y_train = ' + str(y_train.shape))
print('Shape of y_test = ' + str(y_test.shape))


# In[29]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


# In[30]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.linear_model import Lasso,Ridge


# In[31]:


RFR= RandomForestRegressor()
RFR.fit(x_train,y_train)
pred_RFR=RFR.predict(x_test)
pred_train=RFR.predict(x_train)
print('R2_score:',r2_score(y_test,pred_RFR))
print('R2_score on training Data:',r2_score(y_train,pred_train)*100)
print('Mean Absolute Error:',mean_absolute_error(y_test,pred_RFR))
print('Mean Squared Error:',mean_squared_error(y_test,pred_RFR))
print("Root Mean Squared Error:",np.sqrt(mean_squared_error(y_test,pred_RFR)))


# In[32]:


lasso= Lasso()
lasso.fit(x_train,y_train)
pred_lasso=lasso.predict(x_test)
pred_train=lasso.predict(x_train)
print('R2_score:',r2_score(y_test,pred_lasso))
print('R2_score on training Data:',r2_score(y_train,pred_train)*100)
print('Mean Absolute Error:',mean_absolute_error(y_test,pred_lasso))
print('Mean Squared Error:',mean_squared_error(y_test,pred_lasso))
print("Root Mean Squared Error:",np.sqrt(mean_squared_error(y_test,pred_lasso)))


# In[33]:


GBR= GradientBoostingRegressor()
GBR.fit(x_train,y_train)
pred_GBR=GBR.predict(x_test)
pred_train=GBR.predict(x_train)
print('R2_score:',r2_score(y_test,pred_GBR))
print('R2_score on training Data:',r2_score(y_train,pred_train)*100)
print('Mean Absolute Error:',mean_absolute_error(y_test,pred_GBR))
print('Mean Squared Error:',mean_squared_error(y_test,pred_GBR))
print("Root Mean Squared Error:",np.sqrt(mean_squared_error(y_test,pred_GBR)))


# In[34]:


rd= Ridge()
rd.fit(x_train,y_train)
pred_rd=rd.predict(x_test)
pred_train=rd.predict(x_train)
print('R2_score:',r2_score(y_test,pred_rd))
print('R2_score on training Data:',r2_score(y_train,pred_train)*100)
print('Mean Absolute Error:',mean_absolute_error(y_test,pred_rd))
print('Mean Squared Error:',mean_squared_error(y_test,pred_rd))
print("Root Mean Squared Error:",np.sqrt(mean_squared_error(y_test,pred_rd)))


# In[35]:


from sklearn .tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor()

dtr.fit(x_train,y_train)
pred_dtr=dtr.predict(x_test)
pred_train=dtr.predict(x_train)
print('R2_score:',r2_score(y_test,pred_dtr))
print('R2_score on training Data:',r2_score(y_train,pred_train)*100)
print('Mean Absolute Error:',mean_absolute_error(y_test,pred_dtr))
print('Mean Squared Error:',mean_squared_error(y_test,pred_dtr))
print("Root Mean Squared Error:",np.sqrt(mean_squared_error(y_test,pred_dtr)))


# In[36]:


from sklearn.svm import SVR


# In[37]:


svr=SVR()

svr.fit(x_train,y_train)
pred_svr=svr.predict(x_test)
pred_train=svr.predict(x_train)
print('R2_score:',r2_score(y_test,pred_svr))
print('R2_score on training Data:',r2_score(y_train,pred_train)*100)
print('Mean Absolute Error:',mean_absolute_error(y_test,pred_svr))
print('Mean Squared Error:',mean_squared_error(y_test,pred_svr))
print("Root Mean Squared Error:",np.sqrt(mean_squared_error(y_test,pred_svr)))


# In[38]:


from sklearn.ensemble import ExtraTreesRegressor
etr=ExtraTreesRegressor()
etr.fit(x_train,y_train)
pred_etr=etr.predict(x_test)
pred_train=etr.predict(x_train)
print('R2_score:',r2_score(y_test,pred_etr))
print('R2_score on training Data:',r2_score(y_train,pred_train)*100)
print('Mean Absolute Error:',mean_absolute_error(y_test,pred_etr))
print('Mean Squared Error:',mean_squared_error(y_test,pred_etr))
print("Root Mean Squared Error:",np.sqrt(mean_squared_error(y_test,pred_etr)))


# In[39]:


from sklearn.model_selection import cross_val_score


# In[40]:


score = cross_val_score(RFR,x,y, cv=5, scoring= 'r2')
print(score)
print(score.mean())
print("Difference between R2 score and cross validation score is - ",(r2_score(y_test,pred_RFR) - score.mean())*100)


# In[41]:


score = cross_val_score(GBR,x,y, cv=5, scoring= 'r2')
print(score)
print(score.mean())
print("Difference between R2 score and cross validation score is - ",(r2_score(y_test,pred_GBR) - score.mean())*100)


# In[42]:


score = cross_val_score(rd,x,y, cv=5, scoring= 'r2')
print(score)
print(score.mean())
print("Difference between R2 score and cross validation score is - ",(r2_score(y_test,pred_rd) - score.mean())*100)


# In[43]:


score = cross_val_score(svr,x,y, cv=5, scoring= 'r2')
print(score)
print(score.mean())
print("Difference between R2 score and cross validation score is - ",(r2_score(y_test,pred_svr) - score.mean())*100)


# In[44]:


from sklearn.model_selection import GridSearchCV


# In[45]:


param = {'alpha':[1.0,.05,.4,2], 'fit_intercept':[True,False], 'solver':['auto', 'sud', ' cholesky', 'lsqr', 'sag', 'saga', 'lbfgs'], 'positive': [False,True],'random_state':[1,4,10,20]}


# In[46]:


gscv = GridSearchCV(Ridge(),param,cv=5)
gscv.fit(x_train,y_train)


# In[47]:


gscv.best_params_


# In[48]:


model = Ridge (alpha=0.05,fit_intercept=True,positive=False,random_state=1,solver='auto')


# In[49]:


model.fit(x_train,y_train)
pred=model.predict(x_test)
print('R2_score:',r2_score(y_test,pred))
print('Mean Absolute Error:',mean_absolute_error(y_test,pred))
print('Mean Squared Error:',mean_squared_error(y_test,pred))
print("Root Mean Squared Error:",np.sqrt(mean_squared_error(y_test,pred)))


# In[50]:


import joblib


# In[51]:


from sklearn import metrics 
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# In[52]:


import pickle


# In[53]:


filename = 'Glass.pkl'
pickle.dump(model,open(filename, 'wb'))


# In[54]:


import pickle
loaded_model= pickle.load(open('Glass.pkl', 'rb'))
result = loaded_model.score(x_test,y_test)
print(result*100)


# In[55]:


conclusion= pd.DataFrame([loaded_model.predict(x_test)[:],y_test[:]],index=["Predicted","Original"])
conclusion


# # STUDENT GRADES PROJECT

# In[92]:


df1 = pd.read_csv('Student grades.csv')
df1


# In[93]:


df1.head(10)


# In[94]:


df1.tail(10)


# In[95]:


df1.shape


# In[96]:


print(f"The rows and columns in the dataset:{df.shape}")
print(f"\n The column headers in the dataset:{df.columns}")


# In[97]:


df1.isnull().sum()


# In[98]:


df1.info()


# In[99]:


df1.describe()


# In[100]:


df1.skew()


# In[101]:


df1.head()


# In[102]:


plt.figure(figsize=(10,7))
sns.countplot(x ='HS-101',data = df1)
plt.xlabel('University Of Grades')
plt.ylabel('Count of grades in the dataset')
plt.show()


# In[103]:


sns.pairplot(data=df1,palette = "Dark2")


# In[104]:


index=0
labels = df1['CGPA']
features = df1.drop ('CGPA',axis=1)
for col in features.items():
    plt.figure(figsize=(10,5))
    sns.barplot(x=labels, y= col[index], data=df1, color="green")
    plt.show()


# In[105]:


plt.figure(figsize=(26,14))
sns.heatmap(df1.corr(),annot = True, fmt = '0.2f',linewidth = 0.2, linecolor ='black', cmap = 'Spectral')
plt.xlabel('Figure',fontsize =14)
plt.ylabel('Features_Name',fontsize =14)
plt.title('Descriptive Graph',fontsize =20)
plt.show()


# In[106]:


from scipy.stats import zscore


# In[107]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[108]:


from sklearn import preprocessing
x = preprocessing.scale(x)


# In[109]:


from sklearn.model_selection import train_test_split


# In[110]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=0,stratify=y)


# In[111]:


y_train = y_train.values.ravel()
y_test = y_test.values.ravel()


# In[112]:


print('Shape of x_train = ' + str(x_train.shape))
print('Shape of x_test = ' + str(x_test.shape))
print('Shape of y_train = ' + str(y_train.shape))
print('Shape of y_test = ' + str(y_test.shape))


# In[113]:


df1.shape


# In[114]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.linear_model import Lasso,Ridge


# In[115]:


RFR= RandomForestRegressor()
RFR.fit(x_train,y_train)
pred_RFR=RFR.predict(x_test)
pred_train=RFR.predict(x_train)
print('R2_score:',r2_score(y_test,pred_RFR))
print('R2_score on training Data:',r2_score(y_train,pred_train)*100)
print('Mean Absolute Error:',mean_absolute_error(y_test,pred_RFR))
print('Mean Squared Error:',mean_squared_error(y_test,pred_RFR))
print("Root Mean Squared Error:",np.sqrt(mean_squared_error(y_test,pred_RFR)))


# In[116]:


lasso= Lasso()
lasso.fit(x_train,y_train)
pred_lasso=lasso.predict(x_test)
pred_train=lasso.predict(x_train)
print('R2_score:',r2_score(y_test,pred_lasso))
print('R2_score on training Data:',r2_score(y_train,pred_train)*100)
print('Mean Absolute Error:',mean_absolute_error(y_test,pred_lasso))
print('Mean Squared Error:',mean_squared_error(y_test,pred_lasso))
print("Root Mean Squared Error:",np.sqrt(mean_squared_error(y_test,pred_lasso)))


# In[117]:


GBR= GradientBoostingRegressor()
GBR.fit(x_train,y_train)
pred_GBR=GBR.predict(x_test)
pred_train=GBR.predict(x_train)
print('R2_score:',r2_score(y_test,pred_GBR))
print('R2_score on training Data:',r2_score(y_train,pred_train)*100)
print('Mean Absolute Error:',mean_absolute_error(y_test,pred_GBR))
print('Mean Squared Error:',mean_squared_error(y_test,pred_GBR))
print("Root Mean Squared Error:",np.sqrt(mean_squared_error(y_test,pred_GBR)))


# In[118]:


rd= Ridge()
rd.fit(x_train,y_train)
pred_rd=rd.predict(x_test)
pred_train=rd.predict(x_train)
print('R2_score:',r2_score(y_test,pred_rd))
print('R2_score on training Data:',r2_score(y_train,pred_train)*100)
print('Mean Absolute Error:',mean_absolute_error(y_test,pred_rd))
print('Mean Squared Error:',mean_squared_error(y_test,pred_rd))
print("Root Mean Squared Error:",np.sqrt(mean_squared_error(y_test,pred_rd)))


# In[119]:


from sklearn .tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor()

dtr.fit(x_train,y_train)
pred_dtr=dtr.predict(x_test)
pred_train=dtr.predict(x_train)
print('R2_score:',r2_score(y_test,pred_dtr))
print('R2_score on training Data:',r2_score(y_train,pred_train)*100)
print('Mean Absolute Error:',mean_absolute_error(y_test,pred_dtr))
print('Mean Squared Error:',mean_squared_error(y_test,pred_dtr))
print("Root Mean Squared Error:",np.sqrt(mean_squared_error(y_test,pred_dtr)))


# In[120]:


from sklearn.svm import SVR


# In[121]:


svr=SVR()

svr.fit(x_train,y_train)
pred_svr=svr.predict(x_test)
pred_train=svr.predict(x_train)
print('R2_score:',r2_score(y_test,pred_svr))
print('R2_score on training Data:',r2_score(y_train,pred_train)*100)
print('Mean Absolute Error:',mean_absolute_error(y_test,pred_svr))
print('Mean Squared Error:',mean_squared_error(y_test,pred_svr))
print("Root Mean Squared Error:",np.sqrt(mean_squared_error(y_test,pred_svr)))


# In[122]:


from sklearn.ensemble import ExtraTreesRegressor
etr=ExtraTreesRegressor()
etr.fit(x_train,y_train)
pred_etr=etr.predict(x_test)
pred_train=etr.predict(x_train)
print('R2_score:',r2_score(y_test,pred_etr))
print('R2_score on training Data:',r2_score(y_train,pred_train)*100)
print('Mean Absolute Error:',mean_absolute_error(y_test,pred_etr))
print('Mean Squared Error:',mean_squared_error(y_test,pred_etr))
print("Root Mean Squared Error:",np.sqrt(mean_squared_error(y_test,pred_etr)))


# In[123]:


from sklearn.model_selection import cross_val_score


# In[124]:


score = cross_val_score(RFR,x,y, cv=5, scoring= 'r2')
print(score)
print(score.mean())
print("Difference between R2 score and cross validation score is - ",(r2_score(y_test,pred_RFR) - score.mean())*100)


# In[125]:


score = cross_val_score(GBR,x,y, cv=5, scoring= 'r2')
print(score)
print(score.mean())
print("Difference between R2 score and cross validation score is - ",(r2_score(y_test,pred_GBR) - score.mean())*100)


# In[126]:


score = cross_val_score(rd,x,y, cv=5, scoring= 'r2')
print(score)
print(score.mean())
print("Difference between R2 score and cross validation score is - ",(r2_score(y_test,pred_rd) - score.mean())*100)


# In[127]:


score = cross_val_score(svr,x,y, cv=5, scoring= 'r2')
print(score)
print(score.mean())
print("Difference between R2 score and cross validation score is - ",(r2_score(y_test,pred_svr) - score.mean())*100)


# In[128]:


from sklearn.model_selection import GridSearchCV


# In[129]:


param = {'alpha':[1.0,.05,.4,2], 'fit_intercept':[True,False], 'solver':['auto', 'sud', ' cholesky', 'lsqr', 'sag', 'saga', 'lbfgs'], 'positive': [False,True],'random_state':[1,4,10,20]}


# In[130]:


gscv = GridSearchCV(Ridge(),param,cv=5)
gscv.fit(x_train,y_train)


# In[131]:


gscv.best_params_


# In[132]:


model = Ridge (alpha=0.05,fit_intercept=True,positive=False,random_state=1,solver='auto')


# In[133]:


model.fit(x_train,y_train)
pred=model.predict(x_test)
print('R2_score:',r2_score(y_test,pred))
print('Mean Absolute Error:',mean_absolute_error(y_test,pred))
print('Mean Squared Error:',mean_squared_error(y_test,pred))
print("Root Mean Squared Error:",np.sqrt(mean_squared_error(y_test,pred)))


# In[134]:


import joblib


# In[135]:


from sklearn import metrics 
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# In[136]:


import pickle


# In[138]:


filename = 'Student grades.pkl'
pickle.dump(model,open(filename, 'wb'))


# In[139]:


import pickle
loaded_model= pickle.load(open('Student grades.pkl', 'rb'))
result = loaded_model.score(x_test,y_test)
print(result*100)


# In[140]:


conclusion= pd.DataFrame([loaded_model.predict(x_test)[:],y_test[:]],index=["Predicted","Original"])
conclusion


# In[ ]:




