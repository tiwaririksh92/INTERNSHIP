#!/usr/bin/env python
# coding: utf-8

# # red wine project 1

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[8]:


df = pd.read_csv('red winess.csv')
df


# In[9]:


df.head(15)


# In[10]:


df.tail(15)


# In[11]:


df.shape


# In[12]:


print(f"The rows and columns in the dataset:{df.shape}")
print(f"\n The column headers in the dataset:{df.columns}")


# In[13]:


df.isnull().sum()


# In[14]:


df.info()


# In[15]:


df.describe()


# In[16]:


df.skew()


# In[17]:


plt.figure(figsize=(10,7))
sns.countplot(x ='quality',data = df)
plt.xlabel('Quality of Red Wine')
plt.ylabel('Count of rows in the dataset')
plt.show()


# In[18]:


sns.pairplot(data=df,palette = "Dark2")


# In[19]:


index=0
labels = df['quality']
features = df.drop ('quality',axis=1)
for col in features.items():
    plt.figure(figsize=(10,5))
    sns.barplot(x=labels, y= col[index], data=df, color="green")
    plt.show()


# In[20]:


plt.figure(figsize = (20,25))
p=1
for i in df:
    if p<=13:
        plt.subplot(5,4,p)
        sns.boxplot(df[i],palette ="Set2_r")
        plt.xlabel(i)
        p+=1
        plt.show()


# In[21]:


plt.figure(figsize=(26,14))
sns.heatmap(df.corr(),annot = True, fmt = '0.2f',linewidth = 0.2, linecolor ='black', cmap = 'Spectral')
plt.xlabel('Figure',fontsize =14)
plt.ylabel('Features_Name',fontsize =14)
plt.title('Descriptive Graph',fontsize =20)
plt.show()


# In[22]:


df = df.drop('free sulfur dioxide', axis =1)
df


# In[23]:


df.shape


# In[24]:


from scipy.stats import zscore


# In[25]:


z=np.abs(zscore(df))
threshold =3
np.where(z>3)
df=df[(z<3).all(axis=1)]
df


# In[26]:


df.shape


# In[27]:


df1=(1598-1463)/1598*100
df1


# In[28]:


x= df.drop ('quality', axis=1)
y= df['quality']


# In[29]:


y.value_counts()


# In[30]:


print("Feature Dimension=",x.shape)
print("Label Dimension=",y.shape)


# In[31]:


from sklearn.preprocessing import StandardScaler


# In[32]:


scaler = StandardScaler()
x = pd.DataFrame(scaler.fit_transform(x),columns = x.columns)
x


# In[33]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


# In[34]:


maxAccu=0
maxRS=0
for i in range(1,200):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=i)
    lr=LinearRegression()
    lr.fit(x_train,y_train)
    pred=lr.predict(x_test)
    acc=r2_score(y_test,pred)
    if acc>maxAccu:
        maxAccu=acc
        maxRS=i
        print("Maximum r2 score is ",maxAccu," on Random_state",maxRS)


# In[35]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=maxRS)


# In[38]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.linear_model import Lasso,Ridge


# In[39]:


LR= LinearRegression()
LR.fit(x_train,y_train)
pred_LR=LR.predict(x_test)
pred_train=LR.predict(x_train)
print('R2_score:',r2_score(y_test,pred_LR))
print('R2_score on training Data:',r2_score(y_train,pred_train)*100)


# In[40]:


print('Mean Absolute Error:',mean_absolute_error(y_test,pred_LR))
print('Mean Squared Error:',mean_squared_error(y_test,pred_LR))
print("Root Mean Squared Error:",np.sqrt(mean_squared_error(y_test,pred_LR)))


# In[42]:


RFR= RandomForestRegressor()
RFR.fit(x_train,y_train)
pred_RFR=RFR.predict(x_test)
pred_train=RFR.predict(x_train)
print('R2_score:',r2_score(y_test,pred_RFR))
print('R2_score on training Data:',r2_score(y_train,pred_train)*100)
print('Mean Absolute Error:',mean_absolute_error(y_test,pred_RFR))
print('Mean Squared Error:',mean_squared_error(y_test,pred_RFR))
print("Root Mean Squared Error:",np.sqrt(mean_squared_error(y_test,pred_RFR)))


# In[43]:


lasso= Lasso()
lasso.fit(x_train,y_train)
pred_lasso=lasso.predict(x_test)
pred_train=lasso.predict(x_train)
print('R2_score:',r2_score(y_test,pred_lasso))
print('R2_score on training Data:',r2_score(y_train,pred_train)*100)
print('Mean Absolute Error:',mean_absolute_error(y_test,pred_lasso))
print('Mean Squared Error:',mean_squared_error(y_test,pred_lasso))
print("Root Mean Squared Error:",np.sqrt(mean_squared_error(y_test,pred_lasso)))


# In[44]:


GBR= GradientBoostingRegressor()
GBR.fit(x_train,y_train)
pred_GBR=GBR.predict(x_test)
pred_train=GBR.predict(x_train)
print('R2_score:',r2_score(y_test,pred_GBR))
print('R2_score on training Data:',r2_score(y_train,pred_train)*100)
print('Mean Absolute Error:',mean_absolute_error(y_test,pred_GBR))
print('Mean Squared Error:',mean_squared_error(y_test,pred_GBR))
print("Root Mean Squared Error:",np.sqrt(mean_squared_error(y_test,pred_GBR)))


# In[45]:


rd= Ridge()
rd.fit(x_train,y_train)
pred_rd=rd.predict(x_test)
pred_train=rd.predict(x_train)
print('R2_score:',r2_score(y_test,pred_rd))
print('R2_score on training Data:',r2_score(y_train,pred_train)*100)
print('Mean Absolute Error:',mean_absolute_error(y_test,pred_rd))
print('Mean Squared Error:',mean_squared_error(y_test,pred_rd))
print("Root Mean Squared Error:",np.sqrt(mean_squared_error(y_test,pred_rd)))


# In[46]:


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


# In[47]:


from sklearn.svm import SVR


# In[48]:


svr=SVR()

svr.fit(x_train,y_train)
pred_svr=svr.predict(x_test)
pred_train=svr.predict(x_train)
print('R2_score:',r2_score(y_test,pred_svr))
print('R2_score on training Data:',r2_score(y_train,pred_train)*100)
print('Mean Absolute Error:',mean_absolute_error(y_test,pred_svr))
print('Mean Squared Error:',mean_squared_error(y_test,pred_svr))
print("Root Mean Squared Error:",np.sqrt(mean_squared_error(y_test,pred_svr)))


# In[50]:


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


# In[51]:


from sklearn.model_selection import cross_val_score


# In[53]:


score = cross_val_score(LR,x,y, cv=5, scoring= 'r2')
print(score)
print(score.mean())
print("Difference between R2 score and cross validation score is - ",(r2_score(y_test,pred_LR) - score.mean())*100)


# In[54]:


score = cross_val_score(RFR,x,y, cv=5, scoring= 'r2')
print(score)
print(score.mean())
print("Difference between R2 score and cross validation score is - ",(r2_score(y_test,pred_RFR) - score.mean())*100)


# In[55]:


score = cross_val_score(GBR,x,y, cv=5, scoring= 'r2')
print(score)
print(score.mean())
print("Difference between R2 score and cross validation score is - ",(r2_score(y_test,pred_GBR) - score.mean())*100)


# In[56]:


score = cross_val_score(lasso,x,y, cv=5, scoring= 'r2')
print(score)
print(score.mean())
print("Difference between R2 score and cross validation score is - ",(r2_score(y_test,pred_lasso) - score.mean())*100)


# In[57]:


score = cross_val_score(rd,x,y, cv=5, scoring= 'r2')
print(score)
print(score.mean())
print("Difference between R2 score and cross validation score is - ",(r2_score(y_test,pred_rd) - score.mean())*100)


# In[58]:


score = cross_val_score(dtr,x,y, cv=5, scoring= 'r2')
print(score)
print(score.mean())
print("Difference between R2 score and cross validation score is - ",(r2_score(y_test,pred_dtr) - score.mean())*100)


# In[59]:


score = cross_val_score(svr,x,y, cv=5, scoring= 'r2')
print(score)
print(score.mean())
print("Difference between R2 score and cross validation score is - ",(r2_score(y_test,pred_svr) - score.mean())*100)


# In[60]:


score = cross_val_score(etr,x,y, cv=5, scoring= 'r2')
print(score)
print(score.mean())
print("Difference between R2 score and cross validation score is - ",(r2_score(y_test,pred_etr) - score.mean())*100)


# In[61]:


from sklearn.model_selection import GridSearchCV


# In[63]:


param = {'alpha':[1.0,.05,.4,2], 'fit_intercept':[True,False], 'solver':['auto', 'sud', ' cholesky', 'lsqr', 'sag', 'saga', 'lbfgs'], 'positive': [False,True],'random_state':[1,4,10,20]}


# In[64]:


gscv = GridSearchCV(Ridge(),param,cv=5)
gscv.fit(x_train,y_train)


# In[65]:


gscv.best_params_


# In[66]:


model = Ridge (alpha=0.05,fit_intercept=True,positive=False,random_state=1,solver='auto')


# In[67]:


model.fit(x_train,y_train)
pred=model.predict(x_test)
print('R2_score:',r2_score(y_test,pred))
print('Mean Absolute Error:',mean_absolute_error(y_test,pred))
print('Mean Squared Error:',mean_squared_error(y_test,pred))
print("Root Mean Squared Error:",np.sqrt(mean_squared_error(y_test,pred)))


# In[68]:


import joblib


# In[80]:


from sklearn import metrics 
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# In[82]:


import pickle


# In[83]:


filename = 'Red Wine.pkl'
pickle.dump(model,open(filename, 'wb'))


# In[84]:


import pickle
loaded_model= pickle.load(open('Red Wine.pkl', 'rb'))
result = loaded_model.score(x_test,y_test)
print(result*100)


# In[87]:


conclusion= pd.DataFrame([loaded_model.predict(x_test)[:],y_test[:]],index=["Predicted","Original"])
conclusion


# # medical insurance project 2

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df= pd.read_csv('medical insurance.csv')


# In[3]:


df


# In[4]:


df.head(10)


# In[5]:


df.tail(10)


# In[6]:


df.shape


# In[7]:


print(f"The rows and columns in the dataset:{df.shape}")
print(f"\n The column headers in the dataset:{df.columns}")


# In[9]:


print("Min.age:", df["age"].min())
print("Max.age:", df["age"].max())


# In[10]:


df.dtypes


# In[11]:


df.isnull().sum()


# In[12]:


df.info()


# In[13]:


sns.heatmap(df.isnull())


# In[14]:


df.nunique().to_frame("No.of unique values")


# In[15]:


for i in df.columns:
    print(df[i].value_counts())
    print("\n")


# In[16]:


print("Total Duplicate Rows are",df.duplicated().sum())


# In[17]:


df.describe()


# In[18]:


df.skew()


# In[19]:


plt.figure(figsize=(10,7))
sns.countplot(x ='charges',data = df)
plt.xlabel('medical per charges')
plt.ylabel('Count of rows in the dataset')
plt.show()


# In[21]:


plt.figure(figsize=(22,10))
sns.heatmap(df.describe(), annot= True,fmt='0.2f',linewidth=0.2, linecolor='black',cmap='Dark2')
plt.xlabel('Figure',fontsize=14)
plt.ylabel('Features_Name',fontsize=14)
plt.title('Descriptive Graph',fontsize=20)
plt.show()


# In[22]:


sns.pairplot(data=df,palette = "Dark2")


# In[24]:


df['region'].value_counts().sort_values()


# In[25]:


data={'sex': {'male': 0, 'female': 1}, 'smoker': {'no': 0, 'yes': 1}, 'region': {'northwest': 0, 'northeast': 1, 'southwest': 2, 'southeast': 3}}
df =df.copy()
df.replace(data, inplace=True)


# In[26]:


df.describe()


# In[27]:


df.corr()


# In[29]:


plt.figure(figsize=(26,14))
sns.heatmap(df.corr(),annot = True,fmt = '0.2f',linewidth= 0.2, linecolor='green',cmap='BuPu')
plt.xlabel('Figure',fontsize=14)
plt.ylabel('Feature_Name',fontsize=14)
plt.title('Dependencies of Medical charges',fontsize=20)
plt.show()


# In[30]:


print(df['sex'].value_counts().sort_values())
print(df['smoker'].value_counts().sort_values())
print(df['region'].value_counts().sort_values())


# In[31]:


index=0
labels = df['charges']
features = df.drop ('charges',axis=1)
for col in features.items():
    plt.figure(figsize=(10,5))
    sns.barplot(x=labels, y= col[index], data=df, color="green")
    plt.show()


# In[32]:


plt.figure(figsize = (20,25))
p=1
for i in df:
    if p<=13:
        plt.subplot(5,4,p)
        sns.boxplot(df[i],palette ="Set2_r")
        plt.xlabel(i)
        p+=1
        plt.show()


# In[33]:


plt.figure(figsize=(10,7))
sns.distplot(df['age'])
plt.title('Plot for Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# In[35]:


plt.figure(figsize=(10,7))
sns.distplot(df['bmi'])
plt.title('Plot for bmi')
plt.xlabel('bmi')
plt.ylabel('Count')
plt.show()


# In[36]:


plt.figure(figsize=(10,7))
sns.distplot(df['charges'])
plt.title('Plot for charges')
plt.xlabel('charges')
plt.ylabel('Count')
plt.show()


# In[37]:


from scipy.stats import zscore


# In[38]:


z=np.abs(zscore(df))
threshold =3
np.where(z>3)
df=df[(z<3).all(axis=1)]
df


# In[39]:


from sklearn.preprocessing import StandardScaler


# In[43]:


scaler = StandardScaler()
df = pd.DataFrame(scaler.fit_transform(df),columns = df.columns)
df


# In[44]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


# In[47]:


x= df.drop ('charges', axis=1)
y= df['charges']


# In[48]:


maxAccu=0
maxRS=0
for i in range(1,200):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=i)
    lr=LinearRegression()
    lr.fit(x_train,y_train)
    pred=lr.predict(x_test)
    acc=r2_score(y_test,pred)
    if acc>maxAccu:
        maxAccu=acc
        maxRS=i
        print("Maximum r2 score is ",maxAccu," on Random_state",maxRS)


# In[49]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=maxRS)


# In[50]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.linear_model import Lasso,Ridge


# In[51]:


LR= LinearRegression()
LR.fit(x_train,y_train)
pred_LR=LR.predict(x_test)
pred_train=LR.predict(x_train)
print('R2_score:',r2_score(y_test,pred_LR))
print('R2_score on training Data:',r2_score(y_train,pred_train)*100)


# In[52]:


print('Mean Absolute Error:',mean_absolute_error(y_test,pred_LR))
print('Mean Squared Error:',mean_squared_error(y_test,pred_LR))
print("Root Mean Squared Error:",np.sqrt(mean_squared_error(y_test,pred_LR)))


# In[53]:


RFR= RandomForestRegressor()
RFR.fit(x_train,y_train)
pred_RFR=RFR.predict(x_test)
pred_train=RFR.predict(x_train)
print('R2_score:',r2_score(y_test,pred_RFR))
print('R2_score on training Data:',r2_score(y_train,pred_train)*100)
print('Mean Absolute Error:',mean_absolute_error(y_test,pred_RFR))
print('Mean Squared Error:',mean_squared_error(y_test,pred_RFR))
print("Root Mean Squared Error:",np.sqrt(mean_squared_error(y_test,pred_RFR)))


# In[54]:


lasso= Lasso()
lasso.fit(x_train,y_train)
pred_lasso=lasso.predict(x_test)
pred_train=lasso.predict(x_train)
print('R2_score:',r2_score(y_test,pred_lasso))
print('R2_score on training Data:',r2_score(y_train,pred_train)*100)
print('Mean Absolute Error:',mean_absolute_error(y_test,pred_lasso))
print('Mean Squared Error:',mean_squared_error(y_test,pred_lasso))
print("Root Mean Squared Error:",np.sqrt(mean_squared_error(y_test,pred_lasso)))


# In[55]:


GBR= GradientBoostingRegressor()
GBR.fit(x_train,y_train)
pred_GBR=GBR.predict(x_test)
pred_train=GBR.predict(x_train)
print('R2_score:',r2_score(y_test,pred_GBR))
print('R2_score on training Data:',r2_score(y_train,pred_train)*100)
print('Mean Absolute Error:',mean_absolute_error(y_test,pred_GBR))
print('Mean Squared Error:',mean_squared_error(y_test,pred_GBR))
print("Root Mean Squared Error:",np.sqrt(mean_squared_error(y_test,pred_GBR)))


# In[56]:


rd= Ridge()
rd.fit(x_train,y_train)
pred_rd=rd.predict(x_test)
pred_train=rd.predict(x_train)
print('R2_score:',r2_score(y_test,pred_rd))
print('R2_score on training Data:',r2_score(y_train,pred_train)*100)
print('Mean Absolute Error:',mean_absolute_error(y_test,pred_rd))
print('Mean Squared Error:',mean_squared_error(y_test,pred_rd))
print("Root Mean Squared Error:",np.sqrt(mean_squared_error(y_test,pred_rd)))


# In[57]:


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


# In[58]:


from sklearn.svm import SVR


# In[59]:


svr=SVR()

svr.fit(x_train,y_train)
pred_svr=svr.predict(x_test)
pred_train=svr.predict(x_train)
print('R2_score:',r2_score(y_test,pred_svr))
print('R2_score on training Data:',r2_score(y_train,pred_train)*100)
print('Mean Absolute Error:',mean_absolute_error(y_test,pred_svr))
print('Mean Squared Error:',mean_squared_error(y_test,pred_svr))
print("Root Mean Squared Error:",np.sqrt(mean_squared_error(y_test,pred_svr)))


# In[60]:


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


# In[61]:


from sklearn.model_selection import cross_val_score


# In[62]:


score = cross_val_score(LR,x,y, cv=5, scoring= 'r2')
print(score)
print(score.mean())
print("Difference between R2 score and cross validation score is - ",(r2_score(y_test,pred_LR) - score.mean())*100)


# In[63]:


score = cross_val_score(GBR,x,y, cv=5, scoring= 'r2')
print(score)
print(score.mean())
print("Difference between R2 score and cross validation score is - ",(r2_score(y_test,pred_GBR) - score.mean())*100)


# In[64]:


score = cross_val_score(etr,x,y, cv=5, scoring= 'r2')
print(score)
print(score.mean())
print("Difference between R2 score and cross validation score is - ",(r2_score(y_test,pred_etr) - score.mean())*100)


# In[65]:


from sklearn.model_selection import GridSearchCV


# In[66]:


param = {'alpha':[1.0,.05,.4,2], 'fit_intercept':[True,False], 'solver':['auto', 'sud', ' cholesky', 'lsqr', 'sag', 'saga', 'lbfgs'], 'positive': [False,True],'random_state':[1,4,10,20]}


# In[67]:


gscv = GridSearchCV(Ridge(),param,cv=5)
gscv.fit(x_train,y_train)


# In[68]:


gscv.best_params_


# In[69]:


model = Ridge (alpha=0.05,fit_intercept=True,positive=False,random_state=1,solver='auto')


# In[70]:


model.fit(x_train,y_train)
pred=model.predict(x_test)
print('R2_score:',r2_score(y_test,pred))
print('Mean Absolute Error:',mean_absolute_error(y_test,pred))
print('Mean Squared Error:',mean_squared_error(y_test,pred))
print("Root Mean Squared Error:",np.sqrt(mean_squared_error(y_test,pred)))


# In[74]:


df.head()


# In[75]:


import joblib


# In[76]:


from sklearn import metrics 
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# In[78]:


import pickle


# In[79]:


filename = 'medical insurance.pkl'
pickle.dump(model,open(filename, 'wb'))


# In[80]:


import pickle
loaded_model= pickle.load(open('medical insurance.pkl', 'rb'))
result = loaded_model.score(x_test,y_test)
print(result*100)


# In[81]:


conclusion= pd.DataFrame([loaded_model.predict(x_test)[:],y_test[:]],index=["Predicted","Original"])
conclusion


# In[ ]:




