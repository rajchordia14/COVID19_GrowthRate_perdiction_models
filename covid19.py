#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
df=pd.read_csv('owid-covid-data.csv')
df=df[df['location'].apply(lambda location: location == 'India')]
df = df.reset_index()
del df['index']
del df['iso_code']
del df['location']
del df['date']
df=df.fillna(df.mean())
d={'samples tested':1}
df['tests_units'] = df['tests_units'].map(d)
df['tests_units']=df['tests_units'].fillna(0)
df['tests_units'] = df['tests_units'].astype('float64')
df=df.reset_index()
df=df.rename(columns={'index':'days'})
df.info()


# In[2]:


y= df['total_deaths']
X=df.drop(['total_deaths','total_deaths_per_million'], axis=1)
X=X.iloc[:,:-13]
X


# In[99]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0,test_size=0.25,shuffle=False)
X_train


# In[82]:


from sklearn.ensemble.forest import RandomForestRegressor
forest = RandomForestRegressor(random_state=17)
forest.fit(X_train,y_train)
y_pred = forest.predict(X_test)


# In[83]:


from sklearn.metrics import accuracy_score, precision_score, recall_score


# In[ ]:





# In[84]:


y_pred


# In[85]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_lr_pred=regressor.predict(X_test)


# In[33]:


regressor.score(X_test,y_test)


# In[86]:


from sklearn.metrics import r2_score
r2_score(y_test,y_lr_pred)


# In[87]:


import matplotlib.pyplot as plt

plt.plot(X_test['days'],y_test,"b")
plt.plot(X_test['days'],y_lr_pred,"g")
plt.xlabel('Number of days')
plt.ylabel('Total deaths')


# In[91]:


from sklearn.neural_network import MLPRegressor
mlpreg = MLPRegressor(hidden_layer_sizes = [100,10],solver = 'lbfgs').fit(X_train, y_train)


# In[92]:


y_predict_output = mlpreg.predict(X_test)


# In[93]:


r2_score(y_test,y_predict_output)


# In[94]:


plt.plot(X_test['days'],y_test,"b")
plt.plot(X_test['days'],y_predict_output,"g")


# In[95]:


from sklearn import linear_model
reg = linear_model.BayesianRidge()
reg.fit(X_train, y_train)
y_lr_=reg.predict(X_test)
reg.score(X_test,y_test)


# In[96]:


y_lr_


# In[97]:


plt.plot(X_test['days'],y_test,"b")
plt.plot(X_test['days'],y_lr_,"g")


# In[ ]:





# In[18]:





# In[110]:


pca = decomposition.PCA(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)


# In[104]:


X_pca


# In[109]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, random_state = 0,test_size=0.25,shuffle=False)


# In[106]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_lr_oopred=regressor.predict(X_test)


# In[112]:


from sklearn.metrics import r2_score
r2_score(y_test,y_lr_oopred)


# In[122]:


yt=pd.Series(y_lr_oopred)
y_test=y_test.reset_index()


# In[124]:


del y_test['index']


# In[128]:


yp=y_test.squeeze()


# In[134]:


plt.plot(X_test['days'],yp,"b")
plt.plot(X_test['days'],yt,"g")


# In[ ]:





# In[ ]:





# In[ ]:



