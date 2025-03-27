#!/usr/bin/env python
# coding: utf-8

# # **Cleaning the DataSet**

# # _________________________________________________________________________

# ## **Netflix Stock**

# Importing the libraries.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import tree
import statistics
import seaborn as sns


# In[2]:


#Importing the CSV file
df = pd.read_csv('NFLX.csv')


# _________________________________________________________

# Observing the first five lines using the head function.

# In[3]:


df.head()


# Checking for Null...

# In[4]:


df.isnull().sum()


# *This CSV contains no Null values*

# Checking for duplicates...

# In[5]:


df.duplicated().sum()


# *This CSV contains no dupicated Values*

# Our dataset contains 1009 rows and 7 columns.

# In[6]:


df.shape
df.info


# In[7]:


df.columns


# Statistical Information

# In[8]:


df.describe()


# In[9]:


df.nunique()


# ### **MEAN AND MEDIAN**

# In[10]:


print("Mean price is:", statistics.mean(df['Open']))
print("Median price is:", statistics.median(df['Open']))


# In[11]:


#Date and Time 
df['Date']=pd.to_datetime(df['Date'],format='%Y-%m-%d')
df = df.set_index('Date')


# In[12]:


plt.subplots(figsize=(25, 8))
plt.title("Open Price vs Close Price")
plt.plot(df['Open'], color='red', linestyle='solid',  label = 'Open Price')
plt.plot(df['Close'], color='blue', linestyle='dashed',  label = 'Close Price')
plt.xlabel("Date")
plt.ylabel("Open vs Close Price")
plt.legend(loc="upper left")
plt.show()


# This graph shows the path for both the open and closed prices of the stock of Netflix from 2018 to 2020. Both prices follow an almost identical part throughout all of the graph. The data starts at January of 2018 where the price is less than 300 and it slowly rises throughout the next seven months until it becomes greater than 400. After that, it starts to descend until it reaches it's lowest point in January of 2019 below 300. After the fall, it rises fast to around 400 but remains constant afterwards for the next seven months. Later, the price starts going down below 300 again and reaches its lowest between July of 2019 and January of 2020. However, once the decrease period passes, it starts growing and does not slow down for the rest of the graph until it becomes greater than 500. This becomes the highest prices that have been seen in the graph.

# ## Prepare Data for Prediction

# In[13]:


from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor


# In[14]:


#Split data to predict jan2022
train = df.loc['2018-02-05':'2021-12-31']
test = df.loc['2022-01-01':'2022-01-31']


# In[15]:


#Split Data
X_train = train.drop(columns = ['Open'])
y_train = train['Open']

X_test = test.drop(columns = ['Open'])
y_test = test['Open']


# In[16]:


rf = RandomForestRegressor(max_depth=20, random_state = 42,  n_estimators=150)
rf.fit(X_train, y_train)


# In[17]:


pre=rf.predict(X_test)
train_pre=rf.predict(X_train)


# In[18]:


prediction_df=X_test.copy()
prediction_df['Open']=y_test
prediction_df['Predicted Price']=pre
prediction_df.head()


# In[19]:


plt.subplots(figsize=(25, 8))
plt.title("Open Price Prediction")
#plt.plot(prediction_df['Open'], color='red', linestyle='solid')
plt.plot(df['Open'], color='red', linestyle='solid', label = 'Actual Price')
plt.plot(prediction_df['Predicted Price'], color='blue', linestyle='dashed', label = 'Predicted Price')
plt.xlabel("Date")
plt.ylabel("Predicted Price")
plt.legend(loc="upper left")
plt.show()


# This graph shows the path that the open price of Netflix stocks has followed starting in the year 2018 and ending in the year 2020. Using this data, we will create a prediction of how the prices should look at the beginning of 2022 (seen in the next graph).

# ## **Predicted Price VS Actual Price**

# In[20]:


plt.subplots(figsize=(25, 8))
plt.title("Open Price Prediction")
plt.plot(prediction_df['Open'], color='red', linestyle='solid',  label = 'Actual Price')
plt.plot(prediction_df['Predicted Price'], color='blue', linestyle='solid', label = 'Predicted Price')
plt.xlabel("Date")
plt.ylabel("Predicted Price")
plt.legend(loc="upper left")
plt.show()


# Using the data collected from previous years, the graph shows the predicted price for Netflix stocks for the year 2022. When compared to the actual data collected for the prices that year, we can see that the prediction follows a very similar path to the actual data collected. Starting at 600, they both start slowly decreasing in the first days of the year until the price reaches the 500 mark and almost at the same time, they drop significantly at the end of the first moth of 2022, bringing the price to around 400.

# # **Accuracy**

# In[21]:


from sklearn import metrics


# In[22]:


print("Mean Absolute Error:", round(metrics.mean_absolute_error(y_test, pre), 4))
print("Mean Squared Error:", round(metrics.mean_squared_error(y_test, pre), 4))
print(f'Train Score : {rf.score(X_train, y_train) * 100:.2f}% and Test Score : {rf.score(X_test, y_test) * 100:.2f}% using Random Tree.')
errors = abs(pre - y_test)
mape = 100 * (errors / y_test)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.') 

