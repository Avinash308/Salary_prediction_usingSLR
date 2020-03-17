# Salary_prediction_usingSLR
#Using simple linear regression
import numpy as np
import pandas as pd
dataset = pd.read_csv('D:\Datasets\datascience_probs\salary-data-simple-linear-regression\Salary_Data.csv')
df=pd.DataFrame(dataset)
df.head()


# In[27]:


import matplotlib.pyplot as plt
E=df.loc[:,"YearsExperience"]
S=df.loc[:,"Salary"]
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.scatter(E,S)
#HERE WE CAN OBSERVE THAT THERE IS AN LINEAR RELATIONSHIP BETWEEN TWO VARIABLES SO USING SLR


# In[40]:


X = df.iloc[:,:-1].values
print(X)
y=df.iloc[:,1].values
print(y)


# In[42]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)


# In[46]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression() #So this it the model which will be implement
regressor.fit(X_train,y_train)


# In[51]:


y_predict=regressor.predict([[5]])
print(y_predict)

