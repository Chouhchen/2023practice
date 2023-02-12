#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
# y= a x *x + b x, a =2, b =2
# create random data samples
np.random.seed(0)
noise = np.random.rand(100, 1)
x = np.random.rand(100, 1)
y = 2*x*x + 2*x + noise

plt.scatter(x,y,s=10)
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[12]:


# create model
from sklearn.linear_model import LinearRegression
lModel = LinearRegression(fit_intercept = True)
lModel.fit(x,y)
predicted = lModel.predict(x)


# In[13]:


# model evaluation
from sklearn import metrics
print('R2 score:', lModel.score(x,y))
mse = metrics.mean_squared_error(y, predicted)
print('MSE score: ', mse)  # other evaluation scores incl. MAE, RMSE


# In[25]:


#visualize
plt.scatter(x, y, s=10, label='True')
plt.scatter(x, predicted, color="r", s=10, label='Predicted')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show


# In[19]:


# model function y = a * x + b
coef = lModel.coef_
intercept = lModel.intercept_
print("a = ", coef[0][0])
print("b = ", intercept[0])


# In[20]:


#Polynomia Regression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


# In[30]:


def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression( **kwargs))
# tests for multiple degrees : 2 
ypred=PolynomialRegression(degree = 2).fit(x,y).predict(x)
plt.plot(x, y, 'o')
plt.plot(x, ypred)


# In[37]:


# test data with multple degrees

x_test = np.linspace(-0.1,1.1, 500)[:, None]
plt.scatter(x.ravel(), y, color='black')
 # degree 1, 10, 20
for degree in [1,10,20]:
    y_test = PolynomialRegression(degree).fit(x,y).predict(x_test)
    plt.plot(x_text.ravel(), y_test, label='degree={}'.format(degree))
plt.xlim(0.0, 1.0)
plt.ylim(0, 5)
plt.legend(loc='best')


# In[ ]:




