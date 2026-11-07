#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import accuracy_score, f1_score


# In[2]:


train0 = scipy.io.loadmat('training_data_0.mat')
train0 = train0['nim0']
train1 = scipy.io.loadmat('training_data_1.mat')
train1 = train1['nim1']
train_l0 = np.zeros(train0.shape[2])
train_l1 = np.ones(train1.shape[2])
test0 = scipy.io.loadmat('testing_data_0.mat')
test0 = test0['nim0']
test1 = scipy.io.loadmat('testing_data_1.mat')
test1 = test1['nim1']
test_l0 = np.zeros(test0.shape[2])
test_l1 = np.ones(test1.shape[2])


# In[3]:


X_train = np.concatenate((train0, train1), axis = 2)
X_train = np.transpose(X_train)
X_train = np.reshape(X_train, (X_train.shape[0], 28*28))
print(X_train.shape)
y_train = np.concatenate((train_l0, train_l1))
print(y_train.shape)


# In[4]:


X_test = np.concatenate((test0, test1), axis = 2)
X_test = np.transpose(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], 28*28))
print(X_test.shape)
y_test = np.concatenate((test_l0, test_l1))
print(y_test.shape)


# In[5]:


#Task 1 :Feature normalization
norm_mean = np.mean(X_train, axis = 0)
norm_std = np.std(X_train, axis = 0)
X_train_norm = (X_train - norm_mean) / norm_std
print("Normalized X train:\n",X_train_norm)
X_test_norm = (X_test - norm_mean) / norm_std
print("Normalized X test:\n",X_test_norm)


# In[6]:


#Task 2 - PCA using training samples
covariance = np.cov(X_train_norm , rowvar = False, bias=False)
eigen_values , eigen_vectors = np.linalg.eigh(covariance)
print("Covariance matrix",covariance)
print("Eigen vectors:\n",eigen_vectors)
print("Eigen values:\n",eigen_values)


# In[7]:


#Task 3 - Dimension reduction using PCA
sorted_eigen = np.argsort(eigen_values)
eigen_vectors_desc = eigen_vectors[:, sorted_eigen[-2:]].T
print(eigen_vectors_desc.shape)
X_reduced = np.dot(eigen_vectors_desc, X_train_norm.T)
X_reduced = X_reduced.T
print(X_reduced.shape)
X_test_red = np.dot(eigen_vectors_desc, X_test_norm.T)
X_test_red = X_test_red.T
print(X_test_red.shape)

X0 = X_reduced[:5923]
X1 = X_reduced[5923:]
plt.scatter(X0[:,0], X0[:,1], label = 0)
plt.scatter(X1[:,0], X1[:,1], label = 1)
plt.title('Principal Component Analysis - Train')
plt.xlabel('E-1')
plt.ylabel('E-2')
plt.legend()
plt.show()



# In[8]:


X_test0 = X_test_red[:980]
X_test1 = X_test_red[980:]
plt.scatter(X_test0[:,0], X_test0[:,1], label = 0)
plt.scatter(X_test1[:,0], X_test1[:,1], label = 1)
plt.title('Principal Component Analysis - Test')
plt.xlabel('E-1')
plt.ylabel('E-2')
plt.legend()
plt.show()


# In[9]:


#Task 4 - Density Estimation
mean0 = X0.mean(axis = 0)
cov0 = np.cov(X0, rowvar = False,bias=False)
print("Mean for digit 0 class:\n",mean0)
print("Covariance Matrix for digit 0 class:\n",cov0)
mean1 = X1.mean(axis = 0)
cov1 = np.cov(X1, rowvar = False,bias=False)
print("Mean for digit 1 class:\n",mean1)
print("Covariance Matrix for digit 1 class:\n",cov1)


# In[10]:


var0=[]
pdf0 = multivariate_normal(mean=mean0, cov=cov0)
for row in X0:  
    var0.append(pdf0.pdf(row))

var1=[]
pdf1 = multivariate_normal(mean=mean1, cov=cov1)
for row in X1:
    var1.append(pdf1.pdf(row))

plt.plot(var0, marker = '^', linestyle = '', label = 0)
plt.plot(var1, marker = '*',linestyle = '', label = 1)
plt.title('Gaussian Distribution for two classes')
plt.ylabel("probability density function")
plt.xlabel("values")
plt.legend()
plt.show()


# In[11]:


# Task 5 - Bayesian Decision Theory for optimal classification
pri0 = 0.5
pri1 = 0.5
preds = []
for row in X_reduced:
    p_0 = pdf0.pdf(row)
    p_1 = pdf1.pdf(row)
    if(pri0*p_0 > pri1*p_1):
        preds.append(0)
    else:
        preds.append(1)
acc_train = accuracy_score(y_train, preds)
f1_train = f1_score(y_train, preds)
print("Accuracy for training data:", acc_train*100)
print("F1 Score for training data:", f1_train)
preds = []
for row in X_test_red:
    p_0 = pdf0.pdf(row)
    p_1 = pdf1.pdf(row)
    if(pri0*p_0 > pri1*p_1):
        preds.append(0)
    else:
        preds.append(1)
acc_test = accuracy_score(y_test, preds)
f1_test = f1_score(y_test, preds)
print("Accuracy for test data:", acc_test*100)
print("F1 Score for test data:", f1_test)

