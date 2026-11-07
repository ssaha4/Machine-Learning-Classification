#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
import matplotlib.pyplot as plt



# In[2]:


# load train data
train1 = np.loadtxt("Train1.txt")
print(train1.shape)
train2 = np.loadtxt("Train2.txt")
total_train=np.concatenate((train1, train2), axis=0)


# In[3]:


# normalization on train data
norm_mean = np.mean(total_train, axis = 0)
norm_std = np.std(total_train, axis = 0)
total_train_norm = (total_train - norm_mean) / norm_std
print(total_train_norm.shape)


# In[4]:


#Split train data in train and validation set
train_norm_class1=total_train_norm[:2000]
train_norm_class2=total_train_norm[2000:4000]
X_train_norm=np.concatenate((train_norm_class1[:1500],train_norm_class2[:1500]),axis=0)
Valid_X_train_norm=np.concatenate((train_norm_class1[1500:2000],train_norm_class2[1500:2000]),axis=0)
print(X_train_norm.shape)
print(Valid_X_train_norm.shape)


# In[5]:


# load Test data
test1 = np.loadtxt("Test1.txt")
test2 = np.loadtxt("Test2.txt")


# In[6]:


#normalize test data
X_test=np.concatenate((test1, test2), axis=0)
X_test_norm = (X_test - norm_mean) / norm_std
print(X_test_norm.shape)


# In[7]:


#train data labels
y_train_class1=np.zeros((1500,1))
y_train_class2=np.ones((1500,1))
y_train=np.concatenate((y_train_class1, y_train_class2), axis=0)
print(y_train)


# In[8]:


#validation data labels
valid_y_train_class1=np.zeros((500,1))
valid_y_train_class2=np.ones((500,1))
valid_y_train=np.concatenate((valid_y_train_class1, valid_y_train_class2), axis=0)
print(valid_y_train.shape)


# In[9]:


#test data labels
y_test_class1=np.zeros((1000,1))
y_test_class2=np.ones((1000,1))
y_test=np.concatenate((y_test_class1, y_test_class2), axis=0)
print(y_test.shape)


# In[10]:


#initialize parameters and weights
params=dict()
def params_initialize(inputs,hidden,outputs,learning_rate,decayfactor):
    np.random.seed(1)
    params["W1"]=np.random.randn(inputs,hidden)
    params["b1"]=np.random.randn(hidden,)
    params["W2"]=np.random.randn(hidden,outputs)
    params["b2"]=np.random.randn(outputs)
    params["lear_rate"]=learning_rate
    params["decy_fctr"]=decayfactor
    params["prev_delta_W1"]=0
    params["prev_delta_W2"]=0
    params["prev_delta_b1"]=0
    params["prev_delta_b2"]=0
    return params



# In[11]:


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# In[12]:


def sigmoid_derivative(x):
    s = sigmoid(x)
    ds = s*(1-s)
    return ds


# In[13]:


def relu(x):
	return np.maximum(0.0, x)


# In[14]:


def reluDerivative(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x


# In[15]:


def MSE(t,z):
    return 0.5*((t - z) ** 2)


# In[16]:


def derivative_MSE(t,z):
    return -(t-z)
    


# In[17]:


#forward_propagation
fparams=dict()
def forward_propagation(X_norm,params):
    Z1 = X_norm.dot(params['W1']) + params['b1']
    A1 = relu(Z1)
    Z2 = A1.dot(params['W2']) + params['b2']
    y_output = sigmoid(Z2)
    fparams['Z1']=Z1
    fparams['A1']=A1
    fparams['Z2']=Z2
    fparams['y_output']=y_output
    return fparams
    
    


# In[18]:


#back propagation
bparams=dict()
def back_propagation(inital_params,forward_params,y_target,X_norm):
    derv_error=derivative_MSE(y_target,forward_params['y_output'])
    derv_output = sigmoid_derivative(forward_params['y_output'])

    d_output = derv_error* derv_output
    delta_W2=np.dot(forward_params['A1'].T, d_output)
    #print(delta_W2.shape)

    derv_hidden = reluDerivative(forward_params['A1'])

    derv_dot_hidden = np.dot(d_output, inital_params['W2'].T)
    d_hidden = derv_dot_hidden * derv_hidden

    delta_W1=np.dot(X_norm.T,d_hidden)
    #print(delta_W1.shape)

    delta_b2=np.sum(d_output, axis=0)
    delta_b1=np.sum(d_hidden, axis=0) 
#     print(delta_b2.shape)
#     print(delta_b1.shape)
    
    W1=inital_params['W1']
    W2=inital_params['W2']
    b1=inital_params['b1']
    b2=inital_params['b2']
    learning_rate=inital_params["lear_rate"]
    decayfactor= inital_params["decy_fctr"]
    prev_delta_W1= inital_params["prev_delta_W1"]
    prev_delta_W2=inital_params["prev_delta_W2"]
    prev_delta_b1=inital_params["prev_delta_b1"]
    prev_delta_b2=inital_params["prev_delta_b2"]
    

    
    gradW1=decayfactor*prev_delta_W1-learning_rate*delta_W1
    gradW2=decayfactor*prev_delta_W2-learning_rate*delta_W2
    gradb1=decayfactor*prev_delta_b1-learning_rate*delta_b1
    gradb2=decayfactor*prev_delta_b2-learning_rate*delta_b2
    
    W1=W1+gradW1
    W2=W2+gradW2
    b1=b1+gradb1
    b2=b2+gradb2
    
    prev_delta_W1=gradW1
    prev_delta_W2=gradW2
    prev_delta_b1=gradb1
    prev_delta_b2=gradb2
    
    
    
    bparams['W1']=W1
    bparams['W2']=W2
    bparams['b1']=b1
    bparams['b2']=b2
    bparams['lear_rate']=learning_rate
    bparams["decy_fctr"]=decayfactor
    bparams["prev_delta_W1"]=prev_delta_W1
    bparams["prev_delta_W2"]=prev_delta_W2
    bparams["prev_delta_b1"]=prev_delta_b1
    bparams["prev_delta_b2"]=prev_delta_b2
    
    return bparams
    







# In[19]:


#plot learning curves for train,validation and test data
epochs=2000
inputs=2
outputs=1
learning_rate=0.00005
decayfactor=0.7

nh=[2,4,6,8,10]

for hidden in nh:
    inital_params=params_initialize(inputs,hidden,outputs,learning_rate,decayfactor)
    prev_error_validset=float('inf')
    training_error=[]
    validation_error=[]
    testing_error=[]
    total_epochs=[]
    for e in range(epochs):
        total_epochs.append(e)
        forward_params=forward_propagation(X_train_norm,inital_params)
        inital_params=back_propagation(inital_params,forward_params,y_train,X_train_norm)
        error_train=MSE(y_train,forward_params['y_output'])
        totalerror_train=np.sum(error_train)
        training_error.append(totalerror_train)

    
        forward_params_validset=forward_propagation(Valid_X_train_norm,inital_params)
        error_validset=MSE(valid_y_train,forward_params_validset['y_output'])
        totalerror_validset=np.sum(error_validset)
        validation_error.append(totalerror_validset)


        forward_params_test=forward_propagation(X_test_norm,inital_params)
        error_test=MSE(y_test,forward_params_test['y_output'])
        totalerror_test=np.sum(error_test)
        testing_error.append(totalerror_test) 


        if( totalerror_validset>prev_error_validset):
            break
        else:
            prev_error_validset=totalerror_validset
    

    plt.title('learning curves for nh: '+str(hidden))
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss/Error function(J)')    
    plt.plot(total_epochs, training_error,label="train")
    plt.plot(total_epochs, validation_error,label="validation")
    plt.plot(total_epochs, testing_error,label="test")
    plt.legend()
    plt.show()
    
    forward_params_test=forward_propagation(X_test_norm,inital_params)
    a= forward_params_test['y_output']
    for i in range(len(a)):
        if a[i][0]>=0.5:
            a[i][0]=1
        else:
            a[i][0]=0
    sum=0
    for k in range(len(y_test)):
        if y_test[k][0]==a[k][0]:
            sum+=1
            
    acc=(sum/len(y_test))*100
    print('Accuarcy for hidden unit '+str(hidden)+' is : '+str(round(acc,2)))
            
    
    

    
    
    
    
    
    
    


# In[ ]:




