# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 17:39:19 2018

@author: AkashSrivastava
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import plotly
from plotly.graph_objs import Heatmap,Layout,Bar,Scatter,Line
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import keras
import os


#########################################################      Function to be used in program    ######################
#Function to plot graph of attribute with target attribute
def plot_function(feature,data,title,xax,yax):
    a=data[feature].value_counts()
    type(a)
    a=pd.DataFrame(a)
    a.columns=['count']
    a[feature]=a.index
    sum_list=[]
    for ab in a[feature]:
        su=0
        for x in range(len(data)):
            if ab==data[feature][x]:
                su=su+data['charges'][x]
        sum_list.append(su)
    a['Sum_All']=sum_list
    a['Avg']=a['Sum_All']/a['count']
#####using ploty 
    plotly.offline.plot({"data": [Bar(x=a[feature], y=a['Avg'])],"layout": Layout(title=title,xaxis=dict(title=xax),yaxis=dict(title=yax))})


## function to create keras model
def model_Creation():
	# creating model
    model = keras.Sequential()
    model.add(keras.layers.Dense(6, input_dim=6, kernel_initializer='normal', activation='relu'))
    model.add(keras.layers.Dense(6, input_dim=6, kernel_initializer='normal', activation='relu'))
    model.add(keras.layers.Dense(1, kernel_initializer='normal', activation='relu'))
	# Compile model
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.01), metrics=['mae'])
    return model





########################################################    Importing Data    ####################################
#loading the dataset
dirname = os.getcwd() 
filename = os.path.join(dirname, 'insurance.csv')
df=pd.read_csv(filename)
df.head()




#######################################################     Cleaning Data    ####################################
#cleaning the dataset
df.isnull().sum()

ploting_df=df.copy()
#label encoding
df.dtypes
lb=LabelEncoder()
df['sex']=lb.fit_transform(df['sex'])
df['smoker']=lb.fit_transform(df['smoker'])
df['region']=lb.fit_transform(df['region'])


#finding the correlation

correlation=df.corr()

#plotting the correlation
plotly.offline.plot({"data": [Heatmap(z=np.array(correlation),x=list(df), y=list(df))],"layout": Layout(title="Correlation")})





###################################################     Data visualizatoin    ######################################
# Age wise plot
plot_function('age',ploting_df,'Age wise Claim cost','Age','Average claim cost')

##Sex wise plot
plot_function('sex',ploting_df,'Sex wise claim cost','Gender','Average claim cost')

# Smoke wise plot
plot_function('smoker',ploting_df,'Somker wise claim cost','Smokes','Average claim cost')


#Region wise plot
plot_function('region',ploting_df,'region wise claim cost','region','Average claim cost')



#################################################       Model Building    ######################################
#normalizaing Charges attribute
y=df['charges']
max_y=max(y)
min_y=min(y)
y_changed=(y-min(y))/(max(y)-min(y))
df=df.drop('charges', axis=1)

#training set and testing set
X_train,X_test,Y_train,Y_test=train_test_split(df,y_changed, test_size=0.25)


# creating class for deep learning using pytorch 
class LinearRegression(nn.Module):
    def __init__(self,input_size,output_size,hidden_size):
       
        super(LinearRegression,self).__init__()
       
        #layes in deep learning model
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    # forward propagation
    def forward(self,x): 
        out = self.fc1(x)
        out = nn.functional.relu(self.fc1(x))
        out = self.fc2(out)
        out = nn.functional.relu(self.fc2(x))
        out = self.fc3(out)
        return out
    
#Defining the model
model = LinearRegression(6,1,6) 
#Definig loss function
mse = nn.MSELoss()
#Defining optimization function
optimizer = torch.optim.Adam(model.parameters(),lr = 0.01)

#Model Training
loss_data = []

for a in range(100):
        
  
    optimizer.zero_grad()
    #forward propagation
    results = model(torch.from_numpy(np.array(X_train)).float())
    loss = mse(results, torch.from_numpy(np.array([Y_train]).T).float())
    #backward propagation
    loss.backward()
    optimizer.step()
    # storing loss for plotting purpose
    loss_data.append(loss.data)
    #printing loss for each epoch
    print('epoch {}, loss {}'.format(a+1, loss.data))

#Model Testing
predicted = model(torch.from_numpy(np.array(X_test)).float().requires_grad_()).data.numpy()

##################################################################

#Creating keras model

mod=model_Creation()
mod_data=mod.fit(X_train.values,Y_train.values, epochs=100)

#printing mean squared error for pytorch and keras model
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

print("pytorch mean square error with testing data",np.power(np.power(np.mean((predicted-np.array(Y_test))),2),0.5))
print("keras mean square error with testing data",np.power(np.power(np.mean((mod.predict(X_test)-np.array(Y_test))),2),0.5))
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

loss_keras=mod_data.history['loss']



####plotiing loss for two models keras and pytorch

plotly.offline.plot({"data": [Scatter(x=list(range(150)), y=loss_data,line=Line(),name='Pytorch loss with trainig data'),Scatter(x=list(range(150)), y=loss_keras,line=Line(),name="keras loss with tarining data")],"layout": Layout(title="Loss for Keras model and Pytorch model vs epoch")})


