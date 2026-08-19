#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import datetime


# In[2]:


pd.set_option('display.max_rows', 99999)
pd.set_option('display.max_columns', 500)
warnings.filterwarnings("ignore")


# In[3]:


df = pd.read_csv('epc_best20.csv')
df.shape


# In[4]:


objList = df.select_dtypes(include = "object").columns
# Label Encoding for object to numeric conversion
epc_encoded = df.copy()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for feat in objList:
    epc_encoded[feat] = le.fit_transform(df[feat].astype(str))

epc_encoded.info()


# In[5]:


from sklearn.preprocessing import LabelEncoder
# Split data into features and labels
X = epc_encoded.drop(["Current energy efficiency rating band", "Unnamed: 0"], axis=1)
y = epc_encoded["Current energy efficiency rating band"]


# In[6]:


from sklearn.model_selection import train_test_split
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[7]:


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# In[8]:


dic20 ={2 : 'C',
        3 : 'D',
        1 : 'B',
        4 : 'E',
        5 : 'F',
        0 : 'A',
        6 : 'G' }


# In[9]:


dic19 = {1:1 ,
         2:0 }


# In[10]:


dic18 = {1:2 ,
         2:9 ,
         3:10 ,
         4:7 ,
         5:8 ,
         6:6 ,
         7:0 ,
         8:5 ,
         9:4 ,
         10:3 ,
         11:1 }


# In[11]:


dic17 = {1:1 ,
         2:0 }


# In[12]:


dic16 = {1:1 ,
         2:0 }


# In[13]:


dic15 = {1:0 ,
         2:1 ,
         3:4 ,
         4:5 ,
         5:3 ,
         6:2 }


# In[14]:


dic14 = {1:1 ,
         2:4 ,
         3:0 ,
         4:3 ,
         5:2 }


# In[15]:


dic13 = {1:1 ,
         2:4 ,
         3:0 ,
         4:2 ,
         5:3 }


# In[16]:


dic12 = {1:1 ,
         2:8 ,
         3:5 ,
         4:0 ,
         5:7 ,
         6:6 ,
         7:3 ,
         8:4 ,
         9:2 }


# In[17]:


dic9 = {1:2 ,
         2:1 ,
         3:0 ,
         4:3 ,
         5:4 }


# In[18]:


dic8 = {1:4 ,
        2:8 ,
        3:7 ,
        4:5 ,
        5:0 ,
        6:2 ,
        7:6 ,
        8:1 ,
        9:3 }


# In[19]:


dic7 = {1:0 ,
        2:2 ,
        3:1 ,
        4:3 }


# In[20]:


dic6 = {1:2 ,
        2:1 ,
        3:0 }


# In[21]:


dic3 = {1:10 ,
        2:0 ,
        3:1 ,
        4:2 ,
        5:3 ,
        6:4 ,
        7:5 ,
        8:6 ,
        9:7 ,
        10:8 ,
        11:9 }


# In[24]:


print(" Welcome to our Energy Efficiency Prediction chatbot\n",
      "===================================================\n",  
      "Please choose one of the choices:\n",
        "1- Calculate energy efficiency rating band\n",
        "2- exit from chat")

c = int(input())

number_list = ['The total floor area of the buildings m2',
               'The total current energy costs over 3 years (SR)',
               'The current hot water costs over 3 years (SR)',
               'The age band when building part constructed\n 1- before 1919\n 2- 1919-1929\n 3- 1930-1949\n 4- 1950-1964\n '
               '5- 1965-1975\n 6- 1976-1983\n 7- 1984-1991\n 8- 1992-1998\n 9- 1999-2002\n 10- 2003-2007\n 11- 2008 onwards\n',
               'The average height of the lowest storey of the dwelling (Units: m)',
               'The percentage of low energy lighting ',
               'The type of mechanical ventilation\n 1- natural\n 2- mechanical, supply and extract\n 3- mechanical, extract only',
               'The tenure type of the property\n 1- owner-occupied\n 2- rented (social)\n 3- rented (private)\n 4- unknown',
               'The type of transaction or purpose that triggered EPC assessment\n 1- marketed sale\n 2- rental\n '
               '3- none of the above\n 4- new dwelling\n 5- ECO assessment\n 6- RHI application\n 7- non marketed sale\n '
               '8- FiT application\n 9- assessment for green deal',
               'The type of property:\n 1- House\n 2- Flat\n 3- Bungalow\n 4- Maisonette\n 5- Park home',
               'If the wall is insulated 1, otherwise 0',
               'If the roof is insulated 1, otherwise 0', 
               'The type of material of wall:\n 1- cavity wall\n 2- timber frame\n 3- sandstone or limestone\n 4- Unknown\n 5- system built\n 6- solid brick\n 7- granite or whinstone\n 8- park home wall\n 9- cob',
               'The type of material of roof:\n 1- Pitched\n 2- Unknown\n 3- Flat\n 4- Roof rooms\n 5- Thatched',
               'The type of floor construction:\n 1- Suspended\n 2- Unknown\n 3- Solid\n 4- To unheated space\n 5- To external air ',
               'The glazing type of window:\n 1- double glazed\n 2- high performance glazing\n 3- single glazed\n 4- triple glazed\n 5- secondary glazing\n 6- multiple glazing', 
               'Does the main heat system uses mains gas?\n 1- yes\n 2- no',
               'Does the main heat control system includes a programmer?\n 1- yes\n 2- no  ',
               'The percentage of low lighting\n 1- 0%\n 2- 10%\n 3- 20%\n 4- 30%\n 5- 40%\n 6- 50%\n 7- 60%\n 8- 70%\n 9- 80%\n 10- 90% \n 11- 100%',
               'Does the second heat system is room heaters? \n 1- yes\n 2- no']

if c == 1:
    print()
    print("Before you answer the question please check the url documentation below to help you answer the questions")
    print("https://drive.google.com/file/d/1wKlgjZtFxceuaHkzFsmxWZU41WQXXjmk/view?usp=sharing")
    print()
    features = []
    for i in range(20):       
        print("Enter your answer:\n", number_list[i], )
        if i == 0:
            value = float(input())          
            features.append(value)
            print()
            continue
        elif i == 1:
            value = float(input())
            value = value * 0.22
            features.append(value)
            print()
            continue
        elif i == 2:
            value = float(input())
            value = value * 0.22
            features.append(value)
            print()
            continue
        elif i == 3:
            value = int(input())
            features.append(dic3[value])
            print()
            continue
        elif i == 4:
            value = float(input())
            features.append(value)
            print()
            continue
        elif i == 5:
            value = int(input())
            features.append(value)
            print()
            continue
        elif i == 6:    
            value = int(input())
            features.append(dic6[value])
            print()
            continue
        elif i == 7:    
            value = int(input())
            features.append(dic7[value])
            print()
            continue
        elif i == 8:    
            value = int(input())
            features.append(dic8[value])
            print()
            continue
        elif i == 9:    
            value = float(input())
            features.append(dic9[value])
            print()
            continue
        elif i == 10:
            value = float(input())
            features.append(value)
            print()
            continue
        elif i == 11:
            value = float(input())
            features.append(value)
            print()
            continue
        elif i == 12:    
            value = int(input())
            features.append(dic12[value])
            print()
            continue
        elif i == 13:    
            value = int(input())
            features.append(dic13[value])
            print()
            continue
        elif i == 14:    
            value = int(input())
            features.append(dic14[value])
            print()
            continue
        elif i == 15:    
            value = int(input())
            features.append(dic15[value])
            print()
            continue
        elif i == 16:    
            value = int(input())
            features.append(bool(dic16[value]))
            print()
            continue
        elif i == 17:    
            value = int(input())
            features.append(bool(dic17[value]))
            print()
            continue
        elif i == 18:    
            value = int(input())
            features.append(dic18[value])
            print()
            continue
        elif i == 19:    
            value = int(input())
            features.append(bool(dic19[value]))
            print()
            continue
    f = np.array([features])
    # using inputs to predict the output
    prediction = model.predict(f)
    print("\nThe prediction of Energy Efficiency Rating Band is: ",dic20[prediction[0]])
else:
    print("\nThank you and see you soon...")


# In[ ]:




