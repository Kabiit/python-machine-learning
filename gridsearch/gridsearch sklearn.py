#!/usr/bin/env python
# coding: utf-8

# we began by import the pandas and numpy libraries
# 
# we read the train.csv dataset and visualised it

# In[25]:


import pandas as pd
import  numpy as np
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("train.csv")
df


# we set the independent variable X and the target variable y

# In[17]:


X = df.drop("price_range", axis = 1)
y = df.price_range.values


# we create the Random forest model  and the Grid search.
# 
# we the various Gridsearch parameters such as n_estimators, Max_dept etc

# In[20]:


model = RandomForestClassifier(n_jobs=-1)
prem_gris = {
    "n_estimators":[100,200,300,400],
    "max_depth":[1,3,5,7],
    "criterion":['gini', "entropy"]
}
classifir = model_selection.GridSearchCV(
    estimator=model,
    param_grid= prem_gris,
    scoring='accuracy',
    verbose=10,
    n_jobs=1,
    cv=5
)


# We fit the model created and it trains for all the parameters set

# In[21]:


classifir.fit(X,y)


# The best scpore among all the possibel parameter combination is printed out by calling the .best_score_ method

# In[22]:


classifir.best_score_


# We proced and find out the parameter combination that gives the bess score

# In[28]:


classifir.best_estimator_.get_params


# In[ ]:

print("hello world of programming")


