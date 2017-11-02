
# coding: utf-8

# In[1]:

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pandas as pd
import os #Import os package


# In[2]:

#%%
os.chdir('/Users/eihoman/Documents/GitHub/Kaggle_Competition_4-8/')

test = pd.DataFrame.from_csv('test.csv',index_col=None)
train = pd.DataFrame.from_csv('train2_sample.csv',index_col=None)


# In[3]:

test.head()


# In[4]:

sum(train.target==0)


# In[5]:

#%%
test_noid = test.loc[: , test.columns !='id']
train_noid = train.loc[: , train.columns !='id']


# In[6]:

train["ps_ind_01"][1].dtype


# In[7]:

list(range(1,10,1))


# In[34]:

def unnormalized_gini_index(ground_truth, predicted_probabilities):
    if (len(ground_truth) !=  len(predicted_probabilities)):
        stop("Actual and Predicted need to be equal lengths!")
    x = len(ground_truth)   
    gini_table = pd.DataFrame(index = list(range(1,x+1,1)), predicted_probabilities=predicted_probabilities, ground_truth=ground_truth)
    gini_table = gini.table[order(-gini_table.predicted_probabilities, gini_table.index), ]
    num_ground_truth_positivies = sum(gini_table.ground_truth)
    model_percentage_positives_accumulated = gini_table.ground_truth / num_ground_truth_positivies
    random_guess_percentage_positives_accumulated = 1 / nrow(gini_table)
    gini_sum = cumsum(model_percentage_positives_accumulated - random_guess_percentage_positives_accumulated)
    gini_index = sum(gini_sum) / nrow(gini_table) 
    return(gini_index)


# In[35]:

def normalized_gini_index(ground_truth, predicted_probabilities):
    model_gini_index = unnormalized_gini_index(ground_truth, predicted_probabilities)
    optimal_gini_index = unnormalized_gini_index(ground_truth, ground_truth)
    return(model_gini_index / optimal_gini_index)


# In[36]:

Y = train_noid[["target"]]
x = train_noid.loc[:,train_noid.columns !="target"]
x2 = test_noid.loc[: , test_noid.columns !='target']


# In[37]:

train_RF = pd.DataFrame.sample(train_noid,frac=.8)
train_RF_noT = train_RF.loc[:,train_RF.columns !="target"]
train_RF_T = train_RF[["target"]]
valid_RF = train_noid[~train_noid.index.isin(valid_RF.index)]
valid_RF_noT = valid_RF.loc[:,valid_RF.columns !="target"]
valid_RF_T = valid_RF[["target"]]


# In[67]:

len(train_RF)


# In[70]:

sum(train_RF.target==1)


# In[68]:

len(valid_RF)


# In[38]:

Y.head()


# In[85]:

#%%
clf = RandomForestClassifier(max_depth=200, max_features=20, n_estimators=1000, random_state=0)
clf.fit(train_RF_noT,train_RF_T)
preds = (clf.predict(valid_RF_noT))
preds


# In[56]:

preds10 = pd.DataFrame(preds)


# In[58]:

preds10.columns = ["target"]


# In[59]:

preds10.head()


# In[60]:

sum(preds10.target==0)


# In[61]:

len(preds10)


# In[62]:

preds10.to_csv("preds10.csv")


# In[65]:

1-(21555/21694)


# In[66]:

valid_RF_T.to_csv("validtarget.csv")


# In[ ]:




# In[ ]:




# In[ ]:




# In[71]:

#Set x == whatever worked best
clf = RandomForestClassifier(max_depth=10, max_features=20, n_estimators=1000, random_state=0)


# In[72]:

#%%
Y = train[["target"]]


# In[73]:

x = train.loc[:,train.columns !="id"]


# In[74]:

z = x.loc[:,x.columns !="target"]


# In[75]:

clf.fit(z,Y)


# In[76]:

#%%
print(clf.feature_importances_)


# In[77]:

x2 = test.loc[: , test.columns !='id']


# In[78]:

preds = (clf.predict(x2))


# In[80]:

predsfinal = pd.DataFrame(preds)


# In[81]:

predsfinal.columns = ["target"]


# In[82]:

predsfinal.head()


# In[84]:

predsfinal.to_csv("finalpreds.csv")


# In[ ]:



