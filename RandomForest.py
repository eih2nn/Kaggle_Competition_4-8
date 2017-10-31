
# coding: utf-8

# In[1]:

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pandas as pd
import os #Import os package


# In[23]:

#%%
os.chdir('/Users/eihoman/Documents/GitHub/Kaggle_Competition_4-8/')

test = pd.DataFrame.from_csv('test.csv',index_col=None)
train = pd.DataFrame.from_csv('train2_sample.csv',index_col=None)


# In[24]:

test


# In[58]:

sum(train.target==0)


# In[27]:

#%%
test_noid = test.loc[: , test.columns !='id']


# In[28]:

train["ps_ind_01"][1].dtype


# In[106]:

#%%
clf = RandomForestClassifier(max_depth=100, max_features=30, n_estimators=1000, random_state=0)


# In[60]:

train["target"][1]


# In[108]:

#%%
Y = train[["target"]]


# In[109]:

x = train.loc[:,train.columns !="id"]


# In[84]:

#x = train[["ps_car_12","ps_car_13","ps_car_14","ps_reg_03","ps_reg_02","ps_reg_01","ps_ind_17_bin",
        #   "ps_ind_16_bin","ps_ind_15","ps_ind_08_bin","ps_ind_07_bin","ps_ind_03","ps_ind_01"]]


# In[110]:

clf.fit(x,Y)
 


# In[111]:

#%%
print(clf.feature_importances_)


# In[89]:

#x2 = test_noid[["ps_car_12","ps_car_13","ps_car_14","ps_reg_03","ps_reg_02","ps_reg_01","ps_ind_17_bin",
 #          "ps_ind_16_bin","ps_ind_15","ps_ind_08_bin","ps_ind_07_bin","ps_ind_03","ps_ind_01"]]


# In[112]:

x2 = test.loc[: , test.columns !='target']


# In[113]:

preds = (clf.predict(x2))


# In[114]:

preds


# In[116]:

testID = test.loc[: , test.columns=='id']


# In[117]:

testID2 = testID


# In[118]:

testID2["target"]=preds


# In[119]:

testID2


# In[120]:

sum(testID2.target==1)


# In[121]:

sum(testID2.target==0)


# In[98]:

32765/860051


# In[100]:

testID3 = testID2.iloc[:,2-3]


# In[101]:

testID3


# In[102]:

testID3.to_csv("RFrun2.csv")


# In[ ]:



