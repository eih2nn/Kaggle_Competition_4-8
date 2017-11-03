
# coding: utf-8

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pandas as pd
import os #Import os package


# In[ ]:

### Read in file
#%%
os.chdir('/home/yingjie/Desktop')

test_ID = pd.DataFrame.from_csv('test_id.csv',index_col=None)
test = pd.DataFrame.from_csv('testing.csv',index_col=None)
train = pd.DataFrame.from_csv('training_weighted.csv',index_col=None)
valid = pd.DataFrame.from_csv('validation_weighted.csv',index_col=None)
targets = pd.DataFrame.from_csv('validation_target.csv',index_col=None)


# In[ ]:

#Make sure testing set was read in correctly
test.head()


# In[ ]:

#Make sure training set was read in correctly
train.head()


# In[ ]:

#Check zeros in training dataset
sum(train.target==0) 


# In[ ]:

''' Did not use this part of the code (although we attempted to run gini in python prior to changing tactics)
# Define unnormalized gini index and normalized gini index
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

def normalized_gini_index(ground_truth, predicted_probabilities):
    model_gini_index = unnormalized_gini_index(ground_truth, predicted_probabilities)
    optimal_gini_index = unnormalized_gini_index(ground_truth, ground_truth)
    return(model_gini_index / optimal_gini_index)

#%%
'''


# In[ ]:

len(train) #86776
len(valid) #21694


# In[ ]:

# PRE-PROCESSING - SUBSET MODEL
# Training set preparation
train_RF_noT = train.loc[:,train.columns !="target"] # training set predictors
train_RF_T = train[["target"]]

# Validation set preparation (for consistency)
valid_RF_noT = valid
valid_RF_T = targets


# In[ ]:

# Random Forest - Takes a very long time to run
# IMPORTANT NOTE: we altered max_depth and max_features manually and tested with gini in R!

clf = RandomForestClassifier(max_depth=10, max_features=20, n_estimators=1000, random_state=0)
clf.fit(train_RF_noT,train_RF_T) # Fit training dataset
preds = (clf.predict(valid_RF_noT)) # Predict using validation set
preds


# In[ ]:

#For the max_depth=10, max_features=20, n_est=1000...
preds10 = pd.DataFrame(preds)
preds10.columns = ["target"]


# In[ ]:

preds10.head()


# In[ ]:

sum(preds10.target==0)
len(preds10)


# In[ ]:

preds10.to_csv("preds10.csv")


# In[ ]:

#Set parameters to whatever worked best (these produces our highest kaggle score...)
#NOTE: again, this was a semi-iterative process, as we made several kaggle submission for RF

clf = RandomForestClassifier(max_depth=100, max_features=30, n_estimators=1000, random_state=0)
clf.fit(train_RF_noT,train_RF_T)


# In[ ]:

#See feature importances
print(clf.feature_importances_)


# In[ ]:

#Predict on testing set
preds = (clf.predict(test))


# In[ ]:

#Place predictions into dataframe and label with "target"
predsfinal = pd.DataFrame(preds)
predsfinal.columns = ["target"]


# In[ ]:

#Check/view dataframe
predsfinal.head()


# In[ ]:

sum(predsfinal.target==0) # Check how many predicted values are zeros
len(predsfinal)


# In[ ]:

#Write out to csv
predsfinal.to_csv("finalpreds.csv")

