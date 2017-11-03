
# coding: utf-8


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pandas as pd
import os #Import os package


### Read in file
#%%
os.chdir('/home/yingjie/Desktop')

test = pd.DataFrame.from_csv('test.csv',index_col=None)
train = pd.DataFrame.from_csv('train.csv',index_col=None)

test.head()

train.head()

sum(train.target==0) # Check zeros in training dataset


# PRE-PROCESSING - FULL MODEL
# Remove id from both training and testing dataset
test_noid = test.loc[: , test.columns !='id']
train_noid = train.loc[: , train.columns !='id']

Y = train_noid[["target"]] #training set target
x = train_noid.loc[:,train_noid.columns !="target"] #training set predictors
x2 = test_noid.loc[: , test_noid.columns !='target'] # testing set predictors


# PRE-PROCESSING - SUBSET MODEL
# Training set and validation set - set up
train_RF = pd.DataFrame.sample(train_noid,frac=.8) # 20/80 split
valid_RF = train_noid[~train_noid.index.isin(train_RF.index)]

# Training set preparation
train_RF_noT = train_RF.loc[:,train_RF.columns !="target"] # training set predictors
train_RF_T = train_RF[["target"]]

# Validation set preparation
valid_RF_noT = valid_RF.loc[:,valid_RF.columns !="target"] # validation set predictors
valid_RF_T = valid_RF[["target"]]


''' Did not use this part of the code
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

len(train_RF) #476170
len(valid_RF) #119042


# Random Forest - Takes a very long time to run
#%%
clf = RandomForestClassifier(max_depth=200, max_features=20, n_estimators=1000, random_state=0)
clf.fit(train_RF_noT,train_RF_T) # Fit training dataset
preds = (clf.predict(valid_RF_noT)) # Predict using validation set
preds
#%%

predsfinal = pd.DataFrame(preds)
predsfinal.columns = ["target"]
predsfinal.head()
sum(predsfinal.target==0) # Check how many predicted values are zeros
len(predsfinal)

# Write out files
predsfinal.to_csv("finalpreds.csv") # Write out predicted file to be used in gini index function in R
valid_RF_T.to_csv("validtarget.csv") # Write out target values in validation set to be used in gini index function in R

