#TEAM COMPETITION 4-8
#Elizabeth Homan
#Yingjie Liu
#Ali Zaidi

#Load the core tidyverse packages (ggplot2, tibble, tidyr, readr, purrr, and dplyr), 
#as well as tm and MASS
library(tidyverse)

sample_sumbission <- read_csv("sample_submission.csv") #Read in the comma separated value data file

#Read in files:
train <- read_csv("train.csv") #Read in the comma separated value data file for training the model
test <- read_csv("test.csv") #Read in the csv data file for testing the model

#Check to see whether categorical columns are factors or other
class(train['ps_ind_02_cat'][[1]])
#integer

#Select all column names that end with cat (denoting categorical) or bin (denoting binary)
factors <- colnames(train %>% dplyr:: select(ends_with("cat"))) 
factors <- c(factors, colnames(train %>% dplyr:: select(ends_with("bin"))))
factors #Check to see if this works
train[factors] = lapply(train[factors], factor) #Change all integers in these columns to factors
test[factors] = lapply(test[factors], factor) #Repeat for the test set

#Similarly, change all remaining columns with integers to the numeric class (using lapply function):
train[ , (!names(train) %in% factors)] = lapply(train[ , (!names(train) %in% factors)], as.numeric)
test[ , (!names(test) %in% factors)] = lapply(test[ , (!names(test) %in% factors)], as.numeric)

#Recheck to see if this worked
class(train['ps_ind_02_cat'][[1]])
#factor
class(train['ps_ind_03'][[1]])
#numeric

