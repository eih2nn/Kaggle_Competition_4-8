#TEAM COMPETITION 4-8
#Elizabeth Homan
#Yingjie Liu
#Ali Zaidi

#Load the core tidyverse packages (ggplot2, tibble, tidyr, readr, purrr, and dplyr), 
#as well as tm and MASS
library(tidyverse)

#Read in files:
sample_submission <- read_csv("sample_submission.csv") #Read in the comma separated value data file
train <- read_csv("train.csv") #Read in the comma separated value data file for training the model
test <- read_csv("test.csv") #Read in the csv data file for testing the model


##### DATA PREPARATION #####

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

#Put NaNs in for all observations noted as -1 (as these are sentinel values)
train[train == -1] <- NA
test[test == -1] <- NA

#Create mode function
Mode <- function(x, na.rm) {
  xtab <- table(x)
  xmode <- names(which(xtab == max(xtab)))
  if (length(xmode) > 1) xmode <- ">1 mode"
  return(xmode)
}

#Replace all NA values in "factor" columns with the mode of that column
for (var in 1:ncol(train)) {  
  if (lapply((train[,var]), class)=="factor") {
    train[is.na(train[,var]),var] <- Mode(train[,var], na.rm = TRUE)
  }
}

#Repeat for test set
for (var in 1:ncol(test)) {  
  if (lapply((test[,var]), class)=="factor") {
    test[is.na(test[,var]),var] <- Mode(test[,var], na.rm = TRUE)
  }
}

#Replace all NA values in "numeric" columns with the mean of that column 
for (var in 1:ncol(train)) {
  if (lapply((train[,var]), class)=="numeric") {
    train[is.na(train[,var]),var] <- sapply(train[,var], mean, na.rm=TRUE)
  }
}

#Repeat for test set
for (var in 1:ncol(test)) {
  if (lapply((test[,var]), class)=="numeric") {
    test[is.na(test[,var]),var] <- sapply(test[,var], mean, na.rm=TRUE)
  }
}

##### PARAMETRIC APPROACH - BASIC LINEAR MODEL #####

train2 <- train[ , (!names(train) %in% 'id')]

train2.lm <- lm(target~.,data=train2)

#summary(train2.lm) -- used this to select out parameters with significance...

train2.lm2 <- lm(target~ps_car_12+ps_car_13+ps_car_14+ps_car_11_cat+
                  ps_car_09_cat+ps_car_07_cat+ps_car_04_cat+
                  ps_car_01_cat+ps_reg_03+ps_reg_02+
                  ps_reg_01+ps_ind_17_bin+ps_ind_16_bin+ps_ind_15+
                  ps_ind_08_bin+ps_ind_07_bin+ps_ind_05_cat+ps_ind_04_cat+
                  ps_ind_03+ps_ind_02_cat+
                  ps_ind_01, data=train2)

summary(train2.lm2)

#Create predictions for test set
#Use the predict function to apply the above linear model to the test data
mypreds.lm <- data.frame(predict(train2.lm2, newdata = test))  #Put these values into a dataframe

colnames(mypreds.lm)[1] <- "target" #Assign the column the appropriate name

#Make sure all target values are within desired range
mypreds.lm$target[mypreds.lm$target<0] <- 0
mypreds.lm$target[mypreds.lm$target>1] <- 1

mypreds.lm["id"] <- sample_submission['id'] #Add the id column to the newest dataframe
mypreds.lm <- mypreds.lm[,c(2,1)] #Switch the columns so that ID is the first column

write.table(mypreds.lm, file = "mypreds_lm1.csv", row.names=F, sep=",") #Write out to a csv


##### SECOND PARAMETRIC APPROACH - NON-LINEAR MODEL #####

# Polynomial
cv.error.10 =rep(0,10)
for (i in 1:10){fitted = lm(target~ps_car_12+ps_car_13+ps_car_14+ps_car_11_cat+
                              ps_car_09_cat+ps_car_07_cat+ps_car_04_cat+
                              ps_car_01_cat+poly(ps_reg_03,i)+ps_reg_02+
                              ps_reg_01+ps_ind_17_bin+ps_ind_16_bin+ps_ind_15+
                              ps_ind_08_bin+ps_ind_07_bin+ps_ind_05_cat+ps_ind_04_cat+
                              ps_ind_03+ps_ind_02_cat+
                              ps_ind_01, data=train2)
cv.error.10[i] = cv.glm(train2,fitted,K=10)$delta[1]
}

cv.error.10 # i =3

mypreds.lm2 <- data.frame(predict(train2.lm3, newdata = test))  #Put these values into a dataframe

colnames(mypreds.lm2)[1] <- "target" #Assign the column the appropriate name

#Make sure all target values are within desired range
mypreds.lm2$target[mypreds.lm2$target<0] <- 0
mypreds.lm2$target[mypreds.lm2$target>1] <- 1
mypreds.lm2["id"] <- sample_submission['id'] #Add the id column to the newest dataframe
mypreds.lm2 <- mypreds.lm2[,c(2,1)] #Switch the columns so that ID is the first column

write.table(mypreds.lm2, file = "mypreds_lm2.csv", row.names=F, sep=",") #Write out to a csv

##### NON-PARAMETRIC APPROACH - SPLINE MODEL #####

# Spline
library(splines)
train2.lm4 <- lm(target~bs(ps_car_12, knots = c(0,0.8,1.2))+ns(ps_car_13,df=3)+ns(ps_car_14, df=3)+ps_car_11_cat+
                   ps_car_09_cat+ps_car_07_cat+ps_car_04_cat+
                   ps_car_01_cat+poly(ps_reg_03,3)+poly(ps_reg_02,2)+
                   ps_reg_01+ps_ind_17_bin+ps_ind_16_bin+ps_ind_15+
                   ps_ind_08_bin+ps_ind_07_bin+ps_ind_05_cat+ps_ind_04_cat+
                   ps_ind_03+ps_ind_02_cat+
                   ps_ind_01, data=train2)
summary(train2.lm4)
mypreds.lm3 <- data.frame(predict(train2.lm4, newdata = test))  #Put these values into a dataframe

colnames(mypreds.lm3)[1] <- "target" #Assign the column the appropriate name

#Make sure all target values are within desired range
mypreds.lm3$target[mypreds.lm3$target<0] <- 0
mypreds.lm3$target[mypreds.lm3$target>1] <- 1
mypreds.lm3["id"] <- sample_submission['id'] #Add the id column to the newest dataframe
mypreds.lm3 <- mypreds.lm3[,c(2,1)] #Switch the columns so that ID is the first column

write.table(mypreds.lm3, file = "mypreds_lm3.csv", row.names=F, sep=",") #Write out to a csv
