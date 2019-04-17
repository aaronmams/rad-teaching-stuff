rm(list=ls())
library(dplyr)
library(data.table)
library(ggplot2)
library(caret)
library(nnet)
library(lubridate)
library(gbm)

dat <- tbl_df(read.csv('caret-example/data.csv')) %>%
         mutate(utc_date=as.POSIXct(utc_date,format="%Y-%m-%d %H:%M:%S",tz="UTC"),
                local_time=as.POSIXct(local_time,format="%Y-%m-%d %H:%M:%S"))

###############################################################################################################
###############################################################################################################
###############################################################################################################
# Elemetary Data Preprocessing:

# just gonna make some of the continuous variables categorical
dat <- dat %>% ungroup() %>% mutate(length=as.factor(as.numeric(cut(length,5))),
                                    hour=as.factor(as.numeric(cut(hour,4))))


# make the data set smaller so we can deal with it
df <- dat %>% filter(year(local_time)==2014 & month(local_time)==10) %>% 
  select(foraging,speed,depth,length,hour,angle) %>%
  mutate(y=foraging) %>% select(-foraging)


###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################
###############################################################################################################

# A proper CV experiment with ANNs

#--------------------------------------------------------------------
#create the training and testing data
index <- createDataPartition(df$y,p=0.8,list=F,times=1)
df.train <- df[index,] %>% mutate(y=as.factor(y))
levels(df.train$y) <- c('notfishing','fishing') 
df.test <- df[-index,] %>% mutate(y=as.factor(y))
levels(df.test$y) <- c('notfishing','fishing')  
#--------------------------------------------------------------------

#--------------------------------------------------------------------
# set up control parameters for the neural network
fitControl <- trainControl(method = "cv", 
                           number = 5, 
                           classProbs = TRUE, 
                           summaryFunction = twoClassSummary)

nnetGrid <-  expand.grid(size = seq(from = 1, to = 15, by = 1),
                         decay = seq(from = 0.1, to = 0.5, by = 0.1))

# set up control parameters for the GBM
fitControl.GBM <- trainControl(method = "cv", 
                               number = 5,
                               classProbs = TRUE,
                               summaryFunction = twoClassSummary)

# pre process the data....only the continuous variables
preProcValues <- preProcess(df.train, method = c("range"))
trainTransformed <- predict(preProcValues, df.train)
testTransformed <- predict(preProcValues, df.test)
#----------------------------------------------------------------------

#----------------------------------------------------------------------
# Train the model using 5-fold CV to choose the best model
t <- Sys.time()
nnetFit <- train(y ~ ., 
                 #data = trainTransformed,
                 data = df.train,
                 method = "nnet",
                 metric = "ROC",
                 trControl = fitControl,
                 tuneGrid = nnetGrid,
                 verbose = FALSE)

# tree-based methods generally don't need input scaling
gbmFit <- train(y ~ ., 
                #data = trainTransformed,
                data = df.train,
                method = "gbm",
                metric = "ROC",
                trControl = fitControl.GBM,
                verbose = FALSE)
Sys.time() - t
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
# make predictions
NNpredictions <- predict(nnetFit,df.test)
confmat <- confusionMatrix(NNpredictions,df.test$y)

GBMpredictions <- predict(gbmFit,df.test)
#GBMpredictions <- predict(gbmFit,testTransformed)
GBMmat <- confusionMatrix(GBMpredictions,df.test$y)
#GBMmat <- confusionMatrix(GBMpredictions,testTransformed$y)

#f1 score for the ANN
p <- confmat$table[2,2]/(confmat$table[2,2] + confmat$table[2,1])
r <- confmat$table[2,2]/(confmat$table[2,2] + confmat$table[1,2])
(2*p*r)/(p+r)

#f1 score for the GBM
p <- GBMmat$table[2,2]/(GBMmat$table[2,2] + GBMmat$table[2,1])
r <- GBMmat$table[2,2]/(GBMmat$table[2,2] + GBMmat$table[1,2])
(2*p*r)/(p+r)

plotnet(nnetFit)
#--------------------------------------------------------------------------
