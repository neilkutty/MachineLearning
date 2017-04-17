
rm(list = ls())

library(caret)
library(GGally)
library(rpart)
library(rpart.plot)
library(party)
library(RGtk2)
library(rattle)
library(xgboost)
library(formattable)
library(dplyr)
library(tidyr)
library(tibble)
library(ggthemes)

#Download train and test datasets
train <- read.csv('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv')
test <- read.csv('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv')

inTrain <- createDataPartition(train$classe, p=0.75, list = F)
training <- train[inTrain,]
testing <- train[-inTrain,]

#Find near zero variance predictors
nzvs <- nearZeroVar(training, saveMetrics = T)
nzvars <- nzvs[nzvs$nzv==T,0]

#Remove near zero variance predictors from train and test sets
smallTrain <- training[,!colnames(training) %in% rownames(nzvars)]
smallTest <- testing[,!colnames(testing) %in% rownames(nzvars)]

#tabulate high frequency of NA variables
x <- array()
for(i in 1:ncol(smallTrain)){
        x[i] <- sum(is.na(smallTrain[,i]))
}
print(x)
table(x)

#subset train and test sets by rule above eliminating NA columns
smallerTrain <- smallTrain[,colSums(is.na(smallTrain)) == 0]
smallerTest <- smallTest[,colSums(is.na(smallTest)) == 0]

#get rid of id number which is duplicate of row index
#remove 'classe' variable from test set
sTrain <- smallerTrain[,-c(1,2,3,4,5,6,7)]
sTest <- smallerTest[,-c(1,2,3,4,5,6,7)]

#*&*#0-----------------------####________####-------------------------
# Notes --> (2/16/17 ::: For XGBoost and possibly for feature plotting, may need to convert
#            to all numeric even for outcome variable. Also get rid of name, go back to ID number)
#*&*#0-----------------------####________####-------------------------
#_____________
# Filter Train
#-------------

fTrain <- sTrain[, grepl('belt|dumbell|arm', names(sTrain))]

# -- <Feature Plot> -- not currently working !!<<<<
predictors <- fTrain[,-c(1,4,58)]
outcome <- sTrain$classe

featurePlot(predictors, outcome, "strip")

ggpairs(sTrain[,-c(1,4)])

#______________
# Train Control
#--------------

#--------------#--------------#--------------#--------------#--------------
#--------------#--------------#--------------#--------------#--------------
# Decision Tree
#--------------#--------------#--------------#--------------#--------------
#--------------#--------------#--------------#--------------#--------------

tree <- rpart(classe ~ ., data = sTrain, method = 'class')
fancyRpartPlot(tree)
printcp(tree, digits = 3)

#Predict using tree
TreeFit <- predict(tree, sTest, type = 'class')
TreeResults <- confusionMatrix(TreeFit, sTest$classe)

#Display confusion matrix results
tcm <- as.data.frame(TreeResults$table)

ggplot(tcm, aes(Prediction, Reference)) + geom_tile(aes(fill=Freq)) +
    geom_text(aes(label=digits(Freq,0))) +
    theme_hc()+
    scale_fill_gradient(low = "white", high = "blue") +
    ggtitle(label = paste("Decision Tree Accuracy:",round(TreeResults$overall['Accuracy'],4)),
            subtitle = "Confusion Matrix Plot") +
    theme(plot.title = element_text(hjust = 0.5, size = 12),
          plot.subtitle = element_text(hjust = 0.5, size = 10))


#Tabulate results by class
TreebyClass <- as.data.frame(TreeResults$byClass)
plot(TreebyClass)

#Tree Accuracy
TreeResults$overall[1]

#Predict on Test set
TreeTest <- predict(tree, test, type = 'class')

?rpart


#--------------#--------------#--------------#--------------#--------------
#--------------#--------------#--------------#--------------#--------------
# Random Forest
#--------------#--------------#--------------#--------------#--------------
#--------------#--------------#--------------#--------------#--------------
set.seed(867)

forest <- train(classe ~ ., data = sTrain, method = 'rf')
forestFit <- predict(forest, sTest)
forestResults <- confusionMatrix(forestFit, sTest$classe)
forestResults$overall['Accuracy']


 
## plot of Stats by Class ----------------------------------##
rdf <- as.data.frame(forestResults$byClass)
rdf$ID<-seq.int(nrow(rdf))
s <- rownames_to_column(rdf, var = "Class")
rdf <- gather(s, Class)
colnames(rdf) <- c('Class', 'Measure', 'Value')

library(RColorBrewer)
cols <- rev(brewer.pal(8, 'Pastel2'))

ggplot(rdf, aes(Class, Measure))+
    geom_tile(aes(fill=Value))+
    geom_text(aes(label=digits(Value,3)))+
    theme_solarized()+
    scale_fill_gradientn(colors = cols)
#### -------------------------------------------------------- ###
#>
#>
#>
## plot of Confusion Matrix  ----------------------------------##
fcm <- as.data.frame(forestResults$table)
ggplot(fcm, aes(Prediction, Reference)) + geom_tile(aes(fill=Freq)) +
    geom_text(aes(label=digits(Freq,0))) +
    theme_hc()+
    scale_fill_gradient(low = "white", high = "green") +
    ggtitle(label = paste("Random Forest Accuracy:",round(forestResults$overall['Accuracy'],4)),
            subtitle = "Confusion Matrix Plot") +
    theme(plot.title = element_text(hjust = 0.5, size = 12),
          plot.subtitle = element_text(hjust = 0.5, size = 10))
    #geom_text(aes(x = 2.5, y = 5.5,
    #              label = paste("Accuracy:",round(forestResults$overall['Accuracy'],4))))
    #scale_fill_gradientn(colors = cols)


#--------------------------------------  x -------------------------------------------------------#
#--------------#--------------#--------------#--------------#--------------#--------------
##--------------#--------------#--------------#--------------#--------------
###--------------#--------------#--------------#--------------
#####--------------#--------------#--------------
######--------------#--------------
########--------------
#--------------
#--------------#--------------#--------------#--------------#--------------#--------------
##--------------#--------------#--------------#--------------#--------------
###--------------#--------------#--------------#--------------
#####--------------#--------------#--------------
######--------------#--------------
########--------------
#-#--------------#--------------#--------------#--------------#--------------#--------------
##--------------#--------------#--------------#--------------#--------------
###--------------#--------------#--------------#--------------
#####--------------#--------------#--------------
######--------------#--------------
########--------------
#-#--------------#--------------#--------------#--------------#--------------#--------------
##--------------#--------------#--------------#--------------#--------------
###--------------#--------------#--------------#--------------
#####--------------#--------------#--------------
######--------------#--------------
########--------------
#-
# Extreme Gradient Boosting
#--------------
###--------------#--------------#--------------#--------------
#####--------------#--------------#--------------
######--------------#--------------
########--------------
#--------------


set.seed(867)

#Using train()
xgb_trcontrol_1 = trainControl(
    method = "repeatedcv",
    number = 5,
    repeats = 1)

xgb <- train(classe ~ ., data = sTrain, method = 'gbm', trControl = xgb_trcontrol_1, verbose = T)

xgb$finalModel

xgbResults <- predict(xgb, sTest)
confusionMatrix(xgbResults, sTest$classe)


#Using xgboost()  
mTrain <- as.matrix(sTrain[,-c(1,4,58)])
mTest <- as.matrix(sTest[,-c(1,4,58)])

mode(mTrain) = 'numeric'
mode(mTest) = 'numeric'

mTrain.label <- sTrain$classe
mTest.label <- sTest$classe

levels(mTrain.label) <- 1:length(levels(sTrain$classe))
levels(mTest.label) <- 1:length(levels(sTrain$classe))

xtrain <- cbind(mTrain,mTrain.label)
xtest <- cbind(mTest,mTest.label)

xgb2 <- xgboost(data = mTrain, label = mTrain.label,
               eta = 0.1,
               max_depth = 15, 
               nround=50, 
               subsample = 0.5,
               colsample_bytree = 0.5,
               seed = 1,
               eval_metric = "merror",
               objective = "multi:softprob",
               num_class = 12,
               nthread = 3)

xgbResults <- predict(xgb, mTest)

confusionMatrix(xgbResults, mTest.label)

xgbResults
