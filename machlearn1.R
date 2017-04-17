library(caret)
library(kernlab)
data(spam)

inTrain <- createDataPartition(y=spam$type,p=0.75, list = FALSE)

training <- spam[inTrain,]
testing <- spam[-inTrain,]

#Fit a Model: SPAM example

set.seed(32343)
modelFit <- train(type ~., data = training, method = "glm")

#predict using model on different data
predictions <- predict(modelFit, newdata = testing)
predictions 

#validate with confusionMatrix
confusionMatrix(predictions, testing$type)


#SPAM Example: K-fold
#list = TRUE returns list element
# returnTrain = FALSE returns test set
set.seed(32323)
folds <- createFolds(y = spam$type, k = 10, list = TRUE, returnTrain = TRUE)

#Resampling or Bootstrapping
set.seed(32323)
folds <- createResample(y = spam$type, times = 10, list = TRUE)
folds[[1]][1:10]

#time slices
set.seed(32323)
tme <- 1:1000
folds <- createTimeSlices(y = tme, initialWindow = 20,
                          horizon = 10)
folds$train[[2]]

############################
# TRAINING OPTIONS
############################# ------------------------------------------------ ###

library(caret)
library(kernlab)
data(spam)

inTrain <- createDataPartition(y = spam$type, p = 0.75, list = FALSE)

training <- spam[inTrain, ]
testing <- spam[-inTrain, ]

modelFit <- train(type ~., data = training, method = "glm")

##################################
# PLOTTING PREDICTORS
##################################

library(ISLR)
library(ggplot2)
library(caret)
library(dplyr)
library(Hmisc)
library(GGally)

data(Wage)

inTrain <- createDataPartition(y = Wage$wage, p = 0.7, list = FALSE)
training <- Wage[inTrain, ]
testing <- Wage[-inTrain, ]

featurePlot(x = training[,c('age','education','jobclass')],
            y = training$wage,
            plot = "pairs")

ggpairs(training[,c('age','education','jobclass')],
        lower = list(continuous = "smooth"),
        upper = list(continuous = "cor"),
        aes(color="black")) + 
    theme(axis.text.x = element_text(size=5),
          axis.text.y = element_text(size=5),
          panel.grid = element_line(color="white"))

qplot(age, wage, color = education, data = training) +
    geom_smooth(method = 'lm', formula = y ~ x)

# -- percentiles -- #

cutWage <- cut2(training$wage, g = 3)

qplot(cutWage, age, data = training, fill = cutWage, geom = c("boxplot","jitter"))

table(cutWage, training$jobclass)

prop.table(table(cutWage, training$jobclass), 1)


########################
#  Basic Preprocessing
######################## ----------------------------------------------------- #

smallSpam <- spam[,c(34, 32)]
prComp <- prcomp(smallSpam)

### !!!!! ####  
#####   HIGH CORRELATION ANALYSIS with PCA
#_________________________________________

library(caret)
library(kernlab)
data(spam)

inTrain <- createDataPartition(y = spam$type, p = 0.75, list = F)

training <- spam[inTrain,]
testing <- spam[-inTrain,]

M <- abs(cor(training[,-58]))

diag(M) <- 0
which(M > 0.8, arr.ind = T)

names(spam)[c(34,32)]
plot(spam[,34],spam[,32])


smallSpam <- spam[,c(34, 32)]
prComp <- prcomp(smallSpam)
plot(prComp$x[,1],prComp$x[,2])

prComp$rotation

#PCA on Spam data
typeColor <- ((spam$type == "spam") * 1 + 1)

#_______________###----#
#PCA with CARET
###----#

preProc <- preProcess(log10(spam[,-58]+1), method = "pca", pcaComp = 2)
spamPC <- predict(preProc, log10(spam[,-58]+1))
plot(spamPC[,1], spamPC[,2], col = typeColor)

#Preprocessing with PCA
preProc <- preProcess(log10(training[,-58]+1), method = "pca", pcaComp = 2)
trainPC <- predict(preProc, log10(training[,-58]+1))
modelFit <- train(x = trainPC, y = training$type,method="glm")
testPC <- predict(preProc, log10(testing[,-58]+1))

confusionMatrix(testing$type, predict(modelFit, testPC))

#---------------- ######### ------ #######______________*______________*______________*
# Prediction with Multivariate Regression ______________*______________*______________*
#---------------- ######### ------ #######______________*______________*______________*

library(ISLR)
library(ggplot2)
library(caret)

data(Wage)
View(Wage)
Wage <- subset(Wage, select = -c(logwage))

inTrain <- createDataPartition(y = Wage$wage, p = 0.7, list = F)
training <- Wage[inTrain,]
testing <- Wage[-inTrain,]




