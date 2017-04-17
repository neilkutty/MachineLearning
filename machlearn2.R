#Predicting with Trees
data(iris)
library(ggplot2)
library(caret)
names(iris)

table(iris$Species)

inTrain <- createDataPartition(y = iris$Species, p = 0.7, list = FALSE)

training <- iris[inTrain,]
testing <- iris[-inTrain,]

modFit <- train(Species ~ ., method = "rpart", data = training)
print(modFit$finalModel)

library(rattle)

suppressMessages(library(rattle))
library(rpart.plot)
fancyRpartPlot(modFit$finalModel)

#Bagging --------------------_-------------_--- #$#--------------------_------ #$#
# --------------------_-------------_--- #$#

# Bagging is most useful for nonlinear models
# Often used with trees - an extension is random forests
# Several models use bagging in caret's train function

library(ElemStatLearn)
data(ozone, package = "ElemStatLearn")
ozone <- ozone[order(ozone$ozone),]

library(dplyr)    
ozone <- arrange(ozone, ozone)

ll <- matrix(NA, nrow = 10, ncol = 155)



for(i in 1:nrow(ll)){
    ss <- sample(1:dim(ozone)[1], replace = T)
    ozone0 <- ozone[ss,]
    ozone0 <- ozone0[order(ozone0$ozone),]
    loess0 <- loess(temperature ~ ozone, data = ozone0, span = 0.2)
    ll[i,] <- predict(loess0, newdata = data.frame(ozone = 1:155))
}

plot(ozone$ozone,ozone$temperature,pch=19,cex=0.5)
for(i in 1:10){lines(1:155,ll[i,],col="grey",lwd=2)}
lines(1:155,apply(ll,2,mean),col="red",lwd=2)

# Bagging in caret
# 
# Some models perform bagging for you, in train function consider method options
# bagEarth
# treebag
# bagFDA

# Alternatively you can bag any model you choose using the bag function


# More bagging in caret

predictors = data.frame(ozone=ozone$ozone)
temperature = ozone$temperature
treebag <- bag(predictors, temperature, B = 10,
               bagControl = bagControl(fit = ctreeBag$fit,
                                       predict = ctreeBag$pred,
                                       aggregate = ctreeBag$aggregate))
# http://www.inside-r.org/packages/cran/caret/docs/nbBag

# Example of custom bagging (continued)

plot(ozone$ozone,temperature,col='lightgrey',pch=19)
points(ozone$ozone,predict(treebag$fits[[1]]$fit,predictors),pch=19,col="red")
points(ozone$ozone,predict(treebag,predictors),pch=19,col="blue")

# Parts of bagging
ctreeBag$fit

# Parts of bagging
ctreeBag$pred

# Parts of bagging
ctreeBag$aggregate

#Random Forests --------------------_-------------_--- #$#--------------------_------ #$#
# --------------------_-------------_--- #$#
data(iris)
library(ggplot2)
library(caret)
library(randomForest)
inTrain <- createDataPartition(y = iris$Species, p = 0.7, list = FALSE)

trRF <- iris[inTrain,]
teRF <- iris[-inTrain,]

fit <- train(Species ~ ., data = trRF, method = "rf", prox = T)

getTree(fit$finalModel, k = 2)

# - Class centers - #

irisP <- classCenter(trRF[,c(3,4)], trRF$Species, fit$finalModel$proximity)
irisP <- as.data.frame(irisP)

qplot(Petal.Width, Petal.Length, col = Species, data = trRF) +
    geom_point(aes(x = Petal.Width, y = Petal.Length, col = ), size = 4, shape = 3, data = irisP)

#Random Forests ex. 2


#Boosting --------------------_-------------_--- #$#--------------------_------ #$#
# --------------------_-------------_--- #$#
library(ISLR)
library(ggplot2)
library(caret)
library(dplyr)

data(Wage)
lWage <- select(Wage, -logwage)

l <- createDataPartition(y = lWage$wage, p = 0.7, list = F)
training <- lWage[l, ]
testing <- lWage[-l, ]

fit <- train(wage ~ ., method = "gbm", data = training, verbose = F)

qplot(predict(fit, testing), wage, data = testing)



#Model Based Prediction --------------------_-------------_--- #$#--------------------_------ #$#
# --------------------_-------------_--- #$#
data(iris)
library(ggplot2)

p <- createDataPartition(y = iris$Species, p = 0.7, list = F)
tr <- iris[p,]
te <- iris[-p,]

modlda <- train(Species~., data = tr, method='lda') 
modnb <- train(Species~., data = tr, method='nb')
plda = predict(modlda, te)
pnb = predict(modnb, te)

equalPs = (plda==pnb)
qplot(Petal.Width, Sepal.Width, colour=equalPs, data=te)
        
table(plda, pnb)
