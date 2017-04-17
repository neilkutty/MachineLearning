

#Regularized Regression --------------------_-------------_--- #$#
## Idea:
## Fit a regression model
## Penalize (or shrink) large coefficients
##
## Pros: 1) Can help with bias/variance tradeoff 2) Can help with model selection
## Cons: 1) May be computationally demanding 2) Doesn't perform as well as RF or Boosting
# --------------------_-------------_--- #$#

####
# regression subset selection in the prostate dataset
library(ElemStatLearn)
data(prostate)

covnames <- names(prostate[-(9:10)])
y <- prostate$lpsa
x <- prostate[,covnames]

form <- as.formula(paste("lpsa~", paste(covnames, collapse="+"), sep=""))
summary(lm(form, data=prostate[prostate$train,]))

set.seed(1)
train.ind <- sample(nrow(prostate), ceiling(nrow(prostate))/2)
y.test <- prostate$lpsa[-train.ind]
x.test <- x[-train.ind,]

y <- prostate$lpsa[train.ind]
x <- x[train.ind,]

p <- length(covnames)
rss <- list()
for (i in 1:p) {
    cat(i)
    Index <- combn(p,i)
    
    rss[[i]] <- apply(Index, 2, function(is) {
        form <- as.formula(paste("y~", paste(covnames[is], collapse="+"), sep=""))
        isfit <- lm(form, data=x)
        yhat <- predict(isfit)
        train.rss <- sum((y - yhat)^2)
        
        yhat <- predict(isfit, newdata=x.test)
        test.rss <- sum((y.test - yhat)^2)
        c(train.rss, test.rss)
    })
}

png("Plots/selection-plots-01.png", height=432, width=432, pointsize=12)
plot(1:p, 1:p, type="n", ylim=range(unlist(rss)), xlim=c(0,p), xlab="number of predictors", ylab="residual sum of squares", main="Prostate cancer data")
for (i in 1:p) {
    points(rep(i-0.15, ncol(rss[[i]])), rss[[i]][1, ], col="blue")
    points(rep(i+0.15, ncol(rss[[i]])), rss[[i]][2, ], col="red")
}
minrss <- sapply(rss, function(x) min(x[1,]))
lines((1:p)-0.15, minrss, col="blue", lwd=1.7)
minrss <- sapply(rss, function(x) min(x[2,]))
lines((1:p)+0.15, minrss, col="red", lwd=1.7)
legend("topright", c("Train", "Test"), col=c("blue", "red"), pch=1)
dev.off()

##
# ridge regression on prostate dataset
library(MASS)
lambdas <- seq(0,50,len=10)
M <- length(lambdas)
train.rss <- rep(0,M)
test.rss <- rep(0,M)
betas <- matrix(0,ncol(x),M)
for(i in 1:M){
    Formula <-as.formula(paste("y~",paste(covnames,collapse="+"),sep=""))
    fit1 <- lm.ridge(Formula,data=x,lambda=lambdas[i])
    betas[,i] <- fit1$coef
    
    scaledX <- sweep(as.matrix(x),2,fit1$xm)
    scaledX <- sweep(scaledX,2,fit1$scale,"/")
    yhat <- scaledX%*%fit1$coef+fit1$ym
    train.rss[i] <- sum((y - yhat)^2)
    
    scaledX <- sweep(as.matrix(x.test),2,fit1$xm)
    scaledX <- sweep(scaledX,2,fit1$scale,"/")
    yhat <- scaledX%*%fit1$coef+fit1$ym
    test.rss[i] <- sum((y.test - yhat)^2)
}

png(file="Plots/selection-plots-02.png", width=432, height=432, pointsize=12) 
plot(lambdas,test.rss,type="l",col="red",lwd=2,ylab="RSS",ylim=range(train.rss,test.rss))
lines(lambdas,train.rss,col="blue",lwd=2,lty=2)
best.lambda <- lambdas[which.min(test.rss)]
abline(v=best.lambda+1/9)
legend(30,30,c("Train","Test"),col=c("blue","red"),lty=c(2,1))
dev.off()


png(file="Plots/selection-plots-03.png", width=432, height=432, pointsize=8) 
plot(lambdas,betas[1,],ylim=range(betas),type="n",ylab="Coefficients")
for(i in 1:ncol(x))
    lines(lambdas,betas[i,],type="b",lty=i,pch=as.character(i))
abline(h=0)
legend("topright",covnames,pch=as.character(1:8))
dev.off()


#######
# lasso
library(lars)
lasso.fit <- lars(as.matrix(x), y, type="lasso", trace=TRUE)
plot(lasso.fit, breaks=FALSE)


png(file="Plots/selection-plots-04.png", width=432, height=432, pointsize=8) 
plot(lasso.fit, breaks=FALSE)
legend("topleft", covnames, pch=8, lty=1:length(covnames), col=1:length(covnames))
dev.off()

# this plots the cross validation curve
png(file="Plots/selection-plots-05.png", width=432, height=432, pointsize=12) 
lasso.cv <- cv.lars(as.matrix(x), y, K=10, type="lasso", trace=TRUE)
dev.off()

#--------------------_-------------_--- #$#--------------------_-------------_--- #$#

# -- Combining Predictors --------------------_-------------_--- #$#

#--------------------_-------------_--- #$#--------------------_-------------_--- #$#
# You can combine classifiers by averaging/voting
# Combining classifiers improves accuracy
# Combining classifiers reduces interpretability
# Boosting, bagging, and random forests are variants on this theme
#
# ## Intuition
#  > Suppose we have 5 completely independent classifiers
#  If accuracy is 70% for each:
#       10 x (0.7)
# Notes:
# Even simple blending can be useful
# Typical model for binary/multiclass data
#   Build an odd number of models
#   Predict with each model
#   Predict the class by majority vote
# This can get dramatically more complicated
#   Simple blending in caret: caretEnsemble
#   See wikipedia "ensemble learning"
#--------------------_-------------_--- #$#--------------------_-------------_--- #$#
#______________________________________ >> ______________________________________ ###



library(ISLR); data(Wage); library(ggplot2); library(caret);
Wage <- subset(Wage,select=-c(logwage))

# Create a building data set and validation set
inBuild <- createDataPartition(y=Wage$wage,
                               p=0.7, list=FALSE)
validation <- Wage[-inBuild,]; buildData <- Wage[inBuild,]

inTrain <- createDataPartition(y=buildData$wage,
                               p=0.7, list=FALSE)
training <- buildData[inTrain,]; testing <- buildData[-inTrain,]

#Build two different models
mod1 <- train(wage ~.,method="glm",data=training)
mod2 <- train(wage ~.,method="rf",
              data=training, 
              trControl = trainControl(method="cv"),number=3)

#Predicting on the test set
pred1 <- predict(mod1,testing); pred2 <- predict(mod2,testing)
qplot(pred1,pred2,colour=wage,data=testing)

#Combining
predDF <- data.frame(pred1,pred2,wage=testing$wage)
combModFit <- train(wage ~.,method="gam",data=predDF)
combPred <- predict(combModFit,predDF)

#Testing errors
sqrt(sum((pred1-testing$wage)^2))
sqrt(sum((pred2-testing$wage)^2))
sqrt(sum((combPred-testing$wage)^2))

#Predict on validation data set
pred1V <- predict(mod1,validation); pred2V <- predict(mod2,validation)
predVDF <- data.frame(pred1=pred1V,pred2=pred2V)
combPredV <- predict(combModFit,predVDF)

#Evaluate on validation
sqrt(sum((pred1V-validation$wage)^2))
sqrt(sum((pred2V-validation$wage)^2))
sqrt(sum((combPredV-validation$wage)^2))

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><> #



#--------------------_-------------_--- #$#--------------------_-------------_--- #$#

# -- Forecasting  --------------------_-------------_--- #$#

#--------------------_-------------_--- #$#--------------------_-------------_--- #$#



# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><> #

# 
# Data are dependent over time
# Specific pattern types
# Trends - long term increase or decrease
# Seasonal patterns - patterns related to time of week, month, year, etc.
# Cycles - patterns that rise and fall periodically
# Subsampling into training/test is more complicated
# Similar issues arise in spatial data
# Dependency between nearby observations
# Location specific effects
# Typically goal is to predict one or more observations into the future.
# All standard predictions can be used (with caution!)

#--------------------_-------------_--- #$#--------------------_-------------_--- #$#


# Google data

library(quantmod)
from.dat <- as.Date("01/01/08", format="%m/%d/%y")
to.dat <- as.Date("12/31/13", format="%m/%d/%y")
getSymbols("GOOG", src="google", from = from.dat, to = to.dat)
head(GOOG)

# Summarize monthly and store as time series
library(dplyr)
mGoog <- GOOG[,-5]
mGoog <- to.monthly(mGoog)
googOpen <- Op(mGoog)
ts1 <- ts(googOpen,frequency=12)
plot(ts1,xlab="Years+1", ylab="GOOG")


# Example time series decomposition
# 
# Trend - Consistently increasing pattern over time
# Seasonal - When there is a pattern over a fixed period of time that recurs.
# Cyclic - When data rises and falls over non fixed periods
# 

# Decompose a time series into parts

library(forecast)
plot(decompose(ts1),xlab="Years+1")

# Training and test sets

ts1Train <- window(ts1,start=1,end=5)
ts1Test <- window(ts1,start=5,end=(7-0.01))
ts1Train

# Simple moving average

plot(ts1Train)
lines(ma(ts1Train,order=3),col="red")

# Exponential smoothing

ets1 <- ets(ts1Train,model="MMM")
fcast <- forecast(ets1)
plot(fcast); lines(ts1Test,col="red")

# Get the accuracy

accuracy(fcast,ts1Test)



# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><> #

#--------------------_-------------_--- #$#--------------------_-------------_--- #$#

# -- Unsupervised Prediction  --------------------_-------------_--- #$#

#--------------------_-------------_--- #$#--------------------_-------------_--- #$#

# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><> #


data(iris); library(ggplot2)
inTrain <- createDataPartition(y=iris$Species,
                               p=0.7, list=FALSE)
training <- iris[inTrain,]
testing <- iris[-inTrain,]
dim(training); dim(testing)

# Cluster with k-means

kMeans1 <- kmeans(subset(training,select=-c(Species)),centers=3)
training$clusters <- as.factor(kMeans1$cluster)
qplot(Petal.Width,Petal.Length,colour=clusters,data=training)

# Compare to real labels

table(kMeans1$cluster,training$Species)

# Build predictor

modFit <- train(clusters ~.,data=subset(training,select=-c(Species)),method="rpart")
table(predict(modFit,training),training$Species)

# Apply on test

testClusterPred <- predict(modFit,testing) 
table(testClusterPred ,testing$Species)










