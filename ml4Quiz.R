# Practical Machine Learning Quiz 4


# Question 1 --------------------------------------------------------------------------#

library(ElemStatLearn)
library(caret)
data(vowel.train)
data(vowel.test)
set.seed(33833)

rf <- train(as.factor(y)~., data = vowel.train, method = 'rf' )
gb <- train(as.factor(y)~., data = vowel.train, method = 'gbm')

rfPs <- predict(rf, vowel.test)
gbPs <- predict(gb, vowel.test)

rfPs == gbPs
table(rfPs == gbPs)

agree = table(rfPs == gbPs)[[2]]/nrow(vowel.test)

table(rfPs == vowel.test$y)
table(gbPs == vowel.test$y)

rfAcc <- table(rfPs == vowel.test$y)[[2]] / nrow(vowel.test)
gbAcc <- table(gbPs == vowel.test$y)[[2]] / nrow(vowel.test)


length(rfPs[rfPs == gbPs])
length(gbPs[rfPs == gbPs])


agreeAcc <- table(rfPs[rfPs == gbPs] == vowel.test$y[rfPs == gbPs])[[2]]/nrow(vowel.test[rfPs == gbPs,])

atable(rfPs == vowel.test$y)[rfPs == gbPs]

# Question 2 --------------------------------------------------------------------------#
# Set the seed to 62433 and predict diagnosis with all the other variables 
# using a random forest ("rf"), boosted trees ("gbm") and linear discriminant analysis
# ("lda") model. 
# 
# Stack the predictions together using random forests ("rf"). 
# What is the resulting accuracy on the test set? 
# Is it better or worse than each of the individual predictions?

library(caret)
library(gbm)
set.seed(3433)
library(AppliedPredictiveModeling)

data(AlzheimerDisease)

adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]

training = adData[ inTrain,]
testing = adData[-inTrain,]

set.seed(62433)

rfAlz <- train(diagnosis ~ ., data = training, method = 'rf')
gbAlz <- train(diagnosis ~ ., data = training, method = 'gbm')
ldAlz <- train(diagnosis ~ ., data = training, method = 'lda')

rfAlzP <- predict(rfAlz, testing)
gbAlzP <- predict(gbAlz, testing)
ldAlzP <- predict(ldAlz, testing)

pDF <- data.frame(rfAlzP, gbAlzP, ldAlzP, y = testing$diagnosis)

allFit <- train(y ~ ., data = pDF, method = 'rf')
allP <- predict(allFit, testing)

confusionMatrix(rfAlzP, testing$diagnosis)$overall
confusionMatrix(gbAlzP, testing$diagnosis)$overall
confusionMatrix(ldAlzP, testing$diagnosis)$overall

confusionMatrix(allP, testing$diagnosis)$overall



# Question 3 --------------------------------------------------------------------------#

set.seed(3523)
library(AppliedPredictiveModeling)
data(concrete)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]

set.seed(233)
lasso <- train(CompressiveStrength ~ ., data = training, method = 'lasso')
plot(lasso$finalModel)

#Question 4 (!!! Review !!!)

dat = read.csv('https://d396qusza40orc.cloudfront.net/predmachlearn/gaData.csv')

library(lubridate) # For year() function below
training = dat[year(dat$date) < 2012,]
testing = dat[(year(dat$date)) > 2011,]

tstrain = ts(training$visitsTumblr)

library(forecast)
mod_ts <- bats(tstrain)
fcast <- forecast(mod_ts, level = 95, h = dim(testing)[1])
sum(fcast$lower < testing$visitsTumblr & testing$visitsTumblr < fcast$upper) / 
    dim(testing)[1]


#Question 5 --------------------------------------------------------------------------#

set.seed(3523)
library(AppliedPredictiveModeling)
data(concrete)

inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]

training = concrete[ inTrain,]
testing = concrete[ -inTrain,]

set.seed(325)

library(e1071)
library(forecast)
svm <- svm(CompressiveStrength ~ ., data = training)
svmP <- forecast(svm, testing)

accuracy(svmP, testing$CompressiveStrength)
