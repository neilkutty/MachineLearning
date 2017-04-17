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


