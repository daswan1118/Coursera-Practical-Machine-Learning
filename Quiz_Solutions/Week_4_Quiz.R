### Problem 1 ### 
## Fit random forest and boosted predictor (gbm) on vowel.train
## Then calculate the accuracies of the two data sets and the agreement accuracy

# load data
library(caret)
library(ElemStatLearn)
data(vowel.train)
data(vowel.test)
vowel.train$y <- as.factor(vowel.train$y)
vowel.test$y <- as.factor(vowel.test$y)

# model fitting
set.seed(33833)
rf_model <- train(y~.,data = vowel.train, method = "rf")
set.seed(33833)
gbm_model <- train(y~., data = vowel.train, method = "gbm")

# predict on test and calculate accuracy
prediction_rf <- predict(rf_model, vowel.test)
prediction_gbm <- predict(gbm_model, vowel.test)
confusionMatrix(vowel.test$y, prediction_rf)$overall[["Accuracy"]]
confusionMatrix(vowel.test$y, prediction_gbm)$overall[["Accuracy"]]
confusionMatrix(prediction_gbm, prediction_rf)$overall[["Accuracy"]]


### Problem 2 ###
## Fit rf, gbm, lda (linear discriminant analysis) on Alzheimer's data
## Caulcuate stacked accuracy

# load data
library(caret)
library(gbm)
set.seed(3433)
library(AppliedPredictiveModeling)
data(AlzheimerDisease)

# Divide test/train
adData = data.frame(diagnosis,predictors)
set.seed(3433)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]

# Fit RF, GBM, LDA
set.seed(62433)
rf_model <- train(diagnosis~.,data = training, method = "rf")
set.seed(62433)
gbm_model <- train(diagnosis~., data = training, method = "gbm")
set.seed(62433)
lda_model <- train(diagnosis~., data = training, method = "lda")

# Predict on test and accuracy
testing$rf <- predict(rf_model, testing)
testing$gbm <- predict(gbm_model, testing)
testing$lda <- predict(lda_model, testing)
confusionMatrix(testing$diagnosis, testing$rf)$overall[["Accuracy"]]
confusionMatrix(testing$diagnosis, testing$gbm)$overall[["Accuracy"]]
confusionMatrix(testing$diagnosis, testing$lda)$overall[["Accuracy"]]

# Fit stacked model on RF and accuracy
model <- train(diagnosis~rf+gbm+lda, data = testing, method = "rf")
prediction <- predict(model, testing)
confusionMatrix(testing$diagnosis, prediction)$overall[["Accuracy"]]


### Problem 3 ###
## Fit Lasso model to concerte data and 
## determine last coefficent to be set to zero as penalty increases

# Load Data
set.seed(3523)
library(AppliedPredictiveModeling)
data(concrete)
library(elasticnet)

# Split train/test
set.seed(3523)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]

# Fit Lasso and plot coefficient penalty graph
set.seed(233)
lasso_model <- enet(CompressiveStrength~., data = training)
plot.enet(lasso_model$finalModel, xvar="penalty")


### Problem 4 ###
## Fit a forecast package (bats) to number of visitor time series data
## Calculate 95% prediction interval bounds

# load data and split training/testing
library(lubridate) # For year() function below
library(forecast)
dat = read.csv("C:/Users/Swan/Downloads/gaData.csv")
training = dat[year(dat$date) < 2012,]
testing = dat[(year(dat$date)) > 2011,]
tstrain = ts(training$visitsTumblr)

# Fit model and forecast
ts_model <- bats(training$visitsTumblr)
testing$upper <- forecast(ts_model,235,level =95)$upper

# Calculate if within 95% interval bounds
sum(testing$visitsTumblr < testing$upper)/length(testing$visitsTumblr)


### Problem 5 ###
## Fit concerete data on SVM and calculate RMSE

# load data and split train/test
set.seed(3523)
library(AppliedPredictiveModeling)
library(e1071)
library(Metrics)
data(concrete)
set.seed(3523)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]

# Fit SVM
svm_model <- svm(CompressiveStrength~., data = training)
predictions <- predict(svm_model, testing)
rmse(testing$CompressiveStrength, predictions)
