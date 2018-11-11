### Problem 1 ###
## Use segmenetation data to build a rpart model

library(AppliedPredictiveModeling); data(segmentationOriginal)
library(caret)
library(e1071)

# Split Train/Test
train <- segmentationOriginal[segmentationOriginal$Case == "Train",]
test  <- segmentationOriginal[segmentationOriginal$Case == "Test",]

# Predict & observe tree
set.seed(125)
tree_model <- train(Class~., data = train, method = "rpart")
plot(tree_model$finalModel)
text(tree_model$finalModel)


### Problem 3 ###
## Build a rpart model for olive oil data and predict on a new data

# load data
library(pgmm)
data(olive)
olive = olive[,-1]

# build rpart model & predict on new data
tree_model <- train(Area~., data = olive, method = "rpart")
newdata = as.data.frame(t(colMeans(olive)))
predict(tree_model, newdata)


### Problem 4 ###
## Fit a GLM model on South Africa Heart Disease Data and 
## calculate misclassification rate on training and test sets

# load data and split train/test
library(ElemStatLearn)
data(SAheart)
set.seed(8484)
train = sample(1:dim(SAheart)[1],size=dim(SAheart)[1]/2,replace=F)
trainSA = SAheart[train,]
testSA = SAheart[-train,]

# fit model and predict on train/test
set.seed(13234)
glm_model <- train(chd~age+alcohol+obesity+tobacco+typea+ldl, data = trainSA, method = "glm", family = "binomial")
prediction_test <- predict(glm_model, testSA)
prediction_train <- predict(glm_model, trainSA)

# calculate misclassification
missClass = function(values,prediction){
  sum(((prediction > 0.5)*1) != values)/length(values)}
missClass(testSA$chd, prediction_test)
missClass(trainSA$chd, prediction_train)


### Problem 5 ###
## Fit a random forest predictor on vowel data and observe the variable imporantance

# load data and set y to factor
library(ElemStatLearn)
library(randomForest)
data(vowel.train)
data(vowel.test)
vowel.train$y <- factor(vowel.train$y)
vowel.test$y <- factor(vowel.test$y)

# Fit RF model & observe varimp
set.seed(33833)
rf_model <- train(y~., data = vowel.train, method = "rf")
varImp(rf_model)
