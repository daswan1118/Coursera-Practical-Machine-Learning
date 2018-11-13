library(caret)
library(Metrics)
library(neuralnet)


# More info on caret: https://topepo.github.io/caret/index.html 


# 1. Get Data
train <- read.csv("~\\Housing Kaggle\\Data\\train.csv")
score <- read.csv("~\\\Housing Kaggle\\Data\\test.csv")


# 2. Split - using Caret
set.seed(695)
inTrain <- createDataPartition(y=train$SalePrice, p=0.70, list=FALSE)
training <- train[inTrain,]
testing <- train[-inTrain,]


# 3. Understand data
# a. basic tools
head(training,6)
str(training)
colnames(training)
summary(training)

# b. Feature Plot - caret
featurePlot(x=training[,c(20,42,52)], y=training$SalePrice, plot="pairs")
featurePlot(x=training[,c(20,18,52)], y=training$SalePrice, plot="scatter",
            type = c("p", "smooth"), layout = c(3,1))


# 4. Preprocess
# Remove target variable
train_tv <- log(training$SalePrice + 1)
training <- training[,-81]
test_tv <- log(testing$SalePrice + 1)
testing <- testing[,-81]

# a. convert factor variables to indicator (1,0) variables  
dummies <- dummyVars(~., data=training)
train_df <- as.data.frame(predict(dummies, newdata=training))
test_df <- as.data.frame(predict(dummies, newdata=testing))
score_df <- as.data.frame(predict(dummies, newdata=score))

# Choose Variables
variables <- c('OverallQual','GarageArea','X1stFlrSF','FullBath','TotRmsAbvGrd',
               'Fireplaces','CentralAir.Y','BsmtFinSF1','HalfBath','YearRemodAdd',
               'LotArea','BsmtQual.Ex','MSZoning.RL','KitchenQual.Gd','BedroomAbvGr',
               'OverallCond','BsmtExposure.Gd','BldgType.1Fam','BsmtFullBath',
               'KitchenQual.Ex','BsmtQual.Gd','GarageYrBlt','LotFrontage','BsmtFinType1.GLQ',
               'ExterQual.Gd','PavedDrive.N','OpenPorchSF','GarageFinish.Unf','HeatingQC.TA',
               'MasVnrArea','LandSlope.Gtl','BsmtExposure.No','BsmtUnfSF','HeatingQC.Gd',
               'GarageType.Attchd','LotShape.IR1','BsmtFinType1.Unf','GarageFinish.RFn',
               'Neighborhood.Somerst','WoodDeckSF','SaleCondition.Abnorml','LandContour.Lvl',
               'RoofStyle.Gable','Exterior2nd.VinylSd','HeatingQC.Ex','Condition1.Norm',
               'Neighborhood.NAmes','Alley.Pave','HouseStyle.1Story','SaleType.WD')
train_df <- train_df[,variables]
test_df <- test_df[,variables]
score_df <- score_df[,variables]

# b. Zero Variance
# nzv <- nearZeroVar(train_df, saveMetrics=TRUE)
# nzv[nzv$nzv,][1:10,]
# nzv <- nearZeroVar(train_df)
# train_df <- train_df[, -nzv]
# test_df <- test_df[, -nzv]
# score_df <- score_df[, -nzv]

# c. Correlated Predictors
# descrCor <-  cor(train_df, use="pairwise.complete.obs")
# print(descrCor)
# descrCor[is.na(descrCor)] <- 0
# highlyCorDescr <- findCorrelation(descrCor, cutoff = .75)
# train_df <- train_df[,-highlyCorDescr]
# test_df <- test_df[,-highlyCorDescr]
# score_df <- score_df[,-highlyCorDescr]

# d. Standardizing - Imputing Data
preObj <- preProcess(train_df, method = "medianImpute")
train_df <- predict(preObj,train_df)
test_df <- predict(preObj,test_df)
score_df <- predict(preObj,score_df)



# 5. Model training
#a. GLM
fitControl <- trainControl(method = "cv", number = 3,
                           verboseIter = TRUE, returnResamp = "all")
set.seed(695)
test_reg_cv_model <- train(train_df, train_tv, 
                           method = "glm", 
                           trControl = fitControl,
                           preProc = c("center", "scale"))
test_reg_imp <- varImp(test_reg_cv_model); test_reg_imp; plot(test_reg_imp) 
test_reg_pred <- predict(test_reg_cv_model, test_df)
rmse(test_tv, test_reg_pred)

## submission
score$SalePrice_glm <- exp(predict(test_reg_cv_model, score_df))+1
submission <- score[,c("Id","SalePrice")]
write.csv(submission,'C:\\Users\\kk82\\Desktop\\Housing Kaggle\\Data\\submission.csv', row.names=FALSE)


#b. Random Forest
tunegrid <- expand.grid(.mtry=10)
fitControl <- trainControl(method = "cv", number = 3,
                           verboseIter = TRUE, returnResamp = "all")  
set.seed(695)
rrfFit <- train(train_df, train_tv, method = "rf", trControl = fitControl,
                preProc = c("center", "scale"), ntree = 200, tuneGrid=tunegrid,
                importance = TRUE, verbose = TRUE)
test_reg_imp_rf <- varImp(rrfFit, scale = FALSE); test_reg_imp; plot(test_reg_imp)
top_var <- rownames(data.frame(test_reg_imp[1]))[order(data.frame(test_reg_imp[1])$Overall, 
                                                       decreasing=TRUE)[1:35]]
test_reg_pred <- predict(rrfFit, test_df)
rmse(test_tv, test_reg_pred)    

## submission
score$SalePrice_rf <- exp(predict(rrfFit, score_df))+1
score$SalePrice <- 0.3*score$SalePrice_rf + 0.7*score$SalePrice_glm
submission <- score[,c("Id","SalePrice")]
write.csv(submission,'C:\\Users\\kk82\\Desktop\\Housing Kaggle\\Data\\submission.csv', row.names=FALSE)

#b1. Random Forest - cforest
fitControl <- trainControl(method = "cv", number = 3,
                           verboseIter = TRUE, returnResamp = "all")
set.seed(695)
rrfFit <- train(train_df, train_tv, method = "cforest", trControl = fitControl,
                preProc = c("center", "scale"), controls = party::cforest_unbiased(ntree = 20))
test_reg_imp <- varImp(rrfFit, scale = FALSE); test_reg_imp; plot(test_reg_imp)
top_var <- rownames(data.frame(test_reg_imp[1]))[order(data.frame(test_reg_imp[1])$Overall,
                                                       decreasing=TRUE)[1:50]]
test_reg_pred <- predict(rrfFit, test_df)
rmse(test_tv, test_reg_pred)

# ## submission
# score$SalePrice <- exp(predict(rrfFit, score_df))+1
# submission <- score[,c("Id","SalePrice")]
# write.csv(submission,'C:\\Users\\kk82\\Desktop\\Housing Kaggle\\Data\\submission.csv', row.names=FALSE)


#c. Neural Network - nnet caret
# nnetFit <- train(train_df, train_tv, method = "nnet", maxit = 100,
#                  trControl = fitControl,preProc = c("center", "scale"),
#                  trace = FALSE, linout = TRUE, verbose = TRUE)    
# test_reg_imp <- varImp(nnetFit); test_reg_imp; plot(test_reg_imp)
# test_reg_pred <- predict(nnetFit, test_df)
# rmse(test_tv, test_reg_pred)       


#d. XG Boost
fitControl <- trainControl(method = "cv", number = 3, 
                           verboseIter = TRUE, returnResamp = "all")  
xgbGrid <- expand.grid(nrounds = c(100),
                       max_depth = c(5),
                       eta = c(.4),
                       gamma = c(0,1),
                       colsample_bytree = c(1.0),
                       min_child_weight = c(0.5),
                       subsample = c(1))
set.seed(695)
xgbFit <- train(train_df, train_tv, method = "xgbTree", trControl = fitControl,
                preProc = c("center", "scale"), tuneGrid = xgbGrid,
                importance = TRUE, verbose = TRUE)
test_reg_imp <- varImp(xgbFit, scale = FALSE); test_reg_imp; plot(test_reg_imp)
top_var <- rownames(data.frame(test_reg_imp[1]))[order(data.frame(test_reg_imp[1])$Overall, 
                                                       decreasing=TRUE)[1:20]]
test_reg_pred <- predict(xgbFit, test_df)
rmse(test_tv, test_reg_pred)    

#submission
score$SalePrice <- exp(predict(xgbFit, score_df))+1
submission <- score[,c("Id","SalePrice")]
write.csv(submission,'submission.csv', row.names=FALSE)


