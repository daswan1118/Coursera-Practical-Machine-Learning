library(caret)
library(kernlab)
library(e1071)

### Sample Caret Model using GLM

data(spam)


#1 Train & Test
  # 1a. Option 1: Data Partition
    inTrain <- createDataPartition(y=spam$type, p=0.75, list=FALSE)
    training <- spam[inTrain,]
    testing <- spam[-inTrain,]
    dim(training)
  
  # 1b. Option 2: K folds
    folds <- createFolds (y=spam$type, k=10, list=TRUE, returnTrain=TRUE)
    sapply(folds,length)
    folds[[1]][1:10]
  
  # 1c. Option 3: Resampling
    folds <- createResample (y=spam$type, times=10, list=TRUE)
    sapply(folds,length)
    folds[[1]][1:10]
  
  # 1d. Option 4: Time Slices
    tme <- 1:1000
    folds <- createTimeSlices (y=tme, initialWindow=20, horizon=10)
    names(fold)
    fold$train[[1]]
    fold$test[[1]]

    
# 2. Basic Preprocessing    
  
  # a. Standardizing (when too skewed)  
    hist(training$capitalAve,main="",xlab="ave. capital run length")
    trainCapAve <- training$capitalAve
    trainCapAveS <- (trainCapAve - mean(trainCapAve))/sd(trainCapAve)
    # standardize testing by train
    testCapAve <- testing$capitalAve
    testCapAveS <- (testCapAve - mean(trainCapAve))/sd(trainCapAve)
    
  # b. Standardizing with -preProcess function
    preObj <- preProcess(training[,-58],method=c("center","scale"))
    preObj_test <- predict(preObj, testing[,-58])
    
  # c. Standardizing - Box-Cox transformations
    preObj <- preProcess(training[,-58],method=c("BoxCox"))
    trainCapAveS <- predict(preObj, training[,-58])$capitalAve    
    par(mfrow=c(1,2)); hist(trainCapAveS); qqnorm(trainCapAveS)
    
  # d. Standardizing - Imputing Data
    preObj <- preProcess(training[,-58], method="knnImpute")
    capAve <- predict(preObj,training[,-58])$capAve
    capAveTruth <- training$capitalAve
    capAveTruth <- (capAveTruth-mean(capAveTruth))/sd(capAveTruth)
  
      
# 3. Preprocessing with PCA
# Reduce number of predictors and reduce noise by averaging
    
  # a. correlated predictors
    M <- abs(cor(training[,-58]))
    diag(M) <- 0
    which(M > 0.8, arr.ind = T)
    
  # b. PCA - 2 variables
    smallSpam <- spam[,c(34,32)]
    prComp <- prcomp(smallSpam)
    plot(prComp$x[,1],prComp$x[,2])
    
  # c. PCA - multiple variables
    typeColor <- ((spam$type=="spam")*1+1)
    prComp <- prcomp(log10(spam[,-58]+1))
    plot(prComp$x[,1],prComp$x[,2],col=typeColor,xlab="PC1",ylab="PC2")
    
  # d. PCA with caret
    preProc <- preProcess(log10(training[,-58]+1), method="pca", pcaComp=2)
    trainPC <- predict(preProc, log10(training[,-58]+1))
    modelFit <- train(x = trainPC, y = training$type, method="glm")
    testPC <- predict(preProc,log10(testing[,-58]+1))
    confusionMatrix(testing$type,predict(modelFit,testPC))
    
    
    
# 4. Fit GLM Model
# arg(train)
# arg(trainControl)
  set.seed(32343)
  modelFit <- train(type ~., data = training, method = 'glm')
  modelFit
  modelFit$finalModel


# 5. Predict & Confusion Matrix
  predictions <- predict(modelFit, newdata = testing)
  predictions
  confusionMatrix(predictions,testing$type)
