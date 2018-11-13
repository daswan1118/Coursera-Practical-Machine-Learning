library(caret)
library(ISLR)
library(ggplot2)
library(Hmisc)

### Sample Caret Model using GLM

Wage <- subset(Wage,select=-c(logwage))
data(Wage)
summary(Wage)


# 1. Train/Test - Data Partition
  inTrain <- createDataPartition(y=Wage$wage, p=0.7, list=FALSE)
  training <- Wage[inTrain,]
  testing <- Wage[-inTrain,]
  dim(training); dim(testing)


# 2. Plotting Predictors
  # a. Plot correlation between several variables
  featurePlot(x=training[,c("age", "education", "jobclass")],
              y=training$wage,
              plot="pairs")

  # b. Plot correlation between 2 variables using ggplot
    qplot(age, wage, colour=jobclass, data=training)
    qq <- qplot(age, wage, colour=education, data=training)
    qq + geom_smooth(method='lm', formula=y~x)
  
  # c. Cut groups 
    cutWage <- cut2(training$wage, g=3)
    table(cutWage)
  
  # d. tables
    t1 <- table(cutWage, training$jobclass)
    t1
    prop.table(t1,1)
  
  # e. density plots
    qplot(wage, colour=education, data=training, geom="density")

    
# 3. Covariate Creation
  
  # a. convert factor variables to indicator (1,0) variables  
    table(training$jobclass)
    dummies<-dummyVars(wage~jobclass, data=training)
    head(predict(dummies),newdata=training)
  
  # b. remove zero covariates
    nsv <- nearZeroVar(training, saveMetrics=TRUE)
    nsv
    
  # c. spline basis
    library(splines)
    bsBasis <- bs(training$age, df=3)
    bsBasis
    lm1 <- lm(wage ~ bsBasis, data=training)
    plot(training$age,training$wage,pch=19,cex=0.5)
    points(training$age, predict(lm1, newdata=training), col="red", pch=19,cex=0.5)
    
    
# 4. Fit a linear model
  modFit <- train(wage ~ age + jobclass + education, 
                  method = "lm", data = training)
  finMod <- modFit$finalModel
  print(modFit)


# 5. Diagnostic 
  # a. residual vs fitted
  plot(finMod, 1, pch=19, cex=0.5, col="#00000010") # want to see a straight line
  
  # a2. residual vs fitted with color
  qplot(finMod$fitted,finMod$residuals,colour=race,data=training) # want to see a straight line
  
  # b. Plot by index
  plot(finMod$residuals,pch=19) # there should be no trend or outliers on one side - suggests missing variable
  
  # c. Predicted vs trust in test set
  pred <- predict(modFit, testing)
  qplot(wage,pred,colour=year,data=testing) # test set should not impact training
  
