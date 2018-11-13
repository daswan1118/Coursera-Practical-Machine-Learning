library(caret);data(faithful);set.seed(236)

# Get data
inTrain <- createDataPartition(y=faithful$waiting, p=0.5, list=FALSE)
trainFaith <- faithful[inTrain,]
testFaith <- faithful[-inTrain,]
head(trainFaith)


# Fit linear model
lm1 <- lm(eruptions ~ waiting, data = trainFaith)
summary(lm1)

# Plot
plot(trainFaith$waiting, trainFaith$eruptions, pch=19, col="blue", xlab="waiting", ylab="duration")
lines(trainFaith$waiting, lm1$fitted,lwd=3)

# Predict a new value
  # option 1
  coef(lm1)[1] + coef(lm1)[2]*80
  # option 2
  newdata <- data.frame(waiting=80)
  predict(lm1,newdata)
  
# Get RMSE
sqrt(sum((lm1$fitted-trainFaith$eruptions)^2))
sqrt(sum((predict(lm1,newdata=testFaith)-testFaith$eruptions)^2))
  
# Prediction Intervals
pred1 <- predict(lm1, newdata=testFaith, interval="prediction")
ord <- order(testFaith$waiting)
plot(testFaith$waiting,testFaith$eruptions,pch=19,col="blue")
matlines(testFaith$waiting[ord], pred1[ord,],type="l",col=c(1,2,2),lty=c(1,1,1),lwd=3)


# Do it with caret
modFit <- train(eruptions~waiting, data=trainFaith, method="lm")
summary(modFit$finalModel)
