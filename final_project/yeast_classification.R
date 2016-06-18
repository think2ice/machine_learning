# Dataset available online at: 
# http://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data

# Description of the variables
#http://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.names
# Delete all variables
# rm(list=ls(all=TRUE))

# So as that the same results could be obtained again 
set.seed(777)

# 0. Reading the dataset

# Locally
setwd("/Users/manel/Documents/Universidad/MIRI/Q1B/ML/my_labs/machine_learning.git/final_project")
yeast <- read.csv2(file = "yeast.csv", header = FALSE, dec = ".")
# Directly from UCI machine learning repository
# yeast <- read.table(url("http://archive.ics.uci.edu/ml/machine-learning-databases/
#                         yeast/yeast.data"),header = FALSE)

# 1. Basic analysis

# Adding row and column names to the dataset
cols.names <- c("seq.name", "mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "nuc","class")
names(yeast) <- cols.names
# The first column should be treated as the row.names but there are inputs that are repeated
# To deal with this problem, the function make names is ideal
row.names(yeast) <- make.names(yeast$seq.name, unique = TRUE)
yeast <- yeast[,-1]
# Basic inspection of the dataset
dim(yeast)
summary(yeast)
str(yeast)
# the erl variable appears as numerical but in fact it is a binary variable, so it 
# should be redefined as a factor
yeast$erl <- as.factor(yeast$erl)
# Plots of 2 vs 2 variables
plot(yeast)
# Plots of the histogram for each continuous variable
par(mfrow = c(3,3))
for (i in c(1:4,6:8)) {
  hist(yeast[,i], main = names(yeast)[i])
}
# Barplot of the erl variable
barplot(table(yeast$erl), main = colnames(yeast)[5])
# Barplot of the target, class
barplot(table(yeast$class), main = colnames(yeast)[9])

# 3. Preprocess the data

# Notice that our target variable, class, is completely unbalanced
# With CYT, NUC, MIT and ME3 the most frequent classes
table(yeast[,ncol(yeast)])
# Looking at the data density, perhaps a transformation could be applied to 
# variables erl and pox
summary(yeast[,6:7])
# pox is always almost 0

# 4. Define training and test data

# Get the training and test data
N <- dim(yeast)[1]
learn <- sample(1:N, round(2/3*N))
yeast.train <- yeast[learn,]
yeast.test <- yeast[-learn,]
dim(yeast.train)
dim(yeast.test)

library(TunePareto)
# Prepare a crossvalidation 10x10 method to get the best model with
# several classifiers
k <- 10
CV.folds <- generateCVRuns(yeast.train$class, ntimes=1, nfold=k, stratified=TRUE)

# 5. Classification methods

# baseline: the error that we get predicting always the most probable class
(baseline <- 100*(1 - max(table(yeast$class))/nrow(yeast)))
# let's see if this can be improved using: 
# 1. Naive Bayes
# 2. LDA
# 3. QDA
# 4. KNN
# 5. PCA + Neural Networks
# 6. Random Forest

# With cross-validation
# prepare the structure to store the partial results
cv.results <- matrix (rep(0,4*k),nrow=k)
colnames (cv.results) <- c("k","fold","TR error","VA error")

cv.results[,"TR error"] <- 0
cv.results[,"VA error"] <- 0
cv.results[,"k"] <- k
for (j in 1:k) {
  # get VA data
  va <- unlist(CV.folds[[1]][[j]])
  tr <- yeast.train[-va,]
  # train on TR data
  yeast.nb <- naiveBayes(tr$class ~ . , data = tr, laplace=3)
  
  # predict TR data
  pred.tr <- predict(yeast.nb,newdata=tr)
  
  tab <- table(tr$class, pred.tr)
  cv.results[j,"TR error"] <- 1-sum(tab[row(tab)==col(tab)])/sum(tab)
  
  # predict VA data
  pred.va <- predict(yeast.nb,newdata=yeast.train[va,])
  tab <- table(yeast.train[va,]$class, pred.va)
  cv.results[j,"VA error"] <- 1-sum(tab[row(tab)==col(tab)])/sum(tab)
  cv.results[j,"fold"] <- j
}
cv.results
# Average of the validation error

(VA.error <- mean(cv.results[,"VA error"]))
# necessary library for Naive Bayes
library(e1071)
# necessary library for Random Forest
library(randomForest)
# necessary library for LDA and QDA
library(MASS)
# necessary library for Neural Networks
library(nnet)

k <- 10
CV.folds <- generateCVRuns(yeast.train$class, ntimes=1, nfold=k, stratified=TRUE)
Model.CV <- function (method)
{
  cv.results <- matrix (rep(0,4*k),nrow=k)
  colnames (cv.results) <- c("k","fold","TR error","VA error")
  cv.results[,"TR error"] <- 0
  cv.results[,"VA error"] <- 0
  cv.results[,"k"] <- k
  resp.var <- which(colnames(yeast.train)=='class')
  for (j in 1:k) {
    # get VA data
    va <- unlist(CV.folds[[1]][[j]])
    tr <- yeast.train[-va,]
    # train on TR data
    if (method == "NaiveBayes") {
      model <- naiveBayes(tr$class ~ . , data = tr, laplace=3)
      # predict TR data
      pred.tr <- predict(model,newdata=tr)
      tab <- table(tr$class, pred.tr)
      cv.results[j,"TR error"] <- 1-sum(tab[row(tab)==col(tab)])/sum(tab)
      pred.va <- predict(model,newdata=yeast.train[va,])
      tab <- table(yeast.train[va,]$class, pred.va)
      cv.results[j,"VA error"] <- 1-sum(tab[row(tab)==col(tab)])/sum(tab)
      cv.results[j,"fold"] <- j
    }
    else if (method == "QDA"){
      model <- qda(tr$class ~ . , data = tr, CV=FALSE)
      # predict TR data
      pred.tr <- predict(model)$class
      tab <- table(tr$class, pred.tr)
      cv.results[j,"TR error"] <- 1-sum(tab[row(tab)==col(tab)])/sum(tab)
      pred.va <- predict(model,newdata=yeast.train[va,])$class
      tab <- table(yeast.train[va,]$class, pred.va)
      cv.results[j,"VA error"] <- 1-sum(tab[row(tab)==col(tab)])/sum(tab)
      cv.results[j,"fold"] <- j
    }
    else if (method == "LDA"){
      model <- lda(tr$class ~ . , data = tr, CV=FALSE)
      # predict TR data
      pred.tr <- predict(model)$class
      tab <- table(tr$class, pred.tr)
      cv.results[j,"TR error"] <- 1-sum(tab[row(tab)==col(tab)])/sum(tab)
      pred.va <- predict(model,newdata=yeast.train[va,])$class
      tab <- table(yeast.train[va,]$class, pred.va)
      cv.results[j,"VA error"] <- 1-sum(tab[row(tab)==col(tab)])/sum(tab)
      cv.results[j,"fold"] <- j
    }
    else if (method == "RandomForest"){
      rf <- randomForest(formula = class ~., data = tr, xtest = yeast.train[va,-resp.var],
                         ytest = yeast.train[va,resp.var])
      cv.results[j,"TR error"] <- 1 - sum(diag(rf$confusion[,-11]) / sum(rf$confusion[,-11]))
      cv.results[j,"VA error"] <- 1 - sum(diag(rf$test$confusion[,-11]) / sum(rf$test$confusion[,-11]))
      cv.results[j,"fold"] <- j
    }
    else {
      stop("Unknown method. The only valid methods ara NaiveBayes, LDA, QDA and RandomForest")
    }
  }
  
  return(list(cv.results, mean(cv.results[,"VA error"])))
  # mean(cv.results[,"VA error"])
}
# Test the behavior for Naive Bayes
Model.CV("NaiveBayes")
# Test the behavior for LDA
Model.CV("LDA")
# Test the behavior for QDA
Model.CV("QDA")
# Test the behavior for KNN
Model.CV("KNN")
# Test the behavior for Random Forest
Model.CV("RandomForest")

# Trying to solve our problem with PCA and Neural Networks
# First we execute PCA to standarized and uncorrelated the data
library("FactoMineR")
par(mfrow = c(1,2))
# find the index of the test individuals
test <- which(!is.na(match(row.names(yeast),row.names(yeast.test))))
# test <- as.vector(test)
pca.yeast <- PCA(yeast, quali.sup = c(5,9), ind.sup = test)

# Plot of the individuals (using class as a color) 
par(mfrow = c(1,1))
plot(pca_yeast$ind$coord, col = yeast$class)
# Try to guess how many PC are the optimal for this problem
# Plot of the eigenvalues
plot(pca.yeast$eig$eigenvalue, type = "b", main = "Eigenvalues")
pca.yeast$eig
# Following the Kaiser rule (keeping at least 80% of the variance) we decided to 
# keep 5 eigenvalues (which is the default number in PCA))
# Now we can train a neural network
pca.yeast.train <- pca.yeast$ind$coord
summary(yeast)



# 6. Results: Which classification algorithm is the best? 


# 7. Train the X algorithm with all the training data and test its behavior
# for predicting the test data

# Naive Bayes

yeast.nb <- naiveBayes (yeast.train$class ~ ., data=yeast.train, laplace=3)
# predict the test data
pred <- predict(yeast.nb,newdata=yeast.test)
(tt <- table(Truth=yeast.test$class, Predicted=pred))
(error <- 100*(1-sum(diag(tt))/sum(tt)))
# Reduction of the error
100*(baseline-error)/baseline
# Real error for each class
for (i in 1:length(levels(yeast$class))) {
  cat(levels(yeast.test$class)[i])
  print(1- tt[i,i]/sum(tt[i,]))
}

# LDA 

model <- lda(yeast.train$class ~ . , data = yeast.train, CV=FALSE)
# predict Test data
pred <- predict(model, newdata = yeast.test)$class
(tt <- table(Truth=yeast.test$class, Predicted=pred))
(error <- 100*(1-sum(diag(tt))/sum(tt)))
# Reduction of the error
100*(baseline-error)/baseline
# Real error for each class
for (i in 1:length(levels(yeast$class))) {
  cat(levels(yeast.test$class)[i])
  print(1- tt[i,i]/sum(tt[i,]))
}

# QDA
# Only works after feature selection, due to the fact that some modalities
# of yeast$class are too rare or appear so few times

model <- qda(yeast.train$class ~ . , data = yeast.train, CV=FALSE)
# predict TR data
pred <- predict(model, yeast.test)$class
(tt <- table(Truth=yeast.test$class, Predicted=pred))
(error <- 100*(1-sum(diag(tt))/sum(tt)))
# Reduction of the error
100*(baseline-error)/baseline
# Real error for each class
for (i in 1:length(levels(yeast$class))) {
  cat(levels(yeast.test$class)[i])
  print(1- tt[i,i]/sum(tt[i,]))
}


# KNN


# Neural Networks

model.nnet <- nnet(class ~., data = yeast, subset=learn, size=20, maxit=200, decay=0.01)
summary(model.nnet)
# TR error
p1 <- as.factor(predict (model.nnet, type="class"))
t1 <- table(p1,yeast.train$class)
(error_rate.learn <- 100*(1-sum(diag(t1))/dim(yeast.train)[1]))
# TEST error
p2 <- as.factor(predict (model.nnet, newdata=yeast.test, type="class"))
t2 <- table(p2,yeast.test$class)
(error_rate.test <- 100*(1-sum(diag(t2))/dim(yeast.test)[1]))
library(caret)
trc <- trainControl (method="repeatedcv", number=10, repeats=10)
decays <- 10^seq(-3,0,by=0.2)
model.10x10CV <- train (class ~., data = yeast, subset=learn, method='nnet', maxit = 200, trace = FALSE,
                        tuneGrid = expand.grid(.size=20,.decay=decays), trControl=trc)
## We can inspect the full results
model.10x10CV$results
## and the best model found
model.10x10CV$bestTune
# Fit the best model with all the training and check the results: 
model.nnet <- nnet(class ~., data = yeast, subset=learn, size=20, maxit=200, decay=0.01)


# Random Forest

rf <- randomForest(formula = class ~., data = yeast.train, xtest = yeast.test[,-9],
                   ytest = yeast.test[,9])
print(rf)



