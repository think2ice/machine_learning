# Dataset available online at: 
# http://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data

# Description of the variables
#http://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.names
# Delete all variables
rm(list=ls(all=TRUE))

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

# necessary library for Naive Bayes
library(e1071)
# necessary library for Random Forest
library(randomForest)
# necessary library for LDA and QDA
library(class)
# necessary library for KNN
library(MASS)
# needed so as to perform a PCA
library(FactoMineR)
# necessary library for Neural Networks
library(nnet)
# needed for use CV with Neural Networks
library(caret)
# needed so as to tune the weights of a NN
library(TunePareto)
# needed so as to do X-validation

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

# Looking at the data density, perhaps a transformation could be applied to 
# variables erl and pox
summary(yeast[,6:7])
# pox is always almost 0
# Notice that our target variable, class, is completely unbalanced
table(yeast[,which(colnames(yeast)=='class')])
table(yeast$vac, yeast$class)

#######################################################################################################
# 3.3 Pre-processing Feature Selection
# With CYT, NUC, MIT and ME3 the most frequent classes
# and ERL, POX and VAC with very few individuals
# there are too few ERL, so we decide to not take into account this class
# so we delete all the individuals with this class
table(yeast$pox, yeast$class)
# table(yeast$erl, yeast$class)
yeast <- yeast[-which(yeast$class == 'ERL'),]
# this decision (deleting VAC) is made after checking that in all the classifications 
# made, the classifiers always make a 100% error in this variable
yeast <- yeast[-which(yeast$class == 'VAC'),]

yeast$class <- factor(yeast$class)
# The variable erl is specially used to detect ERL class, so we also delete it 
yeast <- yeast[,-which(colnames(yeast)=='erl')]
# yeast <- yeast[,-which(colnames(yeast)=='vac')]
str(yeast)

#######################################################################################################


# Also this time we are going to standarize the data 
# Pre-processing 3.1 Standarize the data
resp.var <- which(colnames(yeast)=='class')
yeast[,1:(resp.var-1)] <- scale(yeast[,1:(resp.var-1)],center = TRUE)
summary(yeast)
summary(yeast$class)

# 4. Define training and test data

# Get the training and test data
N <- dim(yeast)[1]
learn <- sample(1:N, round(2/3*N))
yeast.train <- yeast[learn,]
yeast.test <- yeast[-learn,]
dim(yeast.train)
dim(yeast.test)


#######################################################################################################
# 3.2 Pre-processing: Feature extraction, PCA

par(mfrow = c(1,2))
# find the index of the test individuals
test <- which(!is.na(match(row.names(yeast),row.names(yeast.test))))
# test <- as.vector(test)
pca.yeast <- PCA(yeast, quali.sup = c(which(colnames(yeast)=='erl'),which(colnames(yeast)=='class')), ind.sup = test)
# Plot of the individuals (using class as a color) 
par(mfrow = c(1,1))
plot(pca.yeast$ind$coord, col = yeast$class)
# Try to guess how many PC are the optimal for this problem
# Plot of the eigenvalues
plot(pca.yeast$eig$eigenvalue, type = "b", main = "Eigenvalues")
pca.yeast$eig
# Following the Kaiser rule (keeping at least 80% of the variance) we decided to 
# keep 5 eigenvalues (which is the default number in PCA))
# Get the corresponding new coordinates + the response variable class
pca.yeast.train <- pca.yeast$ind$coord
pca.yeast.train <- as.data.frame(pca.yeast.train)
pca.yeast.train[row.names(yeast.train),'class'] <- yeast.train[,which(colnames(yeast.train) == 'class')]
pca.yeast.test <- pca.yeast$ind.sup$coord
pca.yeast.test <- as.data.frame(pca.yeast.test)
pca.yeast.test[row.names(yeast.test),'class'] <- yeast.test[,which(colnames(yeast.test) == 'class')]
pca.yeast.data <- rbind(pca.yeast.train,pca.yeast.test)



# Both test: without PCA and with PCA (uncomment this lines so as to use PCA)
yeast.train <- pca.yeast.train[row.names(yeast.train),]
yeast.test <- pca.yeast.test[row.names(yeast.test),]
yeast <- pca.yeast.data[row.names(yeast),]

#######################################################################################################

# 5. Classification methods

# Prepare a crossvalidation 10x10 method to get the best model with
# several classifiers
k <- 10
CV.folds <- generateCVRuns(yeast.train$class, ntimes=1, nfold=k, stratified=TRUE)

# baseline: the error that we get predicting always the most probable class
(baseline <- 100*(1 - max(table(yeast$class))/nrow(yeast)))
# let's see if this can be improved using: 
# 1. Naive Bayes
# 2. LDA
# 3. QDA
# 4. KNN
# 5. PCA + Neural Networks
# 6. Random Forest

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
    else if (method == 'KNN'){
      tr.pred <- knn(tr[,-resp.var],tr[,-resp.var], cl = tr[,resp.var],k = 9)
      test.pred <- knn(tr[,-resp.var],yeast.train[va,-resp.var], cl = tr[,resp.var],k = 9)
      cv.results[j,"TR error"] <- 1 - sum(tr.pred == tr[,resp.var])/length(tr.pred)
      cv.results[j,"VA error"] <- 1 - sum(test.pred == yeast.train[va,resp.var])/length(test.pred)
      cv.results[j,"fold"] <- j
    }
    else if (method == "RandomForest"){
      tr <- which(!is.na(match(row.names(yeast.train),row.names(tr))))
      rf <- randomForest(formula = class ~., ntree = 400, data = yeast.train, subset = tr, xtest = yeast.train[va,-resp.var],
                         ytest = yeast.train[va,resp.var])
      cv.results[j,"TR error"] <- 1 - sum(diag(rf$confusion[,-11]) / sum(rf$confusion[,-11]))
      cv.results[j,"VA error"] <- 1 - sum(diag(rf$test$confusion[,-11]) / sum(rf$test$confusion[,-11]))
      cv.results[j,"fold"] <- j
    }
    else if (method == 'NeuralNetworks'){
      model.nnet <- nnet(class ~., data = tr, size=20, maxit=200, decay=0.001)
      # TR error
      p1 <- as.factor(predict (model.nnet, type="class"))
      t1 <- table(p1,tr$class)
      cv.results[j,"TR error"] <- 100*(1-sum(diag(t1))/dim(tr)[1])
      # TEST error
      p2 <- as.factor(predict (model.nnet, newdata=yeast.train[va,], type="class"))
      t2 <- table(p2,yeast.train[va,]$class)
      cv.results[j,"VA error"] <- 100*(1-sum(diag(t2))/dim(yeast.train[va,])[1])
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
# Test the behavior for QDA (only available using PCA beforehand)
Model.CV("QDA")
# Test the behavior for KNN
Model.CV("KNN")
# Test the behavior for Random Forest
Model.CV("RandomForest")
# Test the behavior for Neural Networks
Model.CV("NeuralNetworks")

# 6. Results: Which classification algorithm is the best? 

# Looking at the results of CV, LDA is the winner

# 7. Train the X algorithm with all the training data and test its behavior
# for predicting the test data (so as to tune its parameters and see its behavior
# when more data is used as training)

# First identify the response variable
resp.var <- which(colnames(yeast)=='class')

# Naive Bayes

yeast.nb <- naiveBayes (yeast.train$class ~ ., data=yeast.train, laplace=3)
# TR error
pred <- predict(yeast.nb,newdata=yeast.train)
(tt <- table(Truth=yeast.train$class, Predicted=pred))
(error_rate.train <- 100*(1-sum(diag(tt))/sum(tt)))
# TEST error
pred <- predict(yeast.nb,newdata=yeast.test)
(tt <- table(Truth=yeast.test$class, Predicted=pred))
(error_rate.test <- 100*(1-sum(diag(tt))/sum(tt)))
# Reduction of the error
100*(baseline-error_rate.test)/baseline
# Real error for each class
for (i in 1:length(levels(yeast$class))) {
  cat(levels(yeast.test$class)[i])
  print(1- tt[i,i]/sum(tt[i,]))
}

# LDA 

model <- lda(yeast.train$class ~ . ,data = yeast.train, CV=FALSE)
# TR error
pred <- predict(model, newdata = yeast.train)$class
(tt <- table(Truth=yeast.train$class, Predicted=pred))
(error_rate.train <- 100*(1-sum(diag(tt))/sum(tt)))
# TEST error
pred <- predict(model, newdata = yeast.test)$class
(tt <- table(Truth=yeast.test$class, Predicted=pred))
(error_rate.test <- 100*(1-sum(diag(tt))/sum(tt)))
# Reduction of the error
100*(baseline-error_rate.test)/baseline
# Real error for each class
for (i in 1:length(levels(yeast$class))) {
  cat(levels(yeast.test$class)[i])
  print(1- tt[i,i]/sum(tt[i,]))
}

# QDA

# Only works after feature selection (PCA), due to the fact that some modalities
# of yeast$class are too rare or appear so few times
# and detects colinearity, so can invert some covariance matrix

model <- qda(pca.yeast.train$class ~ . , data = pca.yeast.train, CV=FALSE)
# predict TR data
pred <- predict(model, pca.yeast.train)$class
(tt <- table(Truth=pca.yeast.train$class, Predicted=pred))
(error_rate.test <- 100*(1-sum(diag(tt))/sum(tt)))
# predict TEST data
pred <- predict(model, pca.yeast.test)$class
(tt <- table(Truth=pca.yeast.test$class, Predicted=pred))
(error_rate.test <- 100*(1-sum(diag(tt))/sum(tt)))
# Reduction of the error
100*(baseline-error_rate.test)/baseline
# Real error for each class
for (i in 1:length(levels(pca.yeast.test$class))) {
  cat(levels(pca.yeast.test$class)[i])
  print(1- tt[i,i]/sum(tt[i,]))
}

# KNN

# find the optimal number of neighbors
k <- 10
CV.folds <- generateCVRuns(yeast.train$class, ntimes=1, nfold=k, stratified=TRUE)
cv.results <- matrix (rep(0,4*k),nrow=k)
colnames (cv.results) <- c("k","fold","TR error","VA error")
cv.results[,"TR error"] <- 0
cv.results[,"VA error"] <- 0
cv.results[,"k"] <- k
resp.var <- which(colnames(yeast.train)=='class')
for (kn in 1:10) {
  for (j in 1:k) {
    va <- unlist(CV.folds[[1]][[j]])
    tr <- yeast.train[-va,]
    tr.pred <- knn(tr[,-resp.var],tr[,-resp.var], cl = tr[,resp.var],k = kn)
    test.pred <- knn(tr[,-resp.var],yeast.train[va,-resp.var], cl = tr[,resp.var],k = kn)
    cv.results[j,"TR error"] <- 1 - sum(tr.pred == tr[,resp.var])/length(tr.pred)
    cv.results[j,"VA error"] <- 1 - sum(test.pred == yeast.train[va,resp.var])/length(test.pred)
  }
  print(cv.results)
  print(mean(cv.results[,"VA error"]))
  cat('neighbors used ', kn)
}
# Optimal number of neighbors found: 9 
# Check the behavior of the algorithm once it is trained with all the training data and predicts all
# the test data
# TR error
predicted <- knn(yeast.train[,-resp.var], yeast.train[,-resp.var], cl = yeast.train[,resp.var], k=9)
(error_rate.train <- 1- sum(predicted == yeast.train[,resp.var])/length(predicted))
# TEST error
predicted <- knn(yeast.train[,-resp.var], yeast.test[,-resp.var], cl = yeast.train[,resp.var], k=9)
(error_rate.test <- 1- sum(predicted == yeast.test[,resp.var])/length(predicted))
# Reduction of the error
100*(baseline-error_rate.test)/baseline

# Neural Networks

# So as to optimize this model, we set a large enough number of Neurons (20) and we 
# regularize, so as to not overfit the data
trc <- trainControl (method="repeatedcv", number=10, repeats=10)
decays <- 10^seq(-3,0,by=0.2)
model.10x10CV <- train (class ~., data = yeast, subset=learn, method='nnet', maxit = 200, trace = FALSE,
                        tuneGrid = expand.grid(.size=20,.decay=decays), trControl=trc)
## We can inspect the full results
model.10x10CV$results
## and the best model found
model.10x10CV$bestTune
# Fit the best model with all the training and check the behavior of the
# algorithm once it is trained with all the training data and predicts all the test data
model.nnet <- nnet(class ~., data = yeast, subset=learn, size=20, maxit=200, decay=0.001)
# TR error
p1 <- as.factor(predict (model.nnet, type="class"))
t1 <- table(p1,yeast.train$class)
(error_rate.learn <- 100*(1-sum(diag(t1))/dim(yeast.train)[1]))
# TEST error
p2 <- as.factor(predict (model.nnet, newdata=yeast.test, type="class"))
t2 <- table(p2,yeast.test$class)
(error_rate.test <- 100*(1-sum(diag(t2))/dim(yeast.test)[1]))

# Random Forest

rf <- randomForest(formula = class ~., data = yeast, subset = learn, xtest = yeast.test[,-which(colnames(yeast)=='class')],
                   ytest = yeast.test[,which(colnames(yeast)=='class')])
print(rf)



