####################################################################
# Machine Learning - MIRI Master
# Lluís A. Belanche

# LAB 5: LDA/QDA/RDA, kNN, Naïve Bayes (Part1)
# version of March 2016
####################################################################

####################################################################
#### Starter: the multivariate Gaussian distribution
####################################################################

## Create a positive-definite (PD) covariance matrix 

(Sigma <- matrix(c(4,-3,-3,9),2,2))

# We now seek to find a matrix M such that M·M^T = Sigma
# (recall M^T denotes the transpose of M and '·' denotes standard matrix product)

# The Cholesky decomposition of a symmetric PD matrix A is a decomposition of the form

# A = L·L^T 

# here L is a lower triangular matrix with real and positive diagonal entries

# Every real symmetric PD matrix has a unique Cholesky decomposition: there is only one lower triangular matrix L with strictly positive diagonal entries such that A = L·L^T

(M <- t(chol(Sigma)))

# sanity check

M %*% t(M)

## Why do we learn this? Because it is a simple way to generate observations from a multivariate Gaussian distribution (MVGD)

mu <- c(2,-2)

# If Z is a random vector and M is a matrix of numbers, then the covariance matrix of M·Z is M·cov(Z)·M^T. 

# To simulate vectors from a general MVGD, we start by sampling from a MVGD whose mean vector is the origin with a covariance matrix equal to the identity matrix, and we get Z

# Then we obtain an observation from a MVGD with covariance matrix Sigma by multiplying Z by the Cholesky decomposition of the desired covariance matrix. Finally we just add the desired vector of means

## Let illustrate this by creating a dataset with N=200 observations from a MVGD of our choice:

N <- 200

Z <- matrix(rnorm(2*N),2,N) # 2 rows, N columns

X <- t(M %*% Z + mu)

# The transpose above is taken so that X becomes a Nx2 matrix, since R "prefers" to have the vector components as columns

# This is our data:
par(mfrow=c(1,1))

plot(X, xlab="X1", ylab="X2")
abline(v=mean(X[,1]),lty=2)
abline(h=mean(X[,2]),lty=2)

# and its sample covariance matrix (compare it with Sigma)

(S <- cov(X))

# and its sample mean (compare it with mu)

apply(X,2,mean)

## In R there are nice routines for doing this directly, e.g. mvrnorm() in {MASS}

## This function produces an i.i.d. sample from the specified MVGD. 
## For instance, we can generate another, larger, sample from the same bivariate normal above:

library(MASS)

gauss2 <- mvrnorm (10000, mu = mu, Sigma = Sigma)

# now we can do a kernel density estimate (not seen in class)
gauss2.kde <- kde2d(gauss2[,1], gauss2[,2], n = 50)

# fancy contour
image(gauss2.kde)
contour(gauss2.kde, add = T)

# static 3D plot
persp(gauss2.kde, phi = 3, theta = 10)

#### which can be compared with the theoretical plot for this MVGD
library(ellipse)

plot(0,0, type="n", xlim=range(gauss2[,1]), ylim=range(gauss2[,2]), xlab="X1", ylab="X2")

for (l in seq(0.1,0.9, length.out=7))
  lines(ellipse(Sigma, centre=mu, level=l), col='blue', lwd=2)

points(mu[1],mu[2], pch=16,col='blue')

####################################################################
## Example 1: Data reduction with LDA in an artificial problem
####################################################################

## Linear discriminant analysis (LDA) is a method that finds a linear combination of features to project or separate two or more classes of objects. 

## This combination may be used as a linear classifier or for dimensionality reduction
## (the latter use is not explicitly seen in our lectures, but we will cover it here)

## When used for low-dimensional projection, LDA is known as FDA (Fisher's discriminant analysis)

## If your goal is to perform linear dimensionality reduction for class discrimination, you should prefer FDA to PCA; PCA is useful for signal representation (but not necessarily for discrimination)

## Sigma must be a PD symmetric matrix specifying the covariance matrix of the variables

N <- 1000

(Sigma <- matrix(data=c(1,0.3,0.3,1),nrow=2,ncol=2,byrow=TRUE))

# these are the eigenvalues (both should positive):

eigen (Sigma, only.values=TRUE)$values

# Let's create class 1 ('red' class)

mean.1 <- matrix(c(2,0),nrow=2,ncol=1)                         
 
X.red <- mvrnorm(N,mu=mean.1,Sigma=Sigma)

# Let's create class 2 ('green' class)

mean.2 <- -mean.1

X.green <- mvrnorm(N,mu=mean.2,Sigma=Sigma)

par(mfrow=c(2,2))

plot(c(X.red[,1],X.green[,1]), c(X.red[,2],X.green[,2]), 
     col=c(rep('red',N),rep('green',N)), main="Artificial data", xlab="X1", ylab="X2")

## Since both classes share the same covariance matrix, we can compute the theoretical Bayes (true) error (the formula is found in textbooks, but was not given in class)

# The following function computes the true probability of error for two-class normally distributed features with equal covariance matrices and unequal class priors Pw1, Pw2

prob.error <- function (Pw1, Pw2, Sigma, Mu1, Mu2)
{
  stopifnot (Pw2+Pw1==1,Pw2>0,Pw1>0,Pw2<1,Pw1<1)
  alpha <- log(Pw2/Pw1)
  D <- quad.form.inv (Sigma, Mu1-Mu2)
  A1 <- (alpha-D/2)/sqrt(D)
  A2 <- (alpha+D/2)/sqrt(D)
  Pw1*pnorm(A1)+Pw2*(1-pnorm(A2))
}

# Numerically correct way for t(x) %*% solve(M) %*% (x)
# that is, for the quadratic form x^T M^-1 x
quad.form.inv <- function (M, x)
{
  drop(crossprod(x, solve(M, x)))
} 

# there you are ...

prob.error (0.5,0.5,Sigma,mean.1,mean.2)

# Now we put both classes one after the other and create a dataframe

d <- data.frame(c(rep(1,N),rep(2,N)), c(X.red[,1], X.green[,1]), c(X.red[,2], X.green[,2]))

colnames(d) <- c("target", "X1", "X2")
d$target <- as.factor(d$target)
summary(d)

# call to LDA
myLDA <- lda(d[c(2,3)],d[,1])

# Now we show the best projection direction on the original space. This direction maximizes the separability of the classes. For that, we first need the slope:

LDAslope <- myLDA$scaling[2]/myLDA$scaling[1]

# And now we can perform the visualization:

plot(c(X.red[,1],X.green[,1]), c(X.red[,2],X.green[,2]), col=c(rep('red',N),rep('green',N)),
     main="Artificial data using LDA", xlab="X1", ylab="X2")

abline(0,LDAslope,col='black',lwd=2)

# We can also compute the projections of the two classes

myLDA.proj <- d[,2] * myLDA$scaling[1] + d[,3] * myLDA$scaling[2]

plot(myLDA.proj, c(rep(0,N),rep(0,N)), col=c(rep('green',N),rep('red',N)),
     main='Projection as seen in 1D', xlab="Discriminant", ylab="")

# To understand what is going on, do:

myLDA

# of which ...

myLDA$scaling

# ... are the coefficients of the linear discriminant. The discriminant is linear because both classes share the same covariance matrix Sigma.

## Now with LDA we are able to project the data into the direction that maximizes (linear) separability:

# projection(X) = X1*myLDA$scaling[1] + X2*myLDA$scaling[2]

# ... which is what the previous code does. 

# If we now use the projected data, we can classify new points by projecting them and then computing the distance to the nearest (projected) mean of each class. 

# The overall error would be 1.8%, as computed above.


####################################################################
## Example 2: Visualizing and classifying crabs with LDA
####################################################################

# Campbell studied rock crabs of the genus "Leptograpsus" in 1974. One
# species, Leptograpsus variegatus, had been split into two new species,
# previously grouped by colour (orange and blue). Preserved specimens
# lose their colour, so it was hoped that morphological differences
# would enable museum material to be classified.

# Data is available on 50 specimens of each sex of each species (so 200 in total),
# collected on sight at Fremantle, Western Australia. Each specimen has
# measurements on: the width of the frontal lobe (FL), the rear width (RW),
# the length along the carapace midline (CL), the maximum width (CW) of 
# the carapace, and the body depth (BD) in mm, in addition to
# colour (that is, species) and sex.

## the crabs data is also in the MASS package
data(crabs)

## look at data
?crabs
summary(crabs)
head(crabs)

## The goal is to separate the 200 crabs into four classes, given by the 2x2 configurations for sex (M/F) and species (B/O)

Crabs.class <- factor(paste(crabs[,1],crabs[,2],sep=""))

# Now 'BF' stands now for 'Blue Female', and so on
table(Crabs.class)

## using the rest of the variables as predictors (except 'index', which is only an index)
Crabs <- crabs[,4:8]

summary(Crabs)

## Various preliminary plots (notice all 5 predictors are continuous)

par(mfrow=c(1,1))
boxplot(Crabs)

hist(Crabs$FL,col='red',breaks=20,xlab="", main='Frontal Lobe Size (mm)')
hist(Crabs$RW,col='red',breaks=20,xlab="", main='Rear Width (mm)')
hist(Crabs$CL,col='red',breaks=20,xlab="", main='Carapace Length (mm)')
hist(Crabs$CW,col='red',breaks=20,xlab="", main='Carapace Width (mm)')
hist(Crabs$BD,col='red',breaks=20,xlab="", main='Body Depth (mm)')

## We would like first to have a look at the data in 2D:
## These are scatterplots with color coding by class

pairs(Crabs, main="Pairs plot for the crabs", pch=21, bg=c('black', 'red', 'green', 'blue')[Crabs.class]) 

## we can make our own 'pairs plot' by doing:

plot(Crabs,col=unclass(Crabs.class))

## The partimat() function in the 'klaR' package can display the results 
## of linear (or quadratic) classification using two variables at a time.

library(klaR)

# since there 4 classes, we get 4 regions; notice the "app. error rate" at the top of each plot, which is the training (apparent) error

partimat (x=Crabs, grouping=Crabs.class, method="lda")

# These plots suggest that FL and RW is the best pair of variables

## Now let's classify: we fit LDA to the data using all predictors

(lda.model <- lda (x=Crabs, grouping=Crabs.class))

plot(lda.model)

## As there are four classes (called 'groups' in LDA), we get three linear discriminants (LDs) for projection
## We first compute the loadings (the 'loadings' are simply the projected data)

## This time we do it more generally, using matrix multiplication

loadings <- as.matrix(Crabs) %*% as.matrix(lda.model$scaling)

## Now we plot the projected data into the first two LDs (notice that with LDA we are actually performing dimensionality reduction 5D --> 3D (always number of classes minus 1), out of which we are plotting the 2 most important dimensions

# We do our own plotting method, with color and legend:

colors.crabs <- c('blue', 'lightblue', 'orange', 'yellow')

crabs.plot <- function (myloadings)
{
  plot (myloadings[,1], myloadings[,2], type="n", xlab="LD1", ylab="LD2")
  text (myloadings[,1], myloadings[,2], labels=crabs$index, col=colors.crabs[unclass(Crabs.class)], cex=.55)
  legend('topright', c("B-M","B-F","O-M","O-F"), fill=colors.crabs, cex=.55)
}

crabs.plot (loadings)

# The result is quite satisfactory, right? We can see that the 5 continuous predictors do indeed represent 4 different crabs. 

# We can also see that crabs of the Blue "variety" are less different 
# (regarding males and females) than those in the Orange variety

## If you would like to keep this new representation for later use (maybe to build a classifier on it), simply do:

Crabs.new <- data.frame (New.feature = loadings, Target = Crabs.class)

summary(Crabs.new)

## Now let's analyze the numerical output of lda() in more detail:

lda.model

# "Prior probabilities of groups" is self-explanatory (these are estimated from the data, but can be overriden by the 'prior' parameter)

# "Group means" is also self-explanatory (these are our mu's)

# "Coefficients of linear discriminants" are the scaling factors we have been using to project data. These have been normalized so that the within-groups covariance matrix is spherical (a multiple of the identity). 

# This means that the larger the coefficient of a predictor,
# the more important the predictor is for the discrimination:

lda.model$scaling

# We can interpret our plot so that the horizontal axis (LD1) separates the groups mainly by using FL, CW and BD;
# the vertical axis (LD2) separates the groups mainly by using RW and some CL, etc

## The "Proportion of trace" is the proportion of between-class variance that is explained by successive discriminants (LDs)

# For instance, in our case LD1 explains 68.6% of the total between-class variance

## In this case, the first two LDs account for 98.56% of total between-class variance, fairly close to 100%

## This means that the third dimension adds but a little bit of discriminatory information. Let's visualize the crabs in 3D:

library(rgl)

# 3d scatterplot (that can be rotated)
plot3d(loadings[,1], loadings[,2], loadings[,3], "LD1", "LD2", "LD3",
       size = 4, col=colors.crabs[unclass(Crabs.class)], main="Crabs Data")

## As the measurements are lengths, it could be sensible to take logarithms

(lda.logmodel <- lda (x=log(Crabs), grouping=Crabs.class))

## The model looks a bit better, given that he first two LDs now account for 99.09% of total between-class variance, very good indeed

## As an example, the first (log) LD is given by:
# LD1 = -31.2*log(FL) - 9.5*log(RW) - 9.8*log(CL) + 66*log(CW) - 18*log(BD)

## get the new loadings

logloadings <- as.matrix(log(Crabs)) %*% as.matrix(lda.logmodel$scaling)

## plot the projected data in the first two LDs

crabs.plot (logloadings)

## The first coordinate clearly expresses the difference between species, and the second the difference between sexes!

# and finally a 3d scatterplot (that can be rotated)
plot3d(logloadings[,1], logloadings[,2], logloadings[,3], "LD1", "LD2", "LD3",
       size = 4, col=colors.crabs[unclass(Crabs.class)], main="Crabs Data (log)")

## Now let us evaluate the model as a predictor (i.e., as a classifier)

## We can obviously compute the training error (also known as "apparent" or "resubstitution" error), which is usually optimistic! (i.e., biased downwards)

(ct <- table(Truth=Crabs.class, Pred=predict(lda.logmodel, log(Crabs))$class))

# percent by class
diag(prop.table(ct, 1))
# total percent correct
sum(diag(prop.table(ct)))

## If we want to reallly assess the accuracy of the prediction we have to use LOOCV (leave-one-out cross-validation)

## If CV=TRUE, the lda() method returns interesting results: the predicted classes and posterior probabilities for the LOOCV

## WARNING: the default is LOOCV=FALSE (we should use it for plotting only)

lda.logmodel <- lda (x=log(Crabs), grouping=Crabs.class, prior = c(1,1,1,1)/4, CV=TRUE)

## the list 'class' gives the predicted classes
lda.logmodel$class[1:10]

## Note the prior probabilities can be changed at will; if unspecified, proportions from the supplied data will be used

# percent correct for each class
(ct <- table(Truth=Crabs.class, Pred=lda.logmodel$class))

diag(prop.table(ct, 1))
# total percent correct
sum(diag(prop.table(ct)))

## This is a much more reliable figure, because it is a predictive error

## Of special interest is the full set of posterior probabilities (make sure you understand what you get):

lda.logmodel$posterior[1:10,]

## Pearson's Chi-squared Test is useful for goodness-of-fit tests

chisq.test (lda.logmodel$class, Crabs.class)

## This test checks whether two categorical distributions coincide
## In our case, we take them to be the class labels and the predicted labels. The lower the value of the 'X-squared' statistic, the better.

## The test is accepted (actually, not rejected) when p-value < 0.05. So in this case there is strong evidence that our predictions are not random.

## Final note 1: the argument na.omit() in lda() omits any rows having one or more missing values (the default action is for the procedure to fail)

## Final note 2: sadly there is no provision for costs (losses), as seen in class :-(

####################################################################
## Example 3: Visualizing and classifying wines with LDA and QDA
####################################################################

## We have the results of an analysis on wines grown in a region in Italy but derived from three different cultivars.

## The analysis determined the quantities of 13 chemical constituents found in each of the three types of wines. 

## The goal is to separate the three types of wines:

wine <- read.table("wine.data", sep=",", dec=".", header=FALSE)

dim(wine)

colnames(wine) <- c('Wine.type','Alcohol','Malic.acid','Ash','Alcalinity.of.ash','Magnesium','Total.phenols','Flavanoids','Nonflavanoid.phenols','Proanthocyanins','Color.intensity','Hue','OD280.OD315','Proline')

wine$Wine.type <- factor(wine$Wine.type, labels=c("Cultivar.1","Cultivar.2","Cultivar.3"))

summary(wine)

plot(subset(wine,select=-Wine.type),col=unclass(wine$Wine.type))

## For this example let's practice a different call mode to lda(), using a formula; this is most useful when our data is in a dataframe format:

(lda.model <- lda (Wine.type ~ ., data = wine))

## We can see that neither Magnesium or Proline seem useful to separate the wines; while
## Flavanoids and Nonflavanoid.phenols do. Ash is mainly used in the LD2.

## Plot the projected data in the first two LDs
## We can see that the discrimination is very good

plot(lda.model)

# Alternatively, again we can do it ourselves, with more control on color and text (wine number):

wine.pred <- predict(lda.model)
plot(wine.pred$x,type="n")
text(wine.pred$x,labels=as.character(rownames(wine.pred$x)),col=as.integer(wine$Wine.type), cex=.75)
legend('bottomright', c("Cultivar 1","Cultivar 2","Cultivar 3"), lty=1, col=c('black', 'red', 'green'))

# If need be, we can add the (projected) means to the plot

plot.mean <- function (mywine)
{
  m1 <- mean(subset(wine.pred$x[,1],wine$Wine.type==mywine))
  m2 <- mean(subset(wine.pred$x[,2],wine$Wine.type==mywine))
  print(c(m1,m2))
  points(m1,m2,pch=16,cex=1.5,col=as.integer(substr(mywine, 10, 10)))
}

plot.mean ('Cultivar.1')
plot.mean ('Cultivar.2')
plot.mean ('Cultivar.3')

# indeed classification is perfect

table(Truth=wine$Wine.type, Pred=wine.pred$class)

# Let us switch to leave-one-out cross-validation (notice the use of update())

lda.model.loocv <- update(lda.model, CV=TRUE)
head(lda.model.loocv$posterior)

(ct <- table(Truth=wine$Wine.type, Pred=lda.model.loocv$class))

# 2 mistakes (on 178 observations): 1.12% error

chisq.test(ct) # indeed a very good model

## Quadratic Discriminant Analysis is the same, replacing 'lda' by 'qda'
## problems may arise if for some class there are less (or equal) observations than dimensions (this is not the case for our wine data)

qda.model <- qda (Wine.type ~ ., data = wine)

qda.model

## There is no projection this time (because projection is a linear operator and the QDA boundaries are quadratic ones)

# Let's have a look at classification using LOOCV

qda.model.loocv <- qda (Wine.type ~ ., data = wine, CV=TRUE)

head(qda.model.loocv$posterior)

(ct <- table(Truth=wine$Wine.type, Pred=qda.model.loocv$class))

# 1 mistake (on 178 observations): 0.56% error

chisq.test(ct) # a slightly better model

# It would be nice to find out which is the "stubborn" wine: it is a wine of 'Cultivar.2' classified as 'Cultivar.1'. Maybe there is something strange with this wine ...

## Finally, for the sake of illustration, let's use RDA:

# The rda() function is in the 'klaR' package, which was already loaded

# The best (lambda, gamma) are chosen by 10-CV; we change it to 20-CV to make it closer to LOOCV

set.seed(1)
(rda.model.cv <- rda (Wine.type ~ ., data = wine, fold=20, prior=1))

# What we get is essentially lda in this case (lambda=0, gamma=1)

###############################################################
# Example 4: k-Nearest Neighbor analysis in an artificial problem
###############################################################

## k-NN can be found in several places in R packages; one of them is in the "class" package

library(class)

## CAREFUL! use it correctly! train and test cannot intersect!

## Usage:

# knn (train, test, cl, k = 1, l = 0, prob = FALSE, use.all = TRUE)

# This function predicts 'test' data; for each observation in 'test', looks for the k-nearest neighbors in 'train' (using plain Euclidean distance).

# The classification is decided by majority vote, with ties broken at random (but if there are ties for the k-th nearest vector, all candidates are included in the vote)

# 'cl' is the vector of class labels for the observations in 'train'
# If data is large, and k is not small, I would recommend to use prob = TRUE to get some estimations of posterior probabilities

s <- sqrt(1/4)
set.seed(1234)

generate <- function (M, n=100, Sigma=diag(2)*s) 
{
  z <- sample(1:nrow(M), n, replace=TRUE)
  t(apply(M[z,],1, function(mu) mvrnorm(1,mu,Sigma)))
}

# generate 10 means in two dimensions
M0 <- mvrnorm(10, c(1,0), diag(2))

# generate data out of M0
x0 <- generate(M0)

# repeat with M1
M1 <- mvrnorm(10, c(0,1), diag(2))
x1 <- generate(M1)

# Bind them together (by rows)
train <- rbind(x0, x1)
(N <- dim(train)[1])

# generate class labels in {0,1}
t <- c(rep(0,100), rep(1,100))

# Now generate a huge test data using a grid in the correct range

grid.size <- 100
XLIM <- range(train[,1])
grid.x <- seq(XLIM[1], XLIM[2], len=grid.size)

YLIM <- range(train[,2])
grid.y <- seq(YLIM[1], YLIM[2], len=grid.size)

test <- expand.grid(grid.x,grid.y)
dim(test)

# Let's visualize 1-NN (only 1 neighbor) in action

par(mfrow=c(1,1))

nicecolors <- c('black','red')

predicted <- knn (train, test, t, k=1)

# These are the predictions
plot(train, xlab="X1", ylab="X2", xlim=XLIM, ylim=YLIM, type="n")
points(test, col=nicecolors[as.numeric(predicted)], pch=".")
contour(grid.x, grid.y, matrix(as.numeric(predicted),grid.size,grid.size), 
        levels=c(1,2), add=TRUE, drawlabels=FALSE)

# Add training points, for reference
points(train, col=nicecolors[t+1], pch=16)
title("1-NN classification")

## In order to see the effect of a different number of neighbors, let's do something nice:

par(mfrow=c(2,3))

for (myk in c(1,3,5,7,10,round(sqrt(N))))
{
  predicted <- knn(train, test, t, k=myk)
    
  plot(train, xlab="X1", ylab="X2", xlim=XLIM, ylim=YLIM, type="n")
  points(test, col=nicecolors[as.numeric(predicted)], pch=".")
  contour(grid.x, grid.y, matrix(as.numeric(predicted),grid.size,grid.size), 
          levels=c(1,2), add=TRUE, drawlabels=FALSE)
  
  # add training points, for reference
  points(train, col=nicecolors[t+1], pch=16)
  title(paste(myk,"-NN classification",sep=""))
}
     
# Possibly you have to "Zoom" the plot to see it properly

# This maximum value (sqrt of training set size) for the number of nearest neighbors is a popular rule of thumb ... but I do not have an explanation for it
# Folklore also says that it should be a prime number ...

# Now we can illustrate the method on a real problem: we use the previous wine data for comparison

## First setup a k-NN model with 3 neighbours
## Notice there is no "learning" ... the data is the model (just test!)

wine.data <- subset (wine, select = -Wine.type)

# Very crude way of performing LOOCV for k-NN :-)

knn.preds <- rep(NA, nrow(wine.data))

for (i in 1:nrow(wine.data))
{
  knn.preds[i] <- knn (wine.data[-i,], wine.data[i,], wine$Wine.type[-i], k = 3) 
}

(tab <- table(Truth=wine$Wine.type, Preds=knn.preds))
1 - sum(tab[row(tab)==col(tab)])/sum(tab)

## As usual, rows are true targets, columns are predictions (may I suggest that you adhere to this convention too)

## One could also use the function 'knn1()' when k=1 (just one neighbour)

## How do we optimize k? One way is by using the LOOCV

# Actually there is an implementation of LOOCV for k-NN:

myknn.cv <- knn.cv (wine.data, wine$Wine.type, k = 3)

(tab <- table(Truth=wine$Wine.type, Preds=myknn.cv))
1 - sum(tab[row(tab)==col(tab)])/sum(tab)

# The results may not fully coincide by the random tie-breaking mechanism

## Let's loop over k using a function
set.seed (6046)

# Since sqrt(178) is approx 13.34, we take 13 as the max number of neighbours

N <- nrow(wine)
neighbours <- 1:sqrt(N)

loop.k <- function (mydata, mytargets, myneighbours)
{
  errors <- matrix (nrow=length(myneighbours), ncol=2)
  colnames(errors) <- c("k","LOOCV error")

  for (k in myneighbours)
  {
    print(k)
    myknn.cv <- knn.cv (mydata, mytargets, k = myneighbours[k])
  
    # fill in number of neighbours and LOOCV error
    errors[k, "k"] <- myneighbours[k]
  
    tab <- table(Truth=mytargets, Preds=myknn.cv)
    errors[k, "LOOCV error"] <- 1 - sum(tab[row(tab)==col(tab)])/sum(tab)
  }
  errors
}


par(mfrow=c(1,1))

plot(loop.k (wine.data, wine$Wine.type, neighbours), type="l", xaxt = "n")
axis(1, neighbours)

## It seems that kNN does a pretty bad job here; 1-NN is the best choice but the model is terrible, compared to that of LDA/QDA

## k-NN normally benefits from standardizing the variables:

plot(loop.k (scale(wine.data), wine$Wine.type, neighbours), type="l", xaxt = "n")
axis(1, neighbours)

# ... which is no true in this case

## A good idea would be to use the previously computed LDs

lda.model <- lda (Wine.type ~ ., data = wine)
loadings <- as.matrix(wine.data) %*% as.matrix(lda.model$scaling)

## Let's repeat the loop over k
set.seed (6046)

plot(loop.k (loadings, wine$Wine.type, neighbours), type="l", xaxt = "n")
axis(1, neighbours)

# So we would keep 6 neighours

# Notice that the tested values for k need not be consecutive; in a large dataset, this would be very time-consuming; also we would not use LOOCV for the same reason, but rather 10-CV

####################################################################
# Example 5: The Naïve Bayes classifier with real data
####################################################################

library (e1071)

## Naive Bayes Classifier for discrete predictors; we use the 
## 1984 United States Congressional Voting Records

## This data set includes votes for each of the U.S. House of Representatives Congressmen on 16 key votes

## In origin they were nine different types of votes: 

# 1) 'voted for', 'paired for', and 'announced for' (these three simplified to yea or 'y')
# 2) 'voted against', 'paired against', and 'announced against' (these three simplified to nay or 'n')
# 3) voted 'present', voted 'present to avoid conflict of interest', and 'did not vote or otherwise make a position known' (these three simplified to 'undefined' and marked as NA in the data)

## The goal is to classify U.S. Congressmen as either 'Republican' or 'Democrat' based on their voting profiles,

## This is not immediate, because in the U.S., Congressmen have a large freedom of vote (unlike other democracies)

## The vote is obviously linked to their party's directions but ultimately guided by their own feelings, interests and compromises

data (HouseVotes84, package="mlbench") 

## add meaningful names to the votes

colnames(HouseVotes84) <- 
      c("Class","handicapped.infants","water.project.sharing","budget.resolution",
        "physician.fee.freeze","el.salvador.aid","religious.groups.in.schools",
        "anti.satellite.ban", "aid.to.nicaraguan.contras","mx.missile",
        "immigration","synfuels.cutback","education.spending","superfund","crime",
        "duty.free.exports","export.South.Africa")

summary(HouseVotes84)

set.seed(1111)

N <- nrow(HouseVotes84)

## We first split the available data into learning and test sets, selecting randomly 2/3 and 1/3 of the data
## Again we do this for a honest estimation of prediction performance

learn <- sample(1:N, round(2*N/3))

nlearn <- length(learn)
ntest <- N - nlearn

# First we build a model using the learn data

model <- naiveBayes(Class ~ ., data = HouseVotes84[learn,])

# we get all the probabilities (so 'eager' mode)
model

# predict the outcome of the first 10 Congressmen
predict(model, HouseVotes84[1:10,-1]) 

# same but displaying posterior probabilities
predict(model, HouseVotes84[1:10,-1], type = "raw") 

# compute now the apparent error
pred <- predict(model, HouseVotes84[learn,-1])

# form and display confusion matrix & overall error
(tab <- table(Truth=HouseVotes84[learn,]$Class, Preds=pred) )

1 - sum(tab[row(tab)==col(tab)])/sum(tab)

# compute the test (prediction) error
pred <- predict(model, newdata=HouseVotes84[-learn,-1])

# form and display confusion matrix & overall error
(tab <- table(Truth=HouseVotes84[-learn,]$Class, Preds=pred) )

1 - sum(tab[row(tab)==col(tab)])/sum(tab)

# These politicians are so predictable ... even American ones

# Note how most errors (9/12) correspond to democrats wrongly predicted as republicans

## In the event of empty empirical probabilities, this is how we would setup Laplace correction:

model <- naiveBayes(Class ~ ., data = HouseVotes84[learn,], laplace = 1)
