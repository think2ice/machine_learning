####################################################################
# Machine Learning - MIRI Master
# author: 
# think_2ice <- Manel Alba

# LAB 5: LDA/QDA/RDA, kNN, Naïve Bayes (Part2)
# version of March 2016
####################################################################

## Since Part1 was a bit heavy, you will now have the opportunity to choose the exercise you like the most, out of three possiblities.

## Obviously, if you want to practice more, you can workout more than one (this makes a lot of sense because they are quite different).

## Note that, in all cases, you must calculate the missclassification rate (error) for your models in the learn data using LOO cross-validation (LOOCV), both to optimize a parameter (as 'laplace' in Naïve Bayes or 'k' in kNN) and to choose the best model (as the one with lowest LOOCV error). Then you have to refit your model using the selected parameter in the whole learning data (this time without LOOCV) and finally report the error in the leftout data (test data)

#######################
## Exercise 1
#######################

## This exercise involves the use of the 'Glass' data set, a real-world data set about 
## fragments of glass collected in forensic work.

# Each record has a measured refractive index for the glass and its composition:
# percentages by weight of oxides of Na, Mg, Al, Si, K, Ca, Ba and Fe

# The fragments were originally classed as seven types, one of which ('sand') 
# was absent in this dataset. The categories which occur are window float glass (70),
# window non-float glass (76), vehicle window glass (17), containers (13),
# tableware (9) and vehicle headlamps (29). 

# The data frame contains the following columns:

# RI refractive index
# Na sodium
# Mg manganese
# Al aluminium
# Si silicon
# K potassium
# Ca calcium
# Ba barium
# Fe iron
# type of glass (target variable)

library(MASS)
data (fgl)

# warning! 6 classes for only 214 observations overall! Difficult modelling problem ahead ... and some classes are under-represented

# so this is your departing data set

summary(fgl)
attach(fgl)

## Some basic discrimination ability is apparent in the plots:
## e.g. headlamp glass is high in Barium, Sodium and Aluminium but low in Iron:

par(mfrow=c(3,3))
my.glass.boxplot <- function (var) 
{ boxplot (var ~ type, data=fgl, col = "bisque"); title(deparse(substitute(var))) }

my.glass.boxplot (RI)
my.glass.boxplot (Na)
my.glass.boxplot (Mg)

my.glass.boxplot (Al)
my.glass.boxplot (Si)
my.glass.boxplot (K)

my.glass.boxplot (Ca)
my.glass.boxplot (Ba)
my.glass.boxplot (Fe)

## Assignment for this exercise
## Do the following (in separate sections)

# 1. Derive some "hand discrimination" rules by looking at the boxplots (just for fun)
# 2. Perform various preliminary plots (notice all predictors are continuous), with histograms
# 3. Perform scatter plots (pairs), with glass type overlayed (in color)
# 4. Fit LDA to the data using all predictors
# 5. Plot the data into the first two LDs; what do you see? What is the proportion of trace?
#    Repeat with the first three LDs in 3D
# 6. Compute resubstitution error and leave-one-out cross-validation error (LOOCV)
# 7. Perform Pearson's Chi-squared Test to see if your best model is better than random (p-value < 0.05)

# 8. Try to fit QDA to the data using all predictors ... what happens? what is the reason? 
#    There are two possible solutions:
#    a) try to remove one of the classes (the less represented one)
#       Hint code for doing this:

#         fgl2 <- fgl[fgl[,10] != "Tabl",]
#         fgl2$type <- factor(fgl2$type, levels=levels(fgl$type)[-5]
#         qda.model <- qda (x=fgl2[,-10],grouping=fgl2$type,CV=FALSE)
#    b) merge the less represented class with another one; which one would you choose? do the merging!
#    c) use RDA (recommended)

# 9. Compute resubstitution error and LOOCV error for your final QDA; does it improve over LDA?

# 10. Apply kNN by optimizing the number of nearest neighbors
# 11. Apply Naïve Bayes

# 12. Choose your final model as the one with lowest LOOCV. Refit it in the learning (or training data) and make it predict the leftout test data. Discuss the results (have a look at the confusion matrix)


#######################
## Exercise 2
#######################

## This exercise involves the use of the 1984 U.S. Congressional Voting Records
## The thing is that there are some missing values in the data set ...

sum(complete.cases(HouseVotes84))

## Recall there are 435 rows ...
## so more than half of the rows contain at least one missing value

## These are the percentages per variable
sort(round(100*apply(is.na(HouseVotes84), 2, mean),2), decreasing=TRUE)

## In Part1 we used Naïve Bayes for predicting the political party. If NAs are found, 
## the default action is not to count them for the computation of the probabilities

## These NAs correspond to "undefined vote", which is not necessarily something missing (the person was there, and emitted a vote)

## An alternative is to recode them and treat them as a further value; this could make sense in this dataset because the Congressmen declared that they did not want to 'announce their vote yet' ...

## ... which in turn could say something about their political position in some cases


## First we declare a further modality: "u" for "undefined"
HouseVotes84[2:17] <- lapply(HouseVotes84[2:17], function(x) {factor(x,levels = c(levels(x), "u"))})

## Then we recode all missing values to the new value "u"
HouseVotes84[is.na(HouseVotes84)] <- "u"

# So this is your departing data set

summary(HouseVotes84) 

## Assignment for this exercise
## Do the following (in separate sections)

# 1. Fit a Naïve Bayes classifier to this new data set and compare the results to those in Part1
# 2. If you experience problems with empty empirical probabilities (note there are just a few "u" for some variables), you may use Laplace correction
# 3. Decide which is your best model and give a final prediction in the test set. Does it improve over that in Part1?

#######################
## Exercise 3
#######################

## This exercise involves the use of the Vowel data set: speaker-independent recognition
## of the eleven steady-state vowels of British English

## As the name implies, the task is to distinguish between 11 types of vowels
## The available variables are derived using speech recognition techniques

library(ElemStatLearn)
data(vowel.train) 

## Recode the target as factor:

colnames(vowel.train)[1] <- "Vowel"
vowel.train$Vowel <- factor(vowel.train$Vowel)

table (vowel.train$Vowel)

summary(vowel.train)

## Let's start with a couple of humble scatterplots ... first we build a handy plotter:

## Plots data for vowels v1 and v2 together with class means
plot.vowels <- function (vdata,v1,v2)
{
  par(mfrow=c(1,1), mgp=c(1.25,.5,0), mar=c(2.25, 2.1, 1,1))
  
  plot(vdata[,v1], vdata[,v2], type="n", xlab=v1, ylab=v2)
  text(vdata[,v1], vdata[,v2], as.character(vdata$Vowel), col=mycols2[vdata$Vowel], cex=0.8)
  means <- apply(vdata[c(v1,v2)], 2, function(x) tapply(x, vdata$Vowel, mean, simplify=TRUE))
  points(means, col=mycols2, pch=16, cex=1.6)
}

library(RColorBrewer)

mycols2 <- brewer.pal(11, "Paired")

## not bad, but a lot of overlap
plot.vowels (vowel.train,"x.1","x.2")

## buf ...
plot.vowels (vowel.train,"x.9","x.10")

## On what grounds can one pretend to get even so-so results with such a difficult task? 

## Let's listen to the masters:

# "LDA frequently achieves good performance in the tasks of face and object recognition, even though the assumptions of common covariance matrix among groups and normality are often violated (Duda, et al., 2001)"

## Assignment for this exercise
## Do the following (in separate sections)

# 1. Play with some initial plots to get an idea of the problem
# 2. Perform various preliminary plots (notice all predictors are continuous), with histograms
# 3. Fit LDA to the data using all predictors
# 4. Plot the data into the first two LDs; what do you see? What is the proportion of trace? 
#    Repeat with the first three LDs
# 6. Compute resubstitution error and LOOCV error
# 7. Perform Pearson's Chi-squared Test to see if your best model is better than random (p-value < 0.05)

# 8. Try to fit QDA to the data using all predictors

# 9. Compute resubstitution error LOOCV error for your final QDA; does it improve over LDA?

# 10. Apply kNN by optimizing the number of nearest neighbors
# 11. Apply Naïve Bayes

# 12. Choose your final model as the one with lowest LOOCV. Refit it in the learning (or training data) and make it predict the leftout test data. Discuss the results (have a look at the confusion matrix)

# Your code starts here ...
