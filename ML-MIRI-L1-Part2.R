####################################################################
# Machine Learning - MIRI Master
# Author:
# think_2ice <- Manel Alba

# LAB 1: Data pre-processing (Part2)
# version of February 2015
####################################################################

# This exercise involves the use of the 'Auto' data set, which can be found in the file 'Auto.data'.
# The file contains a number of variables for cars.

graphics.off()      # reset/close all graphical devices


##### Reading the file 'Auto.data' (data on cars)

Credit <- read.csv("credsco.csv", header = TRUE, quote = "\"", dec = ".", check.names=TRUE)

Auto <- read.table("Auto.data", header=TRUE, na.strings="?")

# put proper country of origin
Auto[,"origin"] <- factor(c("USA","EU","Japan")[Auto[,"origin"]])

# convert "miles per gallon" to "liters per km"
Auto[,"mpg"] <- 235.4/Auto[,"mpg"]
colnames(Auto)[1] <- "L100km"

# car name is not useful for modelling, but it may be handy to keep it as the row name

# WARNING! surprisingly, car names are not unique, so we first prefix them by their row number

Auto$name <- paste (1:nrow(Auto), Auto$name)
rownames(Auto) <- Auto$name
Auto <- subset (Auto, select=-name)

# so this is your departing data set
summary(Auto)
attach(Auto)

# maybe you remember a plot from Lecture 1 ...

Auto.lm <- lm(L100km ~ horsepower, Auto)

plot(Auto[,"horsepower"],Auto[,"L100km"],
     pch=20,
     xlab="horsepower",ylab="fuel consumption (l/100km)",
     main="Linear regression")

# here 'x' is "horsepower"

a <- Auto.lm$coefficients["(Intercept)"]
b <- Auto.lm$coefficients["horsepower"]
abline(a=a,b=b,col="blue")
text(50,25,sprintf("y(x)=%.3fx+%.2f",b,a),col="red",pos=4)

# Now if you want to create a pdf with the previous plot, just do

# pdf("horsepower.pdf")

# < previous code >

# dev.off()

# In order to crate quick LaTeX code, try this:

# install.packages("xtable")
library(xtable)

xtable(Auto[1:4,])

# Was that nice?
# this is a list of R objects that can be embedded into a LaTeX table code:

methods(xtable)

# Wait ... with all these graphics and facilities, we may have gotten a bit obfuscated ...

# Probably we went too fast ...

table(cylinders)

# that's strange, some cars have an odd number of cylinders (are these errors?)

subset(Auto,cylinders==3)

# These Mazdas wear a Wankel engine, so this is correct

subset(Auto,cylinders==5)

# Yes, these Audis displayed five-cylinder engines, so the data is correct

# but, from summary(Auto) above we see that horsepower has 5 NA's that we'll need to take care of ...



###################################################################################
# Assignment for this lab session
# Do the following (in separate sections)
###################################################################################

# Your code starts here ...
# 1. print the dimensions of the data set
dim(Auto)

# 2. identify possible target variables according to classification or regression problems
summary(Auto)
# Whatever variable of this dataset can be a target

# 3. inspect the first 4 examples and the predictive variables 6 and 7 for the tenth example
Auto[1:4,]
Auto[10,c(6,7)]

# 4. perform a basic inspection of the dataset. Have a look at the minimum and maximum values for each variable; find possible errors and abnormal values (outliers); find possible missing values; decide which variables are continuous and which are categorical
summary(Auto)

# 5. make a decision on a sensible treatment for the missing values and apply it;
#    WARNING: 'origin' is categorical and cannot be used for knn imputation, unless you make it binary temporarily

# Rows with NA's, only in horsepower
Auto[is.na(horsepower),]
names(Auto)
dim(Auto)
aux <- Auto
# Replace origin (categorical variable for 3 numeric binary columns)
aux$Japan <- 0
aux$EU <- 0
aux$USA <- 0
summary(aux)
aux$Japan <- as.numeric(aux$Japan)
aux$EU <- as.numeric(aux$EU)
aux$USA <- as.numeric(aux$USA)
aux$Japan[origin == "Japan"] <- 1
aux$EU[origin == "EU"] <- 1
aux$USA[origin == "USA"] <- 1
summary(aux)
colnames(aux)
aux <-aux[,-8]
aux <-aux[,-4]
dim(aux)
summary(aux)
aux1 <- aux[!is.na(horsepower),]
aux2 <- aux[is.na(horsepower),]
library(class)
knn.horpow <- knn(aux1,aux2,horsepower[!is.na(horsepower)])
detach(Auto)
Auto$horsepower[is.na(Auto$horsepower)] <- as.numeric(as.character(knn.horpow))
summary(Auto)

# 6. derive one new continuous variable: weight/horsepower; derive one new categorical variable: sports_car, satisfying horsepower > 1.2*mean(horsepower) AND acceleration < median(acceleration); do you think this new variable is helpful in predicting 'origin' ?
aux <- Auto
aux$weight_horsepower <-aux$weight/aux$horsepower
aux$weight_horsepower <-as.numeric(aux$weight_horsepower)
summary(aux)
aux$sports_car <- 0
aux$sports_car[(aux$horsepower > 1.2*mean(aux$horsepower)) & (aux$acceleration < median(aux$acceleration)) ] <- 1
summary(aux)
aux$sports_car <- as.factor(aux$sports_car)
summary(aux$origin[aux$sports_car==1])
summary(aux$origin[aux$sports_car==0])

# This variable is very useful; almost every sport car in the Auto dataset
# is from USA

# 7. create a new dataframe that gathers everything and inspect it again
summary(aux)

# 8. perform a graphical summary of some of the variables (both categorical and continuous)
attach(aux)
par(mfrow=c(3,4))
hist(L100km)
hist(cylinders)
hist(displacement)
hist(horsepower)
hist(weight)
hist(acceleration)
hist(year)
barplot(table(origin))
pie(table(origin))
hist(weight_horsepower)
barplot(table(sports_car))
pie(table(sports_car))

# 9. perform a graphical comparison between some pairs of variables (both categorical and continuous)
# First I check the data types of my data.frame variables
str(aux)
# Plot acceleration vs horsepower
par(mfrow=c(1,2))
plot (acceleration, horsepower, main = "Acceleration vs horsepower",
      cex = .5, col = "dark red")
# recommended: ploting categorical vs numerical
plot (sports_car, acceleration, main = "Acceleration vs Sports car",
      cex = .5, col = "dark red")

# 10. do any of the continuous variables "look" Gaussian? can you transform some variable so that it looks more so?
par(mfrow=c(3,3))
hist(L100km)
hist(cylinders)
hist(displacement)
hist(horsepower)
hist(weight)
hist(acceleration)
hist(year)

plot(density(L100km))
plot(density(cylinders))
plot(density(displacement))
plot(density(horsepower))
plot(density(weight))
plot(density(acceleration))
plot(density(year))
# Acceleration seems gaussian
# Let's gaussianize the displacement variable
par(mfrow=c(1,3))
hist(displacement)
library(MASS)
bx <- boxcox(I(displacement+1) ~ . - displacement, data = Auto,
             lambda = seq(-0.25, 0.5, length = 10))
lambda <- bx$x[which.max(bx$y)]
Displacement.BC <- (displacement^lambda - 1)/lambda
hist(Displacement.BC)

# 11. choose one of the continuous variables and recode it as categorical; choose one of the categorical variables and recode it as continuous
aux1 <- aux
detach(aux)
aux1$sports_car <- as.numeric(aux1$sports_car)
aux1$cylinders <- as.factor(aux1$cylinders)
# Another example: acceleration as a factor of 4 levels
summary(acceleration)
aux1$acceleration <- cut(aux1$acceleration, breaks = seq(6, 25, length.out = 5))
summary(aux1$acceleration)

# 12. create a new dataframe that gathers everything and inspect it again; consider 'origin' as the target variable; perform a basic statistical analysis as indicated in SECTION 9
colnames(aux1)
attach(aux1)
Auto.new <- data.frame(L100km,cylinders, displacement,horsepower,
                       weight,acceleration, year, origin,
                       weight_horsepower,sports_car)
detach(aux1)
summary(Auto.new)
str(Auto.new)
range.cont <- c(1,3:5,7,9:10)
str(Auto.new[,range.cont])
library(psych)
describeBy(Auto.new[,range.cont],Auto.new$origin)
pairs(Auto.new[,range.cont],main = "Auto Origin DataBase",col = (1:length(levels(Auto.new$origin)))[unclass(Auto.new$origin)])
names(Auto.new)[range.cont]
varc <- list(Auto.new$L100km, Auto.new$displacement,Auto.new$horsepower,
             Auto.new$weight,Auto.new$year,Auto.new$weight_horsepower,
             Auto.new$sports_car)
pvalcon <- NULL
for (i in 1:length(range.cont))
  pvalcon[i] <- (oneway.test (varc[[i]]~Auto.new$origin))$p.value
pvalcon <- matrix(pvalcon)
row.names(pvalcon) <- colnames(Auto.new)[range.cont]
sort(pvalcon[,1])

ncon <- nrow(pvalcon)

par (mfrow=c(3,4))

# Explain horsepower
for (i in 1:ncon)
{
  barplot (tapply(varc[[i]], Auto.new$origin, mean),main=paste("Means by",row.names(pvalcon)[i]), las=2, cex.names=0.75)
  abline (h=mean(varc[[i]]))
  legend (0,mean(varc[[i]]),"Global mean",bty="n")
}
mosthighlycorrelated <- function(mydataframe,numtoreport)
{
  # find the correlations
  cormatrix <- cor(mydataframe)
  # set the correlations on the diagonal or lower triangle to zero,
  # so they will not be reported as the highest ones:
  diag(cormatrix) <- 0
  cormatrix[lower.tri(cormatrix)] <- 0
  # flatten the matrix into a dataframe for easy sorting
  fm <- as.data.frame(as.table(cormatrix))
  # assign human-friendly names
  names(fm) <- c("First.Variable", "Second.Variable","Correlation")
  # sort and print the top n correlations
  head(fm[order(abs(fm$Correlation),decreasing=TRUE),],n=numtoreport)
}

mosthighlycorrelated(Auto.new[,range.cont], 5)
# 13. shuffle the final dataset and save it into a file for future use
# Shuffling the data (to avoid possible biases)
set.seed (104)
Auto.new <- Auto.new[sample.int(nrow(Credit.new)),]
# Saving the new dataset in .csv format
write.csv(Auto.new, "auto_new.csv")
