####################################################################
# Machine Learning - MIRI Master
# Author: 
# think_2ice <- Manel Alba

# LAB 4: Text pre-processing 
# version of March 2016
####################################################################
# These exercises involve the use of the same two datasets seen in Part1 'Reuters news' and 'spam'
###################################################################################
# Assignment for this lab session
# Do the following (in separate sections)
###################################################################################

# 1. Start anew with the 'Reuters news' data set (I strongly suggest that you quit RStudio, open it again, and start a new code file)

reuters <- read.table("reuters.txt.gz", header=TRUE)

# 2. Do the same pre-processing, this time choosing the two topics ('grain' and 'crude'), which deal with grain-related news (i.e., cereals) and crude oil news, respectively. 

summary(reuters)
dim(reuters)
str(reuters)
# Content should be defines as character
reuters$Content <- as.character(reuters$Content)
# Number of news
N <- dim(reuters)[1]
# Number of topics
l <- length(levels(reuters$Topic))
# Get only grain and crude topics
tops <- table(reuters$Topic)
selected.tops <- tops[tops>N/l]
barplot(selected.tops, cex.names=0.7, main="More frequent Topics")

# Working only with grain and crude topics
selected.tops <- c('grain','crude')
reuters.ss <- reuters[reuters$Topic %in% selected.tops,]
dim(reuters.ss)
str(reuters.ss)
# New number of rows
N.freq <- dim(reuters.ss)[1]
# re-level the factor to have only these 2 levels (VERY IMPORTANT)
reuters.ss$Topic <- factor(reuters.ss$Topic)

# 3. Perform a wordcloud on the selected news subset

# for word clouds
library(wordcloud)  
# for topic modelling
library(tm)
reuters.cp <- VCorpus (VectorSource(as.vector(reuters.ss$Content)) )
inspect(reuters.cp[1:2])
# Reduce the number of words of the wordcloud
# Elimination of extra whitespaces
reuters.cp <- tm_map(reuters.cp, stripWhitespace)
# Elimination of numbers
reuters.cp <- tm_map(reuters.cp, removeNumbers)
# Elimination of punctuation marks
reuters.cp <- tm_map(reuters.cp, removePunctuation)
# Conversion to lowcase (this text is already in lowcase, so just to be sure)
reuters.cp <- tm_map(reuters.cp, content_transformer(tolower))
# Removal of English generic and custom stopwords
# These latter are words typically related to the data at hand
my.stopwords <- c(stopwords('english'), 'reuter', 'reuters', 'said')
reuters.cp <- tm_map(reuters.cp, removeWords, my.stopwords)
# needed for stemming: http://cran.r-project.org/web/packages/SnowballC/index.html
library(SnowballC)
reuters.cp <- tm_map(reuters.cp, stemDocument, language="english", lazy=TRUE)
# Convert to TermDocumentMatrix
# A 'TermDocumentMatrix' is a data structure for *sparse* matrices based on simple 
# triplet representation (i,j,v), standing for row indices i, column indices j, and values v, respectively
tdm <- TermDocumentMatrix (reuters.cp)
# inspect most popular words
findFreqTerms(tdm, lowfreq=1000)
# form fequency counts
v <- sort(rowSums(as.matrix(tdm)),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v/sum(v))
# create the workcloud
wordcloud(d$word,d$freq,scale=c(8,.5),max.words=100, random.order=FALSE)
# generate a new term-document matrix by TfF-IDF (weight by term frequency - inverse document frequency)
word.control <- list(weighting = function(x) weightTfIdf(x, normalize = TRUE))
tdm2 <- TermDocumentMatrix (reuters.cp, control=word.control)
v <- sort(rowSums(as.matrix(tdm2)),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v/sum(v))
# which possibly gives a better impression of the data, but this is rather subjective
wordcloud(d$word,d$freq,scale=c(8,1),max.words=50, random.order=FALSE)
# This 'd' is a useful structure for analysis (note these entries are sorted by decreasing relative frequency)
d$word[1:10]
d$freq[1:10]

# Bag of words
documents <- reuters.ss$Content
source('reuters-aux.R')
# "pre-allocate" an empty list of the required length
bagged <- vector("list", length(documents))
# produce the stripped text
bagged <- lapply(documents,strip.text)
# produce the bag of words
bagged.BoW <- lapply(bagged,table)
# make the result a dataframe
reuters.BOWs <- make.BoW.frame (bagged.BoW)
# inspect the new dataframe
dim(reuters.BOWs)
# Now we weight by inverse document frequency
reuters.BOWs.tfidf <- tfidf.weight(reuters.BOWs)
# and normalize by vector length
reuters.BOWs.tfidf <- div.by.euc.length(reuters.BOWs.tfidf)
dim(reuters.BOWs.tfidf)
summary(colSums(reuters.BOWs.tfidf))
# too many words, needs removal
# remove those words shorter than 3 characters
reuters.BOWs.tfidf <- subset.data.frame (reuters.BOWs.tfidf, select=sapply(colnames(reuters.BOWs.tfidf), FUN=nchar)>2)
dim(reuters.BOWs.tfidf)
# remove those words whose total sum is not greater than the third quartile of the distribution
r.3rdQ <- summary(colSums(reuters.BOWs.tfidf))[5]
reuters.BOWs.tfidf <- subset.data.frame (reuters.BOWs.tfidf, select=colSums(reuters.BOWs.tfidf)>r.3rdQ)
dim(reuters.BOWs.tfidf)
# save this dataset
# (the normalizing and weighting functions don't work well with non-numeric columns so 
# it's simpler to add the labels at the end)
reuters.definitive <- data.frame(Topic=reuters.ss$Topic,reuters.BOWs.tfidf)
dim(reuters.definitive)
save(reuters.definitive,file="Reuters-BOWs.RData.Grain.Crude")
summary(reuters.definitive)

# 4. Create a naiveBayes classifier to predict the topic ('grain' and 'crude')

# Naive Bayes
library(e1071)
set.seed (1234)
N <- nrow(reuters.definitive)                                                                                              
learn <- sample(1:N, round(0.67*N))
nlearn <- length(learn)
ntest <- N - nlearn
reuters.nb <- naiveBayes (Topic ~ ., data=reuters.definitive[learn,], laplace=3)
pred <- predict(reuters.nb,newdata=reuters.definitive[-learn,])
(tt <- table(Truth=reuters.definitive$Topic[-learn], Predicted=pred))
(error <- 100*(1-sum(diag(tt))/sum(tt)))
(baseline <- 100*(1 - max(table(reuters.definitive$Topic))/nrow(reuters.definitive)))
100*(baseline-error)/baseline

## NOTE: you are allowed to modify whatever you want in order to taylor your result to these two topics, including new ideas. Please write down everything you do.

# 5. Move now to the 'spam' database. Same suggestion as before.

#read the data
library(kernlab)  
data(spam)
## please read now the help entry completely in order to understand what we have here
help(spam)
str(spam)
# the emails have already been pre-processed by someone else; in particular, we depart from a specific choice of words
summary(spam)
dim(spam)

# 6. Do the same pre-processing, this time creating *another* new variable of your choice and any other pre-processing you think useful (example: working out the punctuation signs)
boxplot(spam[,49:54]) # you'd better zoom this
# A quick look at the 6 variables indicating the average, longest and 
# total run-length of capital letters reveals these are very skewed; one way to 'fix' this is by taking logs; try this:
boxplot(spam[,55:57])
# against
boxplot(log10(spam[,55:57]+1))
# Apparently the log has some little effect ... but let's look close
par(mfrow=c(1,2))
hist(spam[,55],main=paste(colnames(spam)[55],"before log10"))
hist((log10(spam[,55]+1)),main=paste(colnames(spam)[55],"after log10"))
# it does a lot; so we process these 3 variables ...
spam[,55:57] <- as.matrix(log10(spam[,55:57]+1))
# We want to follow a general perspective and therefore, we must eliminate columns (words) that 
# refer to specific contexts, such as "George" and "650" (which is the zipcode of George)
# first we eliminate those e-mails containing appearances of the word 'george', the number 650, 
# or the words indicating HP company:
spam2 <- spam[spam$george==0,]
spam2 <- spam2[spam2$num650==0,]
spam2 <- spam2[spam2$hp==0,]
spam2 <- spam2[spam2$hpl==0,]
george.vars <- 25:28
spam2 <- spam2[,-george.vars]
dim(spam2)
# The conversations related to money are easily recognized in the spam email classification
# We delve into the available words to look for explicit appearances of money-related words:
# we find "free business credit money", which will be merged into one new variable
moneys.vars <- c(16,17,20,24)
spam3 <- data.frame( spam2[,-moneys.vars], spam2[,16]+spam2[,17]+spam2[,20]+spam2[,24])
colnames(spam3)[51] <- "about.money"
dim(spam3)
summary(spam3)
# I expect that a spam mail will have more special characters as a non-spam mail
colnames(spam3)
which(colnames(spam3)=="charSemicolon")
which(colnames(spam3)=="charHash")
special.char.vars <- 41:46
spam4 <- data.frame(spam3[,-special.char.vars], spam3[,41]+spam2[,42]+spam2[,43]+spam2[,44]+spam2[,45]+spam2[,46])
dim(spam4)
colnames(spam4)
colnames(spam4)[46] <- "special.characters"

# 7. Create a new naiveBayes classifier and compare it to the one in Part1

set.seed (4321)
N <- nrow(spam4)                                                                                              
learn <- sample(1:N, round(0.67*N))
nlearn <- length(learn)
ntest <- N - nlearn
spam.nb <- naiveBayes (type ~ ., data=spam4[learn,], laplace=1)
pred <- predict(spam.nb,newdata=spam4[-learn,])
(tt <- table(Predicted=pred,Truth=spam4$type[-learn]))
(error <- 100*(1-sum(diag(tt))/sum(tt)))
# error 24.44%, lower than in part 1
baseline <- 100*(1 - max(table(spam3$type))/nrow(spam3))
100*(baseline-error)/baseline
# All absolute errors below 42% are an improvement, and actually we are able to 
# get a relative reduction (from 42.4% to 24.44%), 
# There is a marked tendency to predict 'spam', which is catastrophic in a practical deployment. 

