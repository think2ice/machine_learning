####################################################################
# Machine Learning - MIRI Master

# LAB 4: Text pre-processing (Part1)
# version of March 2016
####################################################################


####################################################################
# Example 1: Preprocessing a news dataset
################################################################

## We are going to use a slightly-processed version of the famous Reuters news articles dataset.
## In this version all articles with no Topic annotations are dropped and only the first term from 
## the Topic annotation list is retained (some articles had several topics assigned). 
## Also the text of each article will be converted to lowercase, whitespaces will be normalized to 
## single spaces and other similar pre-processing tasks will be carried out

## The dataset is a list of pairs (Topic, News content)

## DISCLAIMER: this script is not intended to be part of a NLP task; we are only interested in data pre-processing for ML, which is our business

## Reading the file (Reuters news articles dataset)

## Note that we can directly read the compressed version (reuters.txt.gz). 
## There is no need to unpack the gz file; for local files R handles unpacking automagically

reuters <- read.table("reuters.txt.gz", header=TRUE)

## R originally loads this as factor, so it needs fixing

reuters$Content <- as.character(reuters$Content)

## we now have over 9,500 news articles ...

(N <- dim(reuters)[1])  # number of rows

## ... on a variety of topics (actually, on 79):

(l <- length(levels(reuters$Topic)))

(tops <- table(reuters$Topic))

## these are the topics more frequent than the average frequency

selected.tops <- tops[tops>N/l]

barplot(selected.tops, cex.names=0.7, main="More frequent Topics")

## Let's work only with these, for simplicity:

reuters.freq <- reuters[reuters$Topic %in% names(selected.tops),]

## The resulting data frame contains 7,873 news items on 9 topics. The actual news text is the column 
## "Content" and its category is the column "Topic". Possible goals are visualizing the data and creating a classifier
## for the news articles (we'll we doing both tasks several times during the course)

(N.freq <- dim(reuters.freq)[1])  # new number of rows

# re-level the factor to have only these 9 levels (VERY IMPORTANT)
reuters.freq$Topic <- factor(reuters.freq$Topic)              

## an example of a text about 'money-fx'
reuters.freq[130,]

## an example of a text about 'sugar'
reuters.freq[134,]

## some entries are quite long ...

reuters.freq[133,]

## now let's try to do some nice visualization

library(wordcloud)  # for word clouds

library(tm) # for topic modelling

## We first transform the data to a Corpus to have access to the nice {tm} routines for text manipulation

## an toy example of such structure:

toy.text <- c("This is a loving first attempt to my loving memories.", 
              "This a loving second attempt!",
              "And a THIRD (3) third attempt!",
              "I need you you you ...")

toy.cp <- Corpus(VectorSource(toy.text))
inspect(toy.cp)

## Let's go

reuters.cp <- VCorpus (VectorSource(as.vector(reuters.freq$Content)) )

inspect(reuters.cp[1:2])


## If we now tried to create a word cloud it will be too detailed (too many different words, many of which undesired), so we pre-process the Corpus a bit

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

(my.stopwords <- c(stopwords('english'), 'reuter', 'reuters', 'said'))
reuters.cp <- tm_map(reuters.cp, removeWords, my.stopwords)

library(SnowballC) # needed for stemming
# see http://cran.r-project.org/web/packages/SnowballC/index.html

# Stemming means reducing inflected or derived words to their word stem, base or root form

# Simple example
wordStem(c("win", "winning", "winner","winners"), language="english")

# that's not bad at all
getStemLanguages()

reuters.cp <- tm_map(reuters.cp, stemDocument, language="english", lazy=TRUE)

## Convert to TermDocumentMatrix

tdm <- TermDocumentMatrix (reuters.cp)

## What kind of data do we have now? A 'TermDocumentMatrix' is a data structure for *sparse* matrices 
## based on simple triplet representation (i,j,v), standing for row indices i, column indices j, and values v, respectively

# This structure is somewhat hidden for manipulation purposes (which is good). 

## 'inspect' displays detailed information on a corpus or a term-document matrix:

inspect(tdm[80:100,20:50])

# So we really have a very sparse data representation

# inspect most popular words
findFreqTerms(tdm, lowfreq=1000)

## now we can form frequency counts

v <- sort(rowSums(as.matrix(tdm)),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v/sum(v))

# And finally we can create the wordcloud:

wordcloud(d$word,d$freq,scale=c(8,.5),max.words=100, random.order=FALSE)

## Now let's do something better; we generate a new term-document matrix by TfF-IDF (weight by term frequency - inverse document frequency)

word.control <- list(weighting = function(x) weightTfIdf(x, normalize = TRUE))

tdm2 <- TermDocumentMatrix (reuters.cp, control=word.control)

v <- sort(rowSums(as.matrix(tdm2)),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v/sum(v))

## which possibly gives a better impression of the data, but this is rather subjective

wordcloud(d$word,d$freq,scale=c(8,1),max.words=50, random.order=FALSE)

## This 'd' is a useful structure for analysis (note these entries are sorted by decreasing relative frequency)

d$word[1:10]
d$freq[1:10]

## However, rather than visualization, in text processing the goal often is to classify the Topic given the Content
## In this case, the original data structure can be more suited for ML tasks, 

## Let's work only with a couple of topics, for simplicity:

selected.tops <- c('ship','sugar')
reuters.ss <- reuters[reuters$Topic %in% selected.tops,]

## The goal is to classify (and be able to predict) news documents dealing with ships or with sugar

## The resulting data now contains 305 news items

(N.freq <- dim(reuters.ss)[1])  # new number of rows

# re-level the factor to have only 2 levels (VERY IMPORTANT)
reuters.ss$Topic <- factor(reuters.ss$Topic)              

## an example of a text about 'ship'
reuters.ss[130,]

## an example of a text about 'sugar'
reuters.ss[134,]

## OK, let's go. One of the most used ML representations for text is the "bag of words":

documents <- reuters.ss$Content

## first we need some auxiliary functions to actually create the structure
source('reuters-aux.R')

## this is an example of basic text processing
strip.text ("I'm going $home to say 3         hellos to my MUM!")

## and this is an example of basic "bag of words" processing

toy.text

(toy.textl <- as.list(toy.text))

toy.bagged <- vector("list", length(toy.textl))

(toy.list <- lapply(toy.textl,strip.text))

## now finally the bag of words ... with 'table()'

(toy.bow <- lapply(toy.list,table))

## Let's apply this to our 'ship' and 'sugar' data:

## "pre-allocate" an empty list of the required length
bagged <- vector("list", length(documents))

## Now generate bag of words as a list

## BTW, everything that you do to 'scan' a data structure should use R's advanced functions
## (the lapply,sapply, etc family is just a taste); it is much faster than an explicit loop
## (and more plesant from the point of view of a high-level programmer)

# 1. produce the stripped text
bagged <- lapply(documents,strip.text)

bagged[[1]]

# 2. produce the bag of words

bagged.BoW <- lapply(bagged,table)

bagged.BoW[[1]]

# 3. make it a dataframe (this make take a while)

reuters.BOWs <- make.BoW.frame (bagged.BoW)

## let's see that it sort of works

reuters.BOWs[1,"year"]

reuters.BOWs[1,"when"]

dim(reuters.BOWs)

## so we have our 305 news entries "described" by nearly 2,800 features (the words)

# Now we weight by inverse document frequency
reuters.BOWs.tfidf <- tfidf.weight(reuters.BOWs)

# and normalize by vector length
reuters.BOWs.tfidf <- div.by.euc.length(reuters.BOWs.tfidf)

dim(reuters.BOWs.tfidf)

# let's inspect the result
summary(colSums(reuters.BOWs.tfidf))

## too many columns ... a professional pre-processing will try to eliminate common English words ('the','and', 'they', etc)

## by the moment, let's do some "clever" removal:

# 1. remove those words shorter than 3 characters

reuters.BOWs.tfidf <- subset.data.frame (reuters.BOWs.tfidf, 
                                         select=sapply(colnames(reuters.BOWs.tfidf), FUN=nchar)>2)

dim(reuters.BOWs.tfidf)


# 2. remove those words whose total sum is not greater than the third quartile of the distribution

(r.3rdQ <- summary(colSums(reuters.BOWs.tfidf))[5])

reuters.BOWs.tfidf <- subset.data.frame (reuters.BOWs.tfidf, 
                                         select=colSums(reuters.BOWs.tfidf)>r.3rdQ)

# this is roughly a further 75% reduction, corresponding to the less represented words
dim(reuters.BOWs.tfidf)

# Add class labels back (the "Topics")

# (the normalizing and weighting functions don't work well with non-numeric columns so it's simpler to add the labels at the end)

reuters.definitive <- data.frame(Topic=reuters.ss$Topic,reuters.BOWs.tfidf)

dim(reuters.definitive)

# a final look at the first 10 variables

summary(reuters.definitive[,1:10])

# Now save the result ...

save(reuters.definitive,file="Reuters-BOWs.RData")

## Naive Bayes Classifier

# Use now the simplest of the classifiers (which we'll see very soon), to check
# that our data structure and problem approach works and how easy is now to perform
# modelling in R

library(e1071) # naiveBayes

set.seed (1234)
N <- nrow(reuters.definitive)                                                                                              
learn <- sample(1:N, round(0.67*N))
nlearn <- length(learn)
ntest <- N - nlearn

reuters.nb <- naiveBayes (Topic ~ ., data=reuters.definitive[learn,], laplace=3)

# predict the left-out data

pred <- predict(reuters.nb,newdata=reuters.definitive[-learn,])

(tt <- table(Truth=reuters.definitive$Topic[-learn], Predicted=pred))

(error <- 100*(1-sum(diag(tt))/sum(tt)))

# so error is 17.8%, not bad ... although there is a marked tendency to predict 'sugar'. The majority class is:

(baseline <- 100*(1 - max(table(reuters.definitive$Topic))/nrow(reuters.definitive)))

# so all errors below 42% are an improvement over the baseline

100*(baseline-error)/baseline

# actually we are able to get a relative reduction of 57.5% in error

## However note that this result is highly unstable, given the small size of both the learn and test sets

## We'll encounter this nice Reuters data set later on next labs


################################################################
# Example 2: preprocessing a spam e-mail dataset
################################################################

## As you may know, many packages come with their own data; it turns out that 'spam data' is included in {kernlab}
## One has to be very careful, because sometimes this comes at a price: it is difficult to be sure that the dataset has not been 'touched' by modifying it in some way (reported or not)

## My best advice is to go to the original source (the UCI ML repository in this case)

## I would say that the spam data we get when we load it form the {kernlab} package is identical to the original, so we'll go ahead with it ...

library(kernlab)  

data(spam)

## please read now the help entry completely in order to understand what we have here
help(spam)

## 'str' gives the internal structure of almost any R object
str(spam)

## This is again a text-processing problem, but a different one than before, given that the emails have already been pre-processed by someone else; in particular, we depart from a specific choice of words

## 'summary' is always very useful; do you notice anything strange?
summary(spam)

# 1. first we make sure the target variable is a factor with appropriate levels:

is.factor(spam$type)

# 2. a quick look at the 6 variables indicating the frequency of the characters ‘;’, ‘(’, ‘[’, ‘!’, ‘\$’, and ‘\#’ reveals that their distributions are quite skewed (this is highly typical of count data); probably there is nothing we should do about it

boxplot(spam[,49:54]) # you'd better zoom this

# 3. a quick look at the 6 variables indicating the average, longest and total run-length of capital letters reveals these are very skewed; one way to 'fix' this is by taking logs; try this:

boxplot(spam[,55:57])

# against

boxplot(log10(spam[,55:57]+1))

# Apparently the log has some little effect ... but let's look close

par(mfrow=c(1,2))

hist(spam[,55],main=paste(colnames(spam)[55],"before log10"))

hist((log10(spam[,55]+1)),main=paste(colnames(spam)[55],"after log10"))

# it does a lot; so we process these 3 variables ...

spam[,55:57] <- as.matrix(log10(spam[,55:57]+1))


## We want to follow a general perspective and therefore, we must eliminate columns (words) that 
## refer to specific contexts, such as "George" and "650" (which is the zipcode of George)

## first we eliminate those e-mails containing appearances of the word 'george', the number 650, 
## or the words indicating HP company:

spam2 <- spam[spam$george==0,]
spam2 <- spam2[spam2$num650==0,]
spam2 <- spam2[spam2$hp==0,]
spam2 <- spam2[spam2$hpl==0,]

george.vars <- 25:28
spam2 <- spam2[,-george.vars]

# Sadly we loose a good bunch of the data, but that's life 
## (otherwise we would have designed a spam filter for George while working at HP, desk 650!)
dim(spam2)

## Now we can do some feature extraction; reducing dimension while keeping (or even increasing)
## the relationship to the target variable is *always* a good idea from a ML point of view:

## The conversations related to money are easily recognized in the spam email classification
## We delve into the available words to look for explicit appearances of money-related words: 

## we find "free business credit money", which will be merged into one new variable

moneys.vars <- c(16,17,20,24)
spam3 <- data.frame( spam2[,-moneys.vars], spam2[,16]+spam2[,17]+spam2[,20]+spam2[,24])
                     
colnames(spam3)[51] <- "about.money"

dim(spam3)

set.seed (4321)
N <- nrow(spam3)                                                                                              
learn <- sample(1:N, round(0.67*N))
nlearn <- length(learn)
ntest <- N - nlearn

spam.nb <- naiveBayes (type ~ ., data=spam3[learn,], laplace=1)

pred <- predict(spam.nb,newdata=spam3[-learn,])

(tt <- table(Predicted=pred,Truth=spam3$type[-learn]))

(error <- 100*(1-sum(diag(tt))/sum(tt)))

# so error is 26.26%, quite bad at first glance... The majority class is:

baseline <- 100*(1 - max(table(spam3$type))/nrow(spam3))

100*(baseline-error)/baseline

# All absolute errors below 42% are an improvement, and actually we are able to 
# get a relative reduction of over 38% (from 42.4% to 26.3%), but this spam filter is crap: there is a marked tendency to predict 'spam', which is catastrophic in a practical deployment. 

## But don't worry; soon we'll learn other classifiers (not much more complicated than naiveBayes), 
## that are able to create a much nicer (and useful) spam filter
