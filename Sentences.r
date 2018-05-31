----------------# Naive bayes 
install.packages("qdap")
library(qdap)# Quantitative discourse analysis of transcripts.
install.packagess("qdapDictionaries")
library(qdapDictionaries)
install.packages("dplyr")
library(dplyr)# Data wrangling, pipe operator %>%().
install.packages("RColorBrewer")
library(RColorBrewer)# Generate palette of colours for plots.
install.packages("ggplot2")
library(ggplot2)# Plot word frequencies.
install.packages("scales")
library(scales)# Include commas in numbers.
install.packages("SnowballC")
library(SnowballC)
install.packages("caret")
library(caret)
installed.packages("naivebayes")
library(naivebayes)
install.packages("tm")
library(tm) # text preprocessing
install.packages("class")
library(class)
installed.packages("lolR")
library(lolR)

# Loading Data Set for all the labelled classes
labelled <- file.path("C:/Users/home/Downloads/SentenceCorpus/SentenceCorpus/labeled_articles")
labelled
length(dir(labelled))
dir(labelled)
docs <- Corpus(DirSource(labelled))
attach(docs)
docs

# Test Docs arxiv
test_arxiv <- file.path("C:/Users/home/Downloads/SentenceCorpus/SentenceCorpus/unlabeled_articles/arxiv_unlabeled")
test_arxiv
length(dir(test_arxiv))
dir(test_arxiv)
docs_arxiv <- Corpus(DirSource(test_arxiv))
attach(docs_arxiv)
docs_arxiv

# Test Docs jdm
test_jdm <- file.path("C:/Users/home/Downloads/SentenceCorpus/SentenceCorpus/unlabeled_articles/jdm_unlabeled")
test_jdm
length(dir(test_jdm))
dir(test_jdm)
docs_jdm <- Corpus(DirSource(test_jdm))
attach(docs_jdm)
docs_jdm

# Test Docs plos
test_plos <- file.path("C:/Users/home/Downloads/SentenceCorpus/SentenceCorpus/unlabeled_articles/plos_unlabeled")
test_plos
length(dir(test_plos))
dir(test_plos)
docs_plos <- Corpus(DirSource(test_plos))
attach(docs_plos)
docs_plos

# preprocessing

docs <- tm_map(docs, content_transformer(tolower))
docs <- tm_map(docs, removeNumbers)
docs <- tm_map(docs, removePunctuation)
docs <- tm_map(docs, stemDocument)
docs <- tm_map(docs, removeWords, stopwords("english"))
docs <- tm_map(docs, stripWhitespace)
#docs <- tm_map(docs, removeWords, c("department", "email"))

# Test data arxiv.
docs_arxiv <- tm_map(docs_arxiv, content_transformer(tolower))
docs_arxiv <- tm_map(docs_arxiv, removeNumbers)
docs_arxiv <- tm_map(docs_arxiv, removePunctuation)
docs_arxiv <- tm_map(docs_arxiv, stemDocument)
docs_arxiv <- tm_map(docs_arxiv, removeWords, stopwords("english"))
docs_arxiv <- tm_map(docs_arxiv, stripWhitespace)

# Test data jdm
docs_jdm <- tm_map(docs_jdm, content_transformer(tolower))
docs_jdm <- tm_map(docs_jdm, removeNumbers)
docs_jdm <- tm_map(docs_jdm, removePunctuation)
docs_jdm <- tm_map(docs_jdm, stemDocument)
docs_jdm <- tm_map(docs_jdm, removeWords, stopwords("english"))
docs_jdm <- tm_map(docs_jdm, stripWhitespace)

# Test data plos
docs_plos <- tm_map(docs_plos, content_transformer(tolower))
docs_plos <- tm_map(docs_plos, removeNumbers)
docs_plos <- tm_map(docs_plos, removePunctuation)
docs_plos <- tm_map(docs_plos, stemDocument)
docs_plos <- tm_map(docs_plos, removeWords, stopwords("english"))
docs_plos <- tm_map(docs_plos, stripWhitespace)



cl <- c(rep("arxiv", 30), rep("jdm",30), rep("plos",30))

# train

tdm <- TermDocumentMatrix(docs)
tdm
tdm <- removeSparseTerms(tdm, .7) # removing terms not occuring too much
tdm
dtm <- DocumentTermMatrix(docs)
dtm
dtm <- removeSparseTerms(dtm, .7)

# ARXIV
tdm_arxiv <- TermDocumentMatrix(docs_arxiv)
tdm_arxiv
tdm_arxiv <- removeSparseTerms(tdm_arxiv, .7) # removing terms not occuring too much
dtm_arxiv <- DocumentTermMatrix(docs)
dtm_arxiv
dtm_arxiv <- removeSparseTerms(dtm_arxiv, .7)

# JDM
tdm_jdm <- TermDocumentMatrix(docs_jdm)
tdm_jdm
tdm_jdm <- removeSparseTerms(tdm_jdm, .7) # removing terms not occuring too much
dtm_jdm <- DocumentTermMatrix(docs_jdm)
dtm_jdm
dtm_jdm <- removeSparseTerms(dtm_jdm, .7)

#plos

tdm_plos <- TermDocumentMatrix(docs_plos)
tdm_plos
tdm_plos <- removeSparseTerms(tdm_plos, .7) # removing terms not occuring too much
dtm_plos <- DocumentTermMatrix(docs_plos)
dtm_plos
dtm_plos <- removeSparseTerms(dtm_plos, .7)


# Train data
train <- as.matrix(dtm)
train <- cbind(train, cl)
colnames(train)[ncol(train)] <- 'y'
train <- as.data.frame(train)
train$y <- as.factor(train$y)
train$y



# Train model.
fit_nb <- naive_bayes(y ~ ., data = train )

# Test Arxiv data
test_ar <- as.matrix(dtm_arxiv)
test1 <- colSums(as.matrix(test_ar))

# Test Arxiv data
test_jd <- as.matrix(dtm_jdm)
test2 <- colSums(as.matrix(test_jd))


# Test Arxiv data
test_pl <- as.matrix(dtm_plos)
test3 <- colSums(as.matrix(test_pl))

# Generating prediction for arxiv dataset.
predit <- predict(fit_nb, test_ar)
predit
xtab <- table((predit[sample(predit,90)]), train$y)
#xtab <- table(predit, train$y)
conf <- confusionMatrix((xtab))
conf

# ================== KNN =========================

#using arxiv dataset for test
test <- as.matrix(dtm_arxiv) # arxiv 
test <- as.data.frame(test)

# Train model.0
fit_knn <- knn((train[1:129]), ((test)), cl, k = 3, prob = FALSE)
xtab <- table((fit_knn), train$y)
conf <- confusionMatrix((xtab))
conf


# ================== Rocchioo =================

# using arxiv data set for testing

train <- data.matrix(train)
roc <- lol.classify.nearestCentroid(train, cl) 
predit <- predict(roc, (test))
predit
library(Metrics)
accuracy(cl[1:33], predit)

#------------------------ General Implementation to Find Euclidenan distance of centroid and test doc
# Euclidean 

c1 <- docs[1:30] # arxiv class
c2 <- docs[31:60] # jdm class
c3 <- docs[61:90] # plos class


u_c1 <- 1/30*colSums(as.matrix(dtm_arxiv))
u_c2 <- 1/30*colSums(as.matrix(dtm_jdm))
u_c2
u_c3 <- 1/30*colSums(as.matrix(dtm_plos))
u_c3

# Checking distance of test document of class arxiv with each class centroid
length(Terms(dtm_arxiv))
length(Terms(dtm_jdm))
length(Terms(dtm_plos))

r_c1 <- sum((as.numeric(u_c1) - as.numeric(test1))^2)
r_c1 <- sqrt(r_c1)
r_c1
 
# 2675.352 => close to class 1 , proved  

r_c2 <- sum((as.numeric(u_c2)[1:129] - as.numeric(test1))^2)
r_c2 <- sqrt(r_c2)
r_c2
#  2680.816
r_c3 <- sum((as.numeric(u_c3)[1:129] - as.numeric(test1))^2)
r_c3 <- sqrt(r_c3)
r_c3
# 2694.376

predit <- c(r_c1,r_c2,r_c3)
predit
xtab <- table(predit, train$y)
#xtab <- table(predit, train$y)
conf <- confusionMatrix((xtab))
conf
