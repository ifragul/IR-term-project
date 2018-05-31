
----------------# DBWorld Naive bayes 
  install.packages("foreign")
library(foreign)
installed.packages("tm")
library(tm)# Framework for text mining.
install.packages("qdap")
library(qdap)# Quantitative discourse analysis of transcripts.
install.packages("qdapDictionaries")
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
install.packages("naivebayes")
library(naivebayes)
install.packages("RSKC")
library(RSKC)
install.packages("class")
library(class)
install.packages("XML")
library(XML)
install.packages("plyr")
library(plyr)
install.packages("lolR")
library(lolR)
install.packages("MASS")
library(MASS)

bodies <- read.arff("C:/Users/home/Downloads/dbworld/WEKA/dbworld_bodies.arff")
subjects <- read.arff("C:/Users/home/Downloads/dbworld/WEKA/dbworld_subjects.arff")

labelled <- file.path("C:/Users/home/Downloads/dbworld/WEKA/")
labelled
length(dir(labelled))
dir(labelled)
docs <- Corpus(DirSource(labelled))
attach(docs)
docs

# reading lables
data(DBWorld) # DBWorld has also been installed
cl <- rownames(DBWorld)
data.frame(cl)



docs <- tm_map(docs, content_transformer(tolower))
docs <- tm_map(docs, removeNumbers)
docs <- tm_map(docs, removePunctuation)
docs <- tm_map(docs, stemDocument)
docs <- tm_map(docs, stripWhitespace)


labels_new <- cbind(subjects,cl)

tdm <- TermDocumentMatrix((docs))
tdm
tdm <- removeSparseTerms(tdm, .7) # removing terms not occuring too much
dtm <- DocumentTermMatrix(docs)
dtm
dtm <- removeSparseTerms(dtm, .7)

findFreqTerms(dtm, 10)


# Test Docs 
index <- createDataPartition(labels_new$cl, p=.50, list=FALSE)
test <- train[index,]
test <- as.matrix(test)
test <- as.data.frame(test)

# Train data

train <- as.matrix(labels_new)
#train <- cbind(train, cl)
colnames(train)[ncol(train)] <- 'y'
train <- as.data.frame(train)
train$y <- as.factor(train$y)
train$y



# Train model.
fit_nb <- naive_bayes(y ~ ., data = train )

# Check accuracy on training.
predit <- predict(fit_nb, (test))
predit
xtab1 <- table(predit, train$y[sample(length(predit))])
conf <- confusionMatrix((xtab1))
conf

# --------------------- KNN

# Train model.
# Train model.0
fit_knn <- knn(data.matrix(train), data.matrix(test), cl, k = 3, prob = TRUE)
xtab <- table((fit_knn), train$y[sample(length(fit_knn))])
conf <- confusionMatrix((xtab))
conf


# --------------- Rcohhio

train <- data.matrix(train)
roc <- lol.classify.nearestCentroid(train, cl)
predit <- predict(roc, (test))
predit
library(Metrics)
accuracy(cl[1:33], predit)





