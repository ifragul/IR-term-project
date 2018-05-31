
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

labelled <- file.path("C:/Users/home/Downloads/corpus/corpus/citations_class")
labelled
length(dir(labelled))
docs <- Corpus(DirSource(labelled, mode = "text"))
attach(docs)



# reading lables

cl <- c("applied","followed","referred to","considered","considered","cited","applied","cited","applied"
        ,"cited","applied","discussed","referred to","cited","referred to","cited","applied","cited","referred to"
        ,"distinguished","cited","cited","cited","applied","cited","referred to","cited","cited","referred to",
        "referred to","followed","applied","cited","cited","cited","cited","distinguished","distinguished","cited"
        ,"cited","cited","referred to","cited","applied","referred to","cited","referred to","cited","applied","cited" 
        ,"applied","cited","cited","followed","applied","referred to","referred to","applied","cited","referred to"
        ,"cited","cited","followed","cited","cited","referred to","related","cited","related","cited","cited","cited"
        ,"cited","cited","cited","cited","cited","applied","cited","cited")


docs <- tm_map(docs, content_transformer(tolower))
docs <- tm_map(docs, removeNumbers)
docs <- tm_map(docs, removePunctuation)
docs <- tm_map(docs, stemDocument)
docs <- tm_map(docs, stripWhitespace)




tdm <- TermDocumentMatrix((docs))
tdm
tdm <- removeSparseTerms(tdm, .7) # removing terms not occuring too much
dtm <- DocumentTermMatrix(docs)
dtm
dtm <- removeSparseTerms(dtm, .7)

findFreqTerms(dtm, 10)

# Index to use
index <-  c(25:43, 60:80)

# Train data
train <- as.matrix(dtm)

# Test Docs 
test <- train[-index,]
test <- as.matrix(test)
test <- as.data.frame(test)

train <- train[index,]
train <- cbind(train, cl)
colnames(train)[ncol(train)] <- 'y'
train <- as.data.frame(train)


train$y <- as.factor(train$y)
train$y



# Train model.
fit_nb <- naive_bayes(y ~ ., data = train )

# Check accuracy on training.
predit <- predict(fit_nb, (train))
predit
xtab1 <- table(predit, train$y[sample(length(predit))])
conf <- confusionMatrix((xtab1))
conf

# --------------------- KNN

# Train model.
# Train model.0
cl <- c("applied","followed","referred to","considered","considered","cited","applied","cited","applied"
        ,"cited","applied","discussed","referred to","cited","referred to","cited","applied","cited","referred to"
        ,"distinguished","cited","cited","cited","applied","cited","referred to","cited","cited","referred to",
        "referred to","followed","applied","cited","cited","cited","cited","distinguished","distinguished","cited"
        ,"cited" 
        )

train <- as.matrix(dtm)
test <- train[-index,]
train <- train[index,]

fit_knn <- knn(data.matrix(train), data.matrix(test), cl, k = 6, prob = TRUE)
train <- cbind(train, cl)
colnames(train)[ncol(train)] <- 'y'
train <- as.data.frame(train)

xtab <- table((fit_knn), train$y[sample(length(fit_knn))])
conf <- confusionMatrix((xtab))
conf


# --------------- Rcohhio
cl <- c("applied","followed","referred to","considered","considered","cited","applied","cited","applied"
        ,"cited","applied","discussed","referred to","cited","referred to","cited","applied","cited","referred to"
        ,"distinguished","cited","cited","cited","applied","cited","referred to","cited","cited","referred to",
        "referred to","followed","applied","cited","cited","cited","cited","distinguished","distinguished","cited"
        ,"cited" 
)
train <- data.matrix(train)
roc <- lol.classify.nearestCentroid(train, cl)
predit <- predict(roc, (test))
predit
xtab1 <- table(predit, cl)
library(Metrics)
accuracy(cl, predit)


