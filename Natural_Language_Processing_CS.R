# Week 11 CS

# Load packages, import and prepare data 

library(data.table)
library(tidyverse)
library(text2vec)
library(caTools)
library(glmnet)

# Importing NLP data

my_data <- read_csv("nlpdata.csv")

view(my_data)

glimpse(my_data)

any(is.na(my_data)) # False means not having any NA


# Define Preprocessing Function and takenizotion function

# Split data

set.seed(123)
split <- my_data$sentiment %>% sample.split(SplitRatio = 0.8)
train <- my_data %>% subset(split = T)
test <- my_data %>% subset(split = F)

#Tokenize

# 1st way
it_train <- train$Review %>% 
  itoken(preprocessor = tolower, 
         tokenizer = word_tokenizer,
         ids = train$V1,
         progressbar = F) 

# 2nd way
train_tokens <- train$Review %>% tolower() %>% word_tokenizer()

it_train <- train_tokens %>% 
  itoken(ids = train$V1,
         progressbar = F)

vocab <- it_train %>% create_vocabulary()
vocab %>% 
  arrange(desc(term_count)) %>% 
  head(110) %>% 
  tail(10) 

vectorizer <- vocab %>% vocab_vectorizer()
dtm_train <- it_train %>% create_dtm(vectorizer)

dtm_train %>% dim()
identical(rownames(dtm_train), train$V1)


# Modelling

glmnet_classifier <- dtm_train %>% 
  cv.glmnet(y = train[['Liked']],
            family = 'binomial', 
            type.measure = "auc",
            nfolds = 10,
            thresh = 0.001,# high value is less accurate, but has faster training
            maxit = 1000)

glmnet_classifier$cvm %>% max() %>% round(3) %>% paste("-> Max AUC")

it_test <- test$Review %>% tolower() %>% word_tokenizer()

it_test <- it_test %>% 
  itoken(ids = test$V1,
         progressbar = F)

dtm_test <- it_test %>% create_dtm(vectorizer)

# Prediction Of Models

preds <- predict(glmnet_classifier, dtm_test, type = 'response')[,1]
glmnet:::auc(test$Liked, preds) %>% round(2)



stop_words <- c("i", "you", "he", "she", "it", "we", "they",
                "me", "him", "her", "them",
                "my", "your", "yours", "his", "our", "ours",
                "myself", "yourself", "himself", "herself", "ourselves",
                "the", "a", "an", "and", "or", "on", "by", "so",
                "from", "about", "to", "for", "of", 
                "that", "this", "is", "are")

vocab <- it_train %>% create_vocabulary(stopwords = stop_words)


pruned_vocab <- vocab %>% 
  prune_vocabulary(term_count_min = 10, 
                   doc_proportion_max = 0.5,
                   doc_proportion_min = 0.001)


pruned_vocab %>% 
  arrange(desc(term_count)) %>% 
  tail(10) 

# Create DTM for training and testing pruned vocanulary

vectorizer <- pruned_vocab %>% vocab_vectorizer()


dtm_train <- it_train %>% create_dtm(vectorizer)
dtm_train %>% dim()

glmnet_classifier <- dtm_train %>% 
  cv.glmnet(y = train[['Liked']], 
            family = 'binomial',
            type.measure = "auc",
            nfolds = 4,
            thresh = 0.001,
            maxit = 1000)

glmnet_classifier$cvm %>% max() %>% round(3) %>% paste("-> Max AUC")

dtm_test <- it_test %>% create_dtm(vectorizer)
preds <- predict(glmnet_classifier, dtm_test, type = 'response')[,1]
glmnet:::auc(test$Liked, preds) %>% round(2)

#github.com/aliasgerovs/Natural_Language_Processing-

