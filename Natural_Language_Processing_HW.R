# Load packages, import and prepare data ----

library(data.table)
library(tidyverse)
library(text2vec)
library(caTools)
library(glmnet)

my_data <- read.csv("emails.csv")

#Add `id` column defined by number of rows

my_data %>% glimpse ()

my_data %>% is.na()

my_data <- inspect.na()

my_data %>% str()

my_data$id <- nrow(my_data)

set.seed(123)

view(my_data)

 # Prepare data for fitting to the model


split <- my_data$spam %>% sample.split(SplitRatio = 0.8)
train <- my_data %>% subset(split == T)
test <- my_data %>% subset(split == F)


it_train <- train$text %>% 
  itoken(preprocessor = tolower, 
         tokenizer = word_tokenizer,
         ids = train$id,
         progressbar = F) 

vocab <- it_train %>% create_vocabulary()
vocab %>% 
  arrange(desc(term_count)) %>% 
  head(110) %>% 
  tail(10) 

vectorizer <- vocab %>% vocab_vectorizer()
dtm_train <- it_train %>% create_dtm(vectorizer)

dtm_train %>% dim()
identical(rownames(dtm_train), train$id)

#Use cv.glmnet for modeling


glmnet_classifier <- dtm_train %>% 
  cv.glmnet(y = train[['spam']],
            family = 'binomial', 
            type.measure = "auc",
            nfolds = 10,
            thresh = 0.001,# high value is less accurate, but has faster training
            maxit = 1000)# again lower number of iterations for faster training

glmnet_classifier$cvm %>% max() %>% round(3) %>% paste("-> Max AUC")


it_test <- test$text %>% tolower() %>% word_tokenizer()

it_test <- it_test %>% 
  itoken(ids = test$id,
         progressbar = F)

dtm_test <- it_test %>% create_dtm(vectorizer)

preds <- predict(glmnet_classifier, dtm_test, type = 'response')[,1]
glmnet:::auc(test$spam, preds) %>% round(2)


#

# AUC score for our model is 0.994 for train and 0.99 for test


#github.com/aliasgerovs/Natural_Language_Processing-