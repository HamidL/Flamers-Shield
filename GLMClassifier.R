requiredPackages <- c("text2vec", "doParallel","foreach","glmnet", "stringr", "tm", "tidytext", "textstem","hunspell","parallel","wordcloud2", "dplyr", "ggplot2","corrplot")

for (pac in requiredPackages) {
  if(!require(pac,  character.only=TRUE)){
    install.packages(pac, repos="http://cran.rstudio.com")
    library(pac,  character.only=TRUE)
  } 
}
rm(requiredPackages)
rm(pac)

trainingData <- read.csv("train_cleaned.csv", header = T, row.names = 1, stringsAsFactors = FALSE)
testData <- read.csv("test_cleaned.csv", header = T, stringsAsFactors = FALSE)

####################################################################
########################### USING TEXT2VEC #########################
####################################################################
it_train = itoken(trainingData$comment_text, 
                  preprocessor = tolower, 
                  tokenizer = word_tokenizer, 
                  ids = trainingData$id, 
                  progressbar = FALSE)

nFolds = 5
####################################################################
############################# NORMAL GLM ###########################
####################################################################

vocab = create_vocabulary(it_train,
                          stopwords = stopwords('en'))

pruned_vocab = prune_vocabulary(vocab, doc_proportion_min = 0.0001)

vectorizer = vocab_vectorizer(pruned_vocab)

dtm_train  = create_dtm(it_train, vectorizer)

cl <- makeCluster(6)
registerDoParallel(cl)

model.glm.auc <- foreach(col = names(trainingData[,3:8]), .combine=cbind, .packages='glmnet') %dopar% {
  model_glmnet = cv.glmnet(x = dtm_train, y = trainingData[[col]], 
                           family = 'binomial', 
                           type.measure = 'auc', 
                           nfolds = nFolds, 
                           alpha = 0)
  
  round(max(model_glmnet$cvm), 4)
}

stopCluster(cl)

####################################################################
############################ GLM 3-NGRAM ###########################
####################################################################
vocab = create_vocabulary(it_train,
                          stopwords = stopwords('en'),
                          ngram = c(1L, 3L))

pruned_vocab = prune_vocabulary(vocab,
                                doc_proportion_max = 0.5,
                                doc_proportion_min = 0.0001)

vectorizer = vocab_vectorizer(pruned_vocab)

dtm_train  = create_dtm(it_train, vectorizer)

cl <- makeCluster(6)
registerDoParallel(cl)

model.glm_ngram.auc <- foreach(col = names(trainingData[,3:8]), .combine=cbind, .packages='glmnet') %dopar% {
  model_glmnet = cv.glmnet(x = dtm_train, y = trainingData[[col]], 
                           family = 'binomial', 
                           type.measure = 'auc', 
                           nfolds = nFolds, 
                           alpha = 0)
  
  round(max(model_glmnet$cvm), 4)
}

stopCluster(cl)

####################################################################
######################### GLM 3-HASH-NGRAM #########################
####################################################################
h_vectorizer = hash_vectorizer(hash_size = 2 ^ 16, ngram = c(1L, 3L))

dtm_train = create_dtm(it_train, h_vectorizer)

cl <- makeCluster(6)
registerDoParallel(cl)

model.glm_hash_ngram.auc <- foreach(col = names(trainingData[,3:8]), .combine=cbind, .packages='glmnet') %dopar% {
  model_glmnet = cv.glmnet(x = dtm_train, y = trainingData[[col]], 
                           family = 'binomial', 
                           type.measure = 'auc', 
                           nfolds = nFolds, 
                           alpha = 0)
  
  round(max(model_glmnet$cvm), 4)
}

stopCluster(cl)

####################################################################
############################# GLM TFIDF ############################
####################################################################
vocab = create_vocabulary(it_train,
                          stopwords = stopwords('en'))

pruned_vocab = prune_vocabulary(vocab,
                          doc_proportion_min = 0.0001)

vectorizer = vocab_vectorizer(pruned_vocab)

dtm_train  = create_dtm(it_train, vectorizer)

tfidf = TfIdf$new()

dtm_train = fit_transform(dtm_train, tfidf)

cl <- makeCluster(6)
registerDoParallel(cl)

model.glm_tfidf.auc <- foreach(col = names(trainingData[,3:8]), .combine=cbind, .packages='glmnet') %dopar% {
  model_glmnet = cv.glmnet(x = dtm_train, y = trainingData[[col]], 
                           family = 'binomial', 
                           type.measure = 'auc', 
                           nfolds = nFolds, 
                           alpha = 0)
  
  round(max(model_glmnet$cvm), 4)
}

stopCluster(cl)

####################################################################
########################## GLM NGRAM-TFIDF #########################
####################################################################
vocab = create_vocabulary(it_train,
                          stopwords = stopwords('en'),
                          ngram = c(1L, 3L))

pruned_vocab = prune_vocabulary(vocab, doc_proportion_min = 0.0001)

vectorizer = vocab_vectorizer(pruned_vocab)

dtm_train  = create_dtm(it_train, vectorizer)
tfidf = TfIdf$new()

cl <- makeCluster(6)
registerDoParallel(cl)

model.glm_ngram_tfidf.auc <- foreach(col = names(trainingData[,3:8]), .combine=cbind, .packages='glmnet') %dopar% {
  model_glmnet = cv.glmnet(x = dtm_train, y = trainingData[[col]], 
                           family = 'binomial', 
                           type.measure = 'auc', 
                           nfolds = nFolds, 
                           alpha = 0)
  
  round(max(model_glmnet$cvm), 4)
}

stopCluster(cl)

mean(model.glm.auc)
mean(model.glm_ngram.auc)
mean(model.glm_hash_ngram.auc)
mean(model.glm_tfidf.auc)
mean(model.glm_ngram_tfidf.auc)

####################################################################
################### PREDICTION WITH THE BEST MODEL #################
####################################################################
it_train = itoken(trainingData$comment_text, 
                  tokenizer = word_tokenizer, 
                  ids = trainingData$id, 
                  progressbar = FALSE)

it_test = itoken(testData$comment_text,  
                 tokenizer = word_tokenizer, 
                 ids = testData$id, 
                 progressbar = FALSE)

vocab = create_vocabulary(it_train,
                          stopwords = stopwords('en'))
pruned_vocab = prune_vocabulary(vocab,
                          doc_proportion_min = 0.0001)
vectorizer = vocab_vectorizer(pruned_vocab)

dtm_train  = create_dtm(it_train, vectorizer)
tfidf = TfIdf$new()
dtm_train = fit_transform(dtm_train, tfidf)

dtm_test  = create_dtm(it_test, vectorizer)
dtm_test  = fit_transform(dtm_test, tfidf)

nFolds = 5
cl <- makeCluster(6)
registerDoParallel(cl)
model.glm_tfidf.pred <- foreach(col = names(trainingData[,3:8]), .combine=cbind, .packages='glmnet') %dopar% {
  model_glmnet = cv.glmnet(x = dtm_train, y = trainingData[[col]], 
                           family = 'binomial', 
                           type.measure = 'auc', 
                           nfolds = nFolds, 
                           alpha = 0)
  
  pred = predict.cv.glmnet(model_glmnet, newx = dtm_test, 
                           s = 'lambda.min', type = 'response')
  colnames(pred) <- c(col)
  pred
}
stopCluster(cl)

write.csv(model.glm_tfidf.pred, file = "submission_glm_ngrams_tfidf.csv", row.names = TRUE, quote = FALSE)

load('Environment.RData')
plot(model.glm.auc)

plotMatrix <- rbind(model.glm.auc,model.glm_ngram.auc,model.glm_hash_ngram.auc,model.glmnet_tfidf.auc,model.glmnet_ngram_tfidf.auc)
colnames(plotMatrix) <- c("Toxic","SevereToxic","Obscene", "Threat", "Insult", "IdentityHate")
rownames(plotMatrix) <- c("GLM","GLM-3-Gram","GLM-Hashing-3-Gram", "GLM-TfIdf", "GLM-3-Gram-TfIdf")

barplot(plotMatrix, main="Model AUC vs Comment Type",
        xlab="Comment type", ylab="AUC", col=c("red","green","blue","orange","purple"), beside=TRUE)
legend("bottom", cex = 0.75, legend = rownames(plotMatrix), 
       fill = c("red","green","blue","orange","purple"))

