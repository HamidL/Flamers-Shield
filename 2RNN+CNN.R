requiredPackages <- c("readr","tm", "rJava", "qdap", "plyr", "dplyr", "data.table","tidyverse", "qdapRegex", "caret")

library(keras)
install_keras(tensorflow = "gpu")


Sys.setenv(JAVA_HOME='C:\\Program Files\\Java\\jre1.8.0_201')


for (pac in requiredPackages) {
  if(!require(pac,  character.only=TRUE)){
    install.packages(pac, repos="http://cran.rstudio.com")
    library(pac,  character.only=TRUE)
  } 
}

rm(requiredPackages)
rm(pac)

train_data <- read.csv("train_cleaned.csv", header = T, row.names = 1, stringsAsFactors = FALSE)
test_data <- read.csv("test_cleaned.csv", header = T, stringsAsFactors = FALSE)

library(caret)
library(keras)

max_words = 130000
maxl = 200
wordseq = text_tokenizer(num_words = max_words) %>%
  fit_text_tokenizer(c(train_data$comment_text,test_data$comment_text))
#word dictionary
word_index = wordseq$word_index

x_train = texts_to_sequences(wordseq, train_data$comment_text ) %>%
  pad_sequences( maxlen = maxl)
y_train = as.matrix(train_data[,3:8])

x_test = texts_to_sequences(wordseq, test_data$comment_text ) %>%
  pad_sequences( maxlen = maxl)

wgt = fread("glove.840B.300d.txt", data.table = FALSE)  %>%
  rename(word=V1)  %>%
  mutate(word=gsub("[[:punct:]]"," ", rm_white(word) ))

dic_words = wgt$word
wordindex = unlist(wordseq$word_index)

dic = data.frame(word=names(wordindex), key = wordindex,row.names = NULL) %>%
  arrange(key) %>% 
  .[1:max_words,]

w_embed = dic %>% 
  left_join(wgt)

J = ncol(w_embed)
ndim = J-2
w_embed = w_embed [1:(max_words-1),3:J] %>%
  mutate_all(as.numeric) %>%
  mutate_all(round,6) %>%
  #fill na with 0
  mutate_all(funs(replace(., is.na(.), 0))) 

colnames(w_embed) = paste0("V",1:ndim)
w_embed = rbind(rep(0, ndim), w_embed) %>%
  as.matrix()

w_embed = list(array(w_embed , c(max_words, ndim)))


inp = layer_input(shape = list(maxl),
                  dtype = "int32", name = "input")
emm = inp %>%
  layer_embedding(input_dim = max_words, output_dim = ndim, input_length = maxl, weights = w_embed, trainable=FALSE) 
model = emm %>%
  layer_spatial_dropout_1d(rate=0.1) %>%
  bidirectional(
    layer_gru(units = 40, return_sequences = TRUE, recurrent_dropout = 0.1) 
  ) %>% 
  layer_conv_1d(
    60, 
    3, 
    padding = "valid",
    activation = "relu",
    strides = 1
  ) 

model1 = emm %>%
  layer_spatial_dropout_1d(rate=0.1) %>%
  bidirectional(
    layer_gru(units = 80, return_sequences = TRUE, recurrent_dropout = 0.1) 
  ) %>% 
  layer_conv_1d(
    120, 
    2, 
    padding = "valid",
    activation = "relu",
    strides = 1
  ) 

max_pool = model %>% layer_global_max_pooling_1d()
ave_pool = model %>% layer_global_average_pooling_1d()
max_pool1 = model1 %>% layer_global_max_pooling_1d()
ave_pool1 = model1 %>% layer_global_average_pooling_1d()

outp = layer_concatenate(list(ave_pool, max_pool,ave_pool1, max_pool1)) %>%
  layer_dense(units = 6, activation = "sigmoid")

model = keras_model(inp, outp)

model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = c("acc")
)


history = model %>% fit(
  x_train, y_train,
  epochs = 2,
  batch_size = 32,
  validation_split = 0.05,
  callbacks = list(
    callback_model_checkpoint(paste0("toxic_comment_model.h5"), save_best_only = TRUE),
    callback_early_stopping(monitor = "val_loss", min_delta = 0, patience = 0,
                            verbose = 0, mode = c("auto", "min", "max"))
  )
)


model = load_model_hdf5(paste0("toxic_comment_model.h5"))

cat("beginning the prediction & submission \n")
###########################################
#
# PREDICTION & SUBMISSON
#
###########################################

pred = model %>%
  predict(x_test, batch_size = 1024) %>%
  as.data.frame()

pred = cbind(id=test_data$id, pred) 

names(pred)[2:7] = c("toxic", "severe_toxic", "obscene", "threat","insult", "identity_hate")

write_csv(pred,"submission.csv")
