#Instalasi Packages yang Dibutuhkan untuk Melakukan Perancangan Model 
install.packages("textclean")
install.packages("katadasaR")
install.packages("tm")
install.packages("stringr")
install.packages("tokenizers")
install.packages("dplyr")
install.packages("RWeka")
install.packages("caret")
install.packages("e1071")
install.packages("shiny")
install.packages("ggplot2")
install.packages("stringr")
install.packages("topicmodels")
install.packages("tidytext")
install.packages("stringi")
install.packages("LDAvis")
install.packages("doParallel")
install.packages("scales")
install.packages("foreach")
install.packages("udpipe")
install.packages("text2vec")
install.packages("devtools")
install.packages("servr")
install_github("nurandi/katadasaR")
install.packages("plyr")
install.packages("ROSE")
install.packages("RJSONIO")

#Aktivasi Library yang Dibutuhkan untuk Melakukan Perancangan Model 
library(wordcloud)
library(textclean)
library(katadasaR)
library(tm)
library(stringr)
library(tokenizers)
library(dplyr)
library(RWeka)
library(caret)
library(e1071)
library(gmodels)
library(shiny)
library(ggplot2)
library(stringr)
library(topicmodels)
library(tidytext)
library(stringi)
library(LDAvis)
library(doParallel)
library(scales)
library(foreach)
library(udpipe)
library(text2vec)
library(devtools)
library(servr)
library(plyr)
library(ROSE)
library(RJSONIO)

nama_file <- "Training and Testing Set.csv"

tanggal_mulai <- "2019-05-01"

tanggal_selesai <- "2019-06-11"

versi_1 <- "4.40.4"
versi_2 <- "4.39.3"
versi_3 <- "4.38.2"
versi_4 <- "4.33.4"
versi_5 <- "4.35.3"
versi_6 <- "4.37.6"
versi_7 <- "4.34.2"
versi_8 <- "4.36.4"
versi_9 <- "4.9.2"
versi_10 <- "4.5.0"
versi_11 <- "4.23.1"
versi_12 <- "4.8.8"
versi_13 <- "4.6.3"
versi_14 <- "4.4.6"
versi_15 <- "4.7.4"
versi_16 <- "X"
versi_17 <- "X"
versi_18 <- "X"
versi_19 <- "X"
versi_20 <- "X"

#Membaca Dataset (Training dan Testing Set)
review <- read.csv(nama_file, header = TRUE, stringsAsFactors = FALSE)

#Menghapus Data yang Tidak Berkaitan

#Menghapus Data Ulasan yang Tidak Terdapat Ulasan Tekstual
review[!(!is.na(review$Review.Text) & review$Review.Text==""), ] 

#Menghapus Data Ulasan yang Tidak Berada pada Rentang Tanggal Terkait
gsub('.{10}$', '', review$Review.Last.Update.Date.and.Time)
review$Review.Last.Update.Date.and.Time <- 
  as.Date(review$Review.Last.Update.Date.and.Time, format = "%Y-%m-%d")
review[(review$Review.Last.Update.Date.and.Time > tanggal_mulai 
        & review$Review.Last.Update.Date.and.Time < tanggal_selesai),]

#Menghapus Data Ulasan yang Memiliki Rating 4 dan 5
review <- review %>%
  filter((Star.Rating == 1 | Star.Rating == 2 | Star.Rating == 3))

#Menghapus Data Ulasan yang Berasal Dari Aplikasi Berjenis Minor Version
review <- review %>%
  filter((App.Version.Name == versi_1 | App.Version.Name == versi_2 | App.Version.Name == versi_3 
          | App.Version.Name == versi_4 | App.Version.Name == versi_5 | App.Version.Name == versi_6 
          | App.Version.Name == versi_7 | App.Version.Name == versi_8 | App.Version.Name == versi_9 
          | App.Version.Name == versi_10 | App.Version.Name == versi_11 | App.Version.Name == versi_12 
          | App.Version.Name == versi_13 | App.Version.Name == versi_14 | App.Version.Name == versi_15 
          | App.Version.Name == versi_16 | App.Version.Name == versi_17 | App.Version.Name == versi_18 
          | App.Version.Name == versi_19 | App.Version.Name == versi_20))

#Menghapus Data Ulasan yang Duplikat
review <- distinct(review, Review.Submit.Date.and.Time, Review.Text, Device, App.Version.Name, .keep_all = TRUE)

#Filtering dan Case Folding (Prapemrosesan Data)
reviewtext <- review$Review.Text %>% 
  as.character()

reviewtext <- Corpus(VectorSource(reviewtext))

toSpace <- content_transformer(function (x , pattern ) gsub(pattern, " ", x))

reviewtext <- tm_map(reviewtext, toSpace, "/")
reviewtext <- tm_map(reviewtext, toSpace, "@")
reviewtext <- tm_map(reviewtext, toSpace, "\\|")
reviewtext <- tm_map(reviewtext, toSpace, "[[:punct:]]")
reviewtext <- tm_map(reviewtext, toSpace, "[[:digit:]]") 
reviewtext <- tm_map(reviewtext, content_transformer(tolower))

reviewtext <- tm_map(reviewtext, stripWhitespace)

reviewtext <- data.frame(text = get("content", reviewtext))

#Spelling Normalization (Prapemrosesan Data)
reviewtext <- reviewtext$text %>% 
  as.character()

spell.lex <- read.csv("colloquial-indonesian-lexicon.csv", stringsAsFactors = FALSE)

reviewtext <- replace_internet_slang(reviewtext, slang = paste0("\\b",
                                                                spell.lex$slang, "\\b"),
                                     replacement = spell.lex$formal, ignore.case = TRUE)

reviewtext <- strip(reviewtext)

reviewtext <- reviewtext %>%
  as.data.frame()

#Stemming (Prapemrosesan Data)
reviewtext <- reviewtext$. %>% 
  as.character()

stemming <- function(x){paste(lapply(x,katadasar),collapse = " ")}
reviewtext <- lapply(tokenize_words(reviewtext[]),stemming)

reviewtext <- as.character(reviewtext)

reviewtext <- reviewtext %>%
  as.data.frame()

#Stopwords Removal (Prapemrosesan Data)
reviewtext <- Corpus(VectorSource(reviewtext$.))

myStopwords = readLines("stopwordbahasa.csv")

reviewtext <- tm_map(reviewtext, removeWords, myStopwords)

reviewtext <- tm_map(reviewtext, stripWhitespace)

reviewtext <- data.frame(text = get("content", reviewtext))

reviewtext <- data.frame(lapply(reviewtext, trimws), stringsAsFactors = FALSE)

review$Review.Text <- NULL 

review <- data.frame(review, reviewtext)

colnames(review)[16] <- "Review.Text"

#Menghapus Data Ulasan yang Kosong akibat Prapemrosesan Data
review <-  review[!(is.na(review$Review.Text) | review$Review.Text==""), ]

#Mengacak Data Ulasan
review <- review[sample(nrow(review)),]

#Membagi Set Data menjadi Training Set dan Testing Set
set.seed(1234)
indexes <- createDataPartition(review$Kategori, p = 0.8, list = FALSE)

review.train <- review[indexes,]
review.test <- review[-indexes,]

#Handling Imbalance Data
kategori.train <- review$Kategori[indexes]
kategori.train.df <- kategori.train %>%
  as.data.frame()

kategori.train.technical <- kategori.train.df %>%
  filter((. == "Technical"))

N_technical_train <- nrow(kategori.train.technical)

N_non_technical_train <- nrow(kategori.train.df)-N_technical_train

#Melakukan Undersampling pada Training Set
review.train <- ovun.sample(Kategori ~., data = review.train, method = "under", N = 2*N_non_technical_train, seed = 1)$data

#Melakukan Oversampling pada Training Set
review.train <- ovun.sample(Kategori ~., data = review.train, method = "over", N = 2*N_technical_train)$data

#Transformasi Data Teks menjadi Corpus
review.train.corpus <- Corpus(VectorSource(review.train$Review.Text))
review.test.corpus <- Corpus(VectorSource(review.test$Review.Text))

#Tokenisasi Teks
my_tokenize <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 1))

#Vektorisasi Teks dan Dokumen
dtm.review.train <- DocumentTermMatrix(review.train.corpus, 
                                       control = list(tokenize = my_tokenize, 
                                                      weighting = function(x) weightTfIdf(x, normalize = TRUE)))

dtm.review.test <- DocumentTermMatrix(review.test.corpus, 
                                      control = list(tokenize = my_tokenize, 
                                                     weighting = function(x) weightTfIdf(x, normalize = TRUE)))

dtm.review.train <- as.matrix(dtm.review.train)
dtm.review.test <- as.matrix(dtm.review.test)

#Membuat Matriks Testing Set dengan Menyamakan Jumlah Kolom dengan Training Set
testmat_testtrain <- function(dtm.review.train, dtm.review.test){	
  
  train_mat_cols <- colnames(dtm.review.train)
  dtm.review.test <- dtm.review.test[, colnames(dtm.review.test) %in% train_mat_cols]
  df_dtm.review.test <- as.data.frame(dtm.review.test)
  
  miss_names 	<- train_mat_cols[!(train_mat_cols %in% colnames(dtm.review.test))]
  miss_names
  dtm.review_new_kolom <- colnames(dtm.review.test)
  
  kolom_sama <- c(1:ncol(dtm.review.test))
  
  for (i in 1:ncol(dtm.review.test)) {
    for (j in 1:length(train_mat_cols)) {
      if(train_mat_cols[j] == dtm.review_new_kolom[i]) {
        kolom_sama[i] <- j
      }
    }
  }
  
  if(length(miss_names)!=0){
    colClasses  <- rep("numeric", length(miss_names))
    df 			<- read.table(text = '', colClasses = colClasses, col.names = miss_names)
    df[1:nrow(dtm.review.test),] <- 0
    
    df_baru <- as.data.frame(dtm.review.train[1:nrow(dtm.review.test),])
    for (a in 1:length(kolom_sama)) {
      df_baru[,kolom_sama[a]] <- df_dtm.review.test[,a]
    }
    
    for (b in 1:length(miss_names)) {
      for (c in 1:ncol(df_baru)) {
        if (train_mat_cols[c] == miss_names[b]) {
          df_baru[,c] <- df[,b]
        }
      }
    }
  }
  
  as.matrix(df_baru)
}

dtm.review.test <- testmat_testtrain(dtm.review.train,dtm.review.test)

#Optimasi Hyperparameter Model SVM untuk Kernel Linear, Radial, dan Polinomial
kategori.train <- review.train$Kategori

tune.out.linear = tune(svm, train.x = dtm.review.train, 
                       train.y = as.factor(kategori.train), 
                       kernel ="linear", 
                       ranges =list(cost=c(0.001,0.01,0.1, 1,5,10,100, 1000)))

tune.out.radial = tune(svm, train.x = dtm.review.train, 
                       train.y = as.factor(kategori.train), 
                       kernel ="radial", 
                       ranges =list(cost=c(0.001,0.01,0.1, 1,5,10,100, 1000), 
                                    gamma = 10^(-5:-1)))

tune.out.polinomial = tune(svm, train.x = dtm.review.train, 
                                    train.y = as.factor(kategori.train), 
                                    kernel ="polynomial", 
                                    ranges =list(cost=c(0.1, 1,5,10,100), gamma = 10^(-1:1), 
                                                 coef0 = c(0.1, 1, 10), degree= c(2,3,4))
summary(tune.out.linear)
summary(tune.out.radial)
summary(tune.out.polinomial.degree.2)

model_SVM <- svm(dtm.review.train, as.factor(kategori.train), kernel = "polynomial", cost = 1, gamma = 0.1, coef0 = 1, degree = 2)

#Prediksi Kelas dari Testing Data dengan Menggunakan Model SVM  
predict_testSVM <- predict(model_SVM, newdata = dtm.review.test)

#Prediksi Kelas dari Training Data dengan Menggunakan Model SVM
predict_trainSVM <- predict(model_SVM, newdata = dtm.review.train)

table(predict_testSVM, review.test$Kategori)
mean(predict_testSVM == review.test$Kategori)

table(predict_trainSVM, review.train$Kategori)
mean(predict_trainSVM == review.train$Kategori)

#Pembuatan Confusion Matrix untuk Model SVM
CrossTable(predict_testSVM, review.test$Kategori,
           prop.chisq = FALSE,
           prop.t = FALSE, prop.r = FALSE,
           dnn = c("predicted","actual"))

CrossTable(predict_trainSVM, review.train$Kategori,
           prop.chisq = FALSE,
           prop.t = FALSE, prop.r = FALSE,
           dnn = c("predicted","actual"))

#Merubah Nilai pada Document Term Matrix menjadi Nilai Boolean
convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

trainNB_binary <- apply(dtm.review.train, MARGIN = 2, convert_counts)
testNB_binary <- apply(dtm.review.test, MARGIN = 2, convert_counts)

code_1.nb <- as.factor(kategori.train)

#Melakukan Optimisasi Hyperparameter Smoothing Laplace 
cluster_NB <- makeCluster(detectCores(logical = TRUE) - 1) # leave one CPU spare...
registerDoParallel(cluster_NB)

clusterEvalQ(cluster_NB, {
  library(e1071)
  library(gmodels)
})

folds_NB <- 10
splitfolds_NB <- sample(1:folds_NB, nrow(dtm.review.train), replace = TRUE)
candidate_laplace <- c(0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5) 
clusterExport(cluster_NB, c("trainNB_binary", "splitfolds_NB", "folds_NB", "candidate_laplace"))

system.time({
  results_NB <- foreach(j = 1:length(candidate_laplace), .combine = rbind) %dopar%{
    laplace <- candidate_laplace[j]
    results_laplace <- matrix(0, nrow = folds_NB, ncol = 2)
    colnames(results_laplace) <- c("laplace", "error_rate")
    for(i in 1:folds_NB){
      
      error_valid_NB <- numeric(1)
      
      code_train_NB <- as.factor(kategori.train[splitfolds_NB != i])
      
      train_set_NB <- trainNB_binary[splitfolds_NB != i , ]
      valid_set_NB <- trainNB_binary[splitfolds_NB == i, ]
      
      fitted_NB <- naiveBayes(as.matrix(train_set_NB), code_train_NB, laplace = laplace)
      
      predict_valid_set <- predict(fitted_NB, newdata = valid_set_NB)
      
      code_valid_NB <- as.factor(kategori.train[splitfolds_NB == i])
      code_valid_NB <- factor(code_valid_NB, levels = levels(predict_valid_set))
      
      for (j in 1:nrow(valid_set_NB)){
        if(code_valid_NB[j] != predict_valid_set[j]){
          error_valid_NB <- error_valid_NB + 1
        }
      }
      
      results_laplace[i,] <- c(laplace, sqrt(mean(error_valid_NB^2)))
    }
    return(results_laplace)
  }
})
stopCluster(cluster_NB)

results_df_NB <- as.data.frame(results_NB)
results_df_NB[,-1] <- results_df_NB[,-1]/((nrow(trainNB_binary)/10))
results_df_NB_mean <- ddply(results_df_NB, .(laplace), summarize,  error_rate = mean(error_rate))
error_rate_min <- min(results_df_NB_mean[,"error_rate"])

# Membuat Model NB dengan Nilai Parameter Smoothing Laplace Terbaik
best_laplace_val <- results_df_NB_mean[results_df_NB_mean$error_rate == error_rate_min, "laplace"]

model_NB <- naiveBayes(as.matrix(trainNB_binary), code_1.nb, laplace = 1)

#Prediksi Kelas dari Testing Data dengan Menggunakan Model NB  
predict_testNB <- predict(model_NB, newdata = testNB_binary)
mean(predict_testNB == review.test$Kategori)

#Pembuatan Confusion Matrix untuk Model NB
CrossTable(predict_testNB, review.test$Kategori,
           prop.chisq = FALSE,
           prop.t = FALSE, prop.r = FALSE,
           dnn = c("predicted","actual"))

#Prediksi Kelas dari Training Data dengan Menggunakan Model NB
predict_trainNB <- predict(model_NB, newdata = trainNB_binary)
mean(predict_trainNB == review.train$Kategori)

CrossTable(predict_trainNB, review.train$Kategori,
           prop.chisq = FALSE,
           prop.t = FALSE, prop.r = FALSE,
           dnn = c("predicted","actual"))

#Pengecekan Overfitting
folds <- cut(seq(1, nrow(dtm.review.train)), breaks = 10, labels = FALSE)
nb_error.valid <- c(1:10)

for (i in 1:10){
  indexes.nb <- which(folds == i, arr.ind = TRUE)
  dtm.train.nb <- dtm.review.train[-indexes.nb,]
  dtm.valid.nb <- dtm.review.train[indexes.nb,]
  
  error_train.nb <- numeric(1)
  error_valid.nb <- numeric(1)
  
  code_1.nb <- as.factor(review.train$Kategori[-indexes.nb])
  
  convert_counts <- function(x) {
    x <- ifelse(x > 0, "Yes", "No")
  }
  
  train_NB_binary <- apply(dtm.train.nb, MARGIN = 2, convert_counts)
  valid_NB_binary <- apply(dtm.valid.nb, MARGIN = 2, convert_counts)
  
  model_NB <- naiveBayes(as.matrix(train_NB_binary), 
                         code_1.nb, laplace = 0)
  
  predict_valid_NB <- predict(model_NB, newdata = valid_NB_binary)
    
  code_2.nb <- as.factor(review.train$Kategori[indexes.nb])
  code_2.nb <- factor(code_2.nb, levels = levels(predict_valid_NB))
  
  for (j in 1:nrow(dtm.valid.nb)){
    if(code_2.nb[j] != predict_valid_NB[j]){
      error_valid.nb <- error_valid.nb + 1
    }
  }
  nb_error.valid[i] <- sqrt(mean(error_valid.nb^2))
  print(nb_error.valid[i])
  
  print(table(predict_valid_NB,code_2.nb))  
}

error_rate_NB <- c(1:10)

for (i in 1:10) {
  error_rate_NB[i] <- (nb_error.valid[i]/((nrow(review.train)/10)))*100                      
}

mean(error_rate_NB)
error_rate_NB

###############################################################################################################

file_name <- "Score Set.csv"

start_date <- "2019-08-12"

finish_date <- "2019-08-26"

app_version_1 <- "4.34.2"
app_version_2 <- "4.35.2"
app_version_3 <- "4.36.4"
app_version_4 <- "4.37.6"
app_version_5 <- "4.38.2"
app_version_6 <- "4.39.3"
app_version_7 <- "4.4.6"
app_version_8 <- "4.40.4"
app_version_9 <- "4.41.4"
app_version_10 <- "4.42.4"
app_version_11 <- "4.43.1"
app_version_12 <- "4.44.3"
app_version_13 <- "4.45.3"
app_version_14 <- "4.6.3"
app_version_15 <- "4.7.4"
app_version_16 <- "4.8.8"
app_version_17 <- "4.9.2"
app_version_18 <- "X"
app_version_19 <- "X"
app_version_20 <- "X"

#Membaca Dataset (Score Set)
review_new <- read.csv(file_name, header = TRUE, stringsAsFactors = FALSE)

review_new$ID <- seq.int(nrow(review_new))

review.new <- review_new

#Menghapus Data yang Tidak Berkaitan pada Score Set

#Menghapus Data Ulasan yang Tidak Terdapat Ulasan Tekstual pada Score Set
review_new[!(!is.na(review_new$Review.Text) & review_new$Review.Text==""), ] 

#Menghapus Data Ulasan yang Tidak Berada pada Rentang Tanggal Terkait pada Score Set
gsub('.{10}$', '', review_new$Review.Last.Update.Date.and.Time)
review_new$Review.Last.Update.Date.and.Time <- as.Date(review_new$Review.Last.Update.Date.and.Time, format = "%Y-%m-%d")
review_new[(review_new$Review.Last.Update.Date.and.Time > tanggal_mulai & review_new$Review.Last.Update.Date.and.Time < tanggal_selesai),]

#Menghapus Data Ulasan yang Memiliki Rating 4 dan 5 pada Score Set
review_new <- review_new %>%
  filter((Star.Rating == 1 | Star.Rating == 2 | Star.Rating == 3))

#Menghapus Data Ulasan yang Berasal Dari Aplikasi Berjenis Minor Version pada Score Set
review_new <- review_new %>%
  filter((App.Version.Name == app_version_1 | App.Version.Name == app_version_2 | App.Version.Name == app_version_3 | App.Version.Name == app_version_4
          | App.Version.Name == app_version_5 | App.Version.Name == app_version_6 | App.Version.Name == app_version_7 | App.Version.Name == app_version_8 
          | App.Version.Name == app_version_9 | App.Version.Name == app_version_10 | App.Version.Name == app_version_11 | App.Version.Name == app_version_12 
          | App.Version.Name == app_version_13 | App.Version.Name == app_version_14 | App.Version.Name == app_version_15 | App.Version.Name == app_version_16 
          | App.Version.Name == app_version_17 | App.Version.Name == app_version_18 | App.Version.Name == app_version_19 | App.Version.Name == app_version_20))

#Menghapus Data Ulasan yang Duplikat pada Score Set
review_new <- distinct(review_new, Review.Submit.Date.and.Time, Review.Text, Device, App.Version.Name, .keep_all = TRUE)
reviewtext_new <- review_new$Review.Text %>% 
  as.character()

#Filtering dan Case Folding pada Score Set
reviewtext_new <- Corpus(VectorSource(reviewtext_new))

reviewtext_new <- tm_map(reviewtext_new, toSpace, "/")
reviewtext_new <- tm_map(reviewtext_new, toSpace, "@")
reviewtext_new <- tm_map(reviewtext_new, toSpace, "\\|")
reviewtext_new <- tm_map(reviewtext_new, content_transformer(tolower))
reviewtext_new <- tm_map(reviewtext_new, toSpace, "[[:punct:]]")
reviewtext_new <- tm_map(reviewtext_new, toSpace, "[[:digit:]]") 

reviewtext_new <- tm_map(reviewtext_new, stripWhitespace)

reviewtext_new <- data.frame(text = get("content", reviewtext_new))

reviewtext_new <- reviewtext_new$text %>% 
  as.character

#Spelling Normalization pada Score Set
spell.lex.for.LDA <- read.csv("colloquial-indonesian-lexicon for LDA.csv", stringsAsFactors = FALSE)

reviewtext_new <- replace_internet_slang(reviewtext_new, slang = paste0("\\b",
                                                                        spell.lex.for.LDA$slang, "\\b"),
                                         replacement = spell.lex.for.LDA$formal, ignore.case = TRUE)

reviewtext_new <- strip(reviewtext_new)

reviewtext_new <- reviewtext_new %>%
  as.data.frame()

reviewtext_new <- reviewtext_new$. %>% 
  as.character()

#Stemming pada Score Set
reviewtext_new <- lapply(tokenize_words(reviewtext_new[]),stemming)

reviewtext_new <- as.character(reviewtext_new)

reviewtext_new <- reviewtext_new %>%
  as.data.frame()

reviewtext_new <- Corpus(VectorSource(reviewtext_new$.))

myStopwords_for_LDA = readLines("stopwordbahasa for LDA.csv")

reviewtext_new <- tm_map(reviewtext_new, removeWords, myStopwords_for_LDA)

reviewtext_new <- tm_map(reviewtext_new, stripWhitespace)

reviewtext_new <- data.frame(text = get("content", reviewtext_new))

reviewtext_new <- data.frame(lapply(reviewtext_new, trimws), stringsAsFactors = FALSE)

review_new$Review.Text <- NULL 

review_new <- data.frame(review_new, reviewtext_new)

colnames(review_new)[17] <- "Review.Text"

review_new <-  review_new[!(is.na(review_new$Review.Text) | review_new$Review.Text==""), ]

reviewcorpus_new <- Corpus(VectorSource(review_new$Review.Text))

dtm.review_new <- DocumentTermMatrix(reviewcorpus_new, control = list(tokenize = my_tokenize, weighting = function(x) weightTfIdf(x, normalize = TRUE)))

dtm.review.train <- as.matrix(dtm.review.train)
dtm.review_new.m <- as.matrix(dtm.review_new)

testmat <- function(dtm.review.train, dtm.review_new.m){	
  
  train_mat_cols <- colnames(dtm.review.train)
  dtm.review_new.m 	<- dtm.review_new.m[, colnames(dtm.review_new.m) %in% train_mat_cols]
  df_dtm.review_new.m <- as.data.frame(dtm.review_new.m)
  
  miss_names 	<- train_mat_cols[!(train_mat_cols %in% colnames(dtm.review_new.m))]
  miss_names
  dtm.review_new_kolom <- colnames(dtm.review_new.m)
  
  kolom_sama <- c(1:ncol(dtm.review_new.m))
  
  for (i in 1:ncol(dtm.review_new.m)) {
    for (j in 1:length(train_mat_cols)) {
      if(train_mat_cols[j] == dtm.review_new_kolom[i]) {
        kolom_sama[i] <- j
      }
    }
  }
  
  if(length(miss_names)!=0){
    colClasses  <- rep("numeric", length(miss_names))
    df 			<- read.table(text = '', colClasses = colClasses, col.names = miss_names)
    df[1:nrow(dtm.review_new.m),] <- 0
    
    df_baru <- as.data.frame(dtm.review.train[1:nrow(dtm.review_new.m),])
    for (a in 1:length(kolom_sama)) {
      df_baru[,kolom_sama[a]] <- df_dtm.review_new.m[,a]
    }
    
    for (b in 1:length(miss_names)) {
      for (c in 1:ncol(df_baru)) {
        if (train_mat_cols[c] == miss_names[b]) {
          df_baru[,c] <- df[,b]
        }
      }
    }
  }
  
  as.matrix(df_baru)
}

dtm.review_new.match.nb <- testmat(dtm.review.train,dtm.review_new.m)
newNB_binary <- apply(dtm.review_new.match.nb, MARGIN = 2, convert_counts)
predict_newNB <- predict(model_NB, newNB_binary)
print(table(predict_newNB))

predict_newNB.df <- data.frame(Kategori = predict_newNB)

review_new_predicted <- data.frame(review_new, predict_newNB.df)

write.csv(review_new_predicted, "review technical.csv")

review_new_predicted_technical <- review_new_predicted %>%
  filter(Kategori == "Technical")

#Tokenisasi Score Set dengan Kelas Technical

tokens <- review_new_predicted_technical$Review.Text %>% word_tokenizer()

names(tokens) <- review_new_predicted_technical$ID

it <- itoken(tokens)
vocab <- create_vocabulary(it, ngram = c(1L, 1L))

pruned_vocab = prune_vocabulary(vocab,  term_count_min = 1, doc_proportion_max = 1)

v_vectorizer <- vocab_vectorizer(pruned_vocab)

#Pembuatan Document Term Matrix untuk Input Topic Modelling

dtm.review.technical <- create_dtm(it, v_vectorizer)

dtm.review.technical <- as.matrix(dtm.review.technical)

# Penentuan Kandidat Jumlah Topik 

candidate_k <- c(2:40) # Kandidat Jumlah Topik

burnin = 1000
iter = 2000
keep = 50

cluster <- makeCluster(detectCores(logical = TRUE) - 1) # leave one CPU spare...
registerDoParallel(cluster)
clusterEvalQ(cluster, {
  library(topicmodels)
})

#Perhitungan Jumlah Topik Optimal dengan Nilai Perplexity Menggunakan 10 Fold Cross Validation

folds <- 10
splitfolds <- sample(1:folds, nrow(review_new_predicted_technical), replace = TRUE)
clusterExport(cluster, c("dtm.review.technical", "burnin", "iter", 
                         "keep", "splitfolds", "folds", "candidate_k"))
system.time({
  results <- foreach(j = 1:length(candidate_k), .combine = rbind) %dopar%{
    k <- candidate_k[j]
    results_1k <- matrix(0, nrow = folds, ncol = 2)
    colnames(results_1k) <- c("k", "perplexity")
    for(i in 1:folds){
      train_set <- dtm.review.technical[splitfolds != i , ]
      valid_set <- dtm.review.technical[splitfolds == i, ]
      
      fitted <- LDA(train_set, k = k, method = "Gibbs",
                    control = list(burnin = burnin, iter = iter, keep = keep))
      results_1k[i,] <- c(k, perplexity(fitted, newdata = valid_set))
    }
    return(results_1k)
  }
})
stopCluster(cluster)

results_df <- as.data.frame(results)

results_df <- ddply(results_df, .(k), summarize,  perplexity=mean(perplexity))
perplexity_min <- min(results_df[,"perplexity"])

ggplot(results_df, aes(x = k, y = perplexity)) +
  geom_point() +
  geom_smooth(se = FALSE) +
  ggtitle("Pemodelan LDA dengan 10-fold Cross Validation pada Data Ulasan PT X") +
  labs(x = "Kandidat Jumlah Topik", 
       y = "Rata-Rata Perplexity dari Validation Set")

# Membuat Model LDA dengan Jumlah Topik Optimal

best_number_of_topics <- results_df[results_df$perplexity == perplexity_min, "k"]

seed <- list(10005,765)

model_LDA <- LDA(x=dtm.review.technical, k=best_number_of_topics, method="Gibbs",
                 control=list(alpha=0.1, delta=0.01, iter=iter, seed=seed))

#Pembuatan Visualisasi Model LDA 

topicmodels2LDAvis <- function(x, ...){
  post <- topicmodels::posterior(x)
  if (ncol(post[["topics"]]) < 3) stop("Model harus memiliki lebih dari 2 topik")
  mat <- x@wordassignments
  LDAvis::createJSON(
    phi = post[["terms"]], 
    theta = post[["topics"]],
    vocab = colnames(post[["terms"]]),
    doc.length = slam::row_sums(mat, na.rm = TRUE),
    term.frequency = slam::col_sums(mat, na.rm = TRUE)
  )
}

serVis(topicmodels2LDAvis(model_LDA))

#Melihat Dokumen yang Memiliki Nilai Probabilitas Topik pada Dokumen yang Tinggi 
topic <- RJSONIO::fromJSON(topicmodels2LDAvis(model_LDA))$topic.order
topic <- as.data.frame(topic)
topic$new.order <- seq.int(nrow(topic))

td_lda_docs <- tidy(model_LDA, matrix = "gamma")
doc_classes <- td_lda_docs %>%
  group_by(document) %>%
  top_n(1, gamma) %>%
  ungroup()
doc_classes %>%
  arrange(gamma)

colnames(doc_classes)[1] <- "ID"
doc_classes <- doc_classes %>%  
  mutate(ID = as.numeric(ID))
doc_classes <- dplyr::left_join(doc_classes, review.new, by = "ID")
required_doc_classes_column <- as.vector(c("ID","topic","gamma","Review.Text"))
doc_classes <- doc_classes[,required_doc_classes_column] 
doc_classes <- dplyr::left_join(doc_classes, topic, by = "topic")
required_doc_classes_column <- as.vector(c("ID","topic","gamma","Review.Text","new.order"))
doc_classes <- doc_classes[,required_doc_classes_column] 
doc_classes <- doc_classes[c("ID", "new.order", "topic","Review.Text","gamma")]
doc_classes[with(doc_classes, order(new.order, -gamma)), ]

colnames(doc_classes)[2] <- "Urutan Prioritas Topik"
colnames(doc_classes)[3] <- "Topik"
colnames(doc_classes)[4] <- "Data Ulasan"
colnames(doc_classes)[5] <- "Probabilitas Topik Terkait pada Dokumen"

doc_classes$Review.Text <- NULL

write.csv(doc_classes, "doc_classes.csv")

reviewcoba <- read.csv("reviewtesttt.csv", header = TRUE, stringsAsFactors = FALSE)

#Filtering dan Case Folding (Prapemrosesan Data)

reviewtextcoba <- reviewcoba$Review.Text %>% 
  as.character()

reviewtextcoba <- Corpus(VectorSource(reviewtextcoba))

toSpace <- content_transformer(function (x , pattern ) gsub(pattern, " ", x))

reviewtextcoba <- tm_map(reviewtextcoba, toSpace, "/")
reviewtextcoba <- tm_map(reviewtextcoba, toSpace, "@")
reviewtextcoba <- tm_map(reviewtextcoba, toSpace, "\\|")
reviewtextcoba <- tm_map(reviewtextcoba, toSpace, "[[:punct:]]")
reviewtextcoba <- tm_map(reviewtextcoba, toSpace, "[[:digit:]]") 
reviewtextcoba <- tm_map(reviewtextcoba, content_transformer(tolower))

reviewtextcoba <- tm_map(reviewtextcoba, stripWhitespace)

reviewtextcoba <- data.frame(text = get("content", reviewtextcoba))

#Spelling Normalization (Prapemrosesan Data)

reviewtextcoba <- reviewtextcoba$text %>% 
  as.character()

spell.lex <- read.csv("colloquial-indonesian-lexicon.csv", stringsAsFactors = FALSE)

reviewtextcoba <- replace_internet_slang(reviewtextcoba, slang = paste0("\\b",
                                                                        spell.lex$slang, "\\b"),
                                         replacement = spell.lex$formal, ignore.case = TRUE)

reviewtextcoba <- strip(reviewtextcoba)

reviewtextcoba <- reviewtextcoba %>%
  as.data.frame()

#Stemming (Prapemrosesan Data)

reviewtextcoba <- reviewtextcoba$. %>% 
  as.character()

stemming <- function(x){paste(lapply(x,katadasar),collapse = " ")}
reviewtextcoba <- lapply(tokenize_words(reviewtextcoba[]),stemming)

reviewtextcoba <- as.character(reviewtextcoba)

reviewtextcoba <- reviewtextcoba %>%
  as.data.frame()

#Stopwords Removal (Prapemrosesan Data)

reviewtextcoba <- Corpus(VectorSource(reviewtextcoba$.))

myStopwords = readLines("stopwordbahasa.csv")

reviewtextcoba <- tm_map(reviewtextcoba, removeWords, myStopwords)

reviewtextcoba <- tm_map(reviewtextcoba, stripWhitespace)

reviewtextcoba <- data.frame(text = get("content", reviewtextcoba))

reviewtextcoba <- data.frame(lapply(reviewtextcoba, trimws), stringsAsFactors = FALSE)

reviewcoba$Review.Text <- NULL 

reviewcoba <- data.frame(reviewcoba, reviewtextcoba)

colnames(reviewcoba)[16] <- "Review.Text"

review.test.corpus.coba <- Corpus(VectorSource(reviewcoba$Review.Text))

my_tokenize <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = 1))

dtm.review.test.coba <- DocumentTermMatrix(review.test.corpus.coba, 
                                           control = list(tokenize = my_tokenize, 
                                                          weighting = function(x) weightTfIdf(x, normalize = TRUE)))

dtm.review.test.coba <- as.matrix(dtm.review.test.coba)

testmat_testtrain <- function(dtm.review.train, dtm.review.test.coba){	
  
  train_mat_cols <- colnames(dtm.review.train)
  dtm.review.test.coba <- dtm.review.test.coba[, colnames(dtm.review.test.coba) %in% train_mat_cols]
  df_dtm.review.test.coba <- as.data.frame(dtm.review.test.coba)
  
  miss_names 	<- train_mat_cols[!(train_mat_cols %in% colnames(dtm.review.test.coba))]
  miss_names
  dtm.review_new_kolom <- colnames(dtm.review.test.coba)
  
  kolom_sama <- c(1:ncol(dtm.review.test.coba))
  
  for (i in 1:ncol(dtm.review.test.coba)) {
    for (j in 1:length(train_mat_cols)) {
      if(train_mat_cols[j] == dtm.review_new_kolom[i]) {
        kolom_sama[i] <- j
      }
    }
  }
  
  if(length(miss_names)!=0){
    colClasses  <- rep("numeric", length(miss_names))
    df 			<- read.table(text = '', colClasses = colClasses, col.names = miss_names)
    df[1:nrow(dtm.review.test.coba),] <- 0
    
    df_baru <- as.data.frame(dtm.review.train[1:nrow(dtm.review.test.coba),])
    for (a in 1:length(kolom_sama)) {
      df_baru[,kolom_sama[a]] <- df_dtm.review.test.coba[,a]
    }
    
    for (b in 1:length(miss_names)) {
      for (c in 1:ncol(df_baru)) {
        if (train_mat_cols[c] == miss_names[b]) {
          df_baru[,c] <- df[,b]
        }
      }
    }
  }
  
  as.matrix(df_baru)
}

dtm.review.test.coba <- testmat_testtrain(dtm.review.train,dtm.review.test.coba)

testNB_binary.coba <- apply(dtm.review.test.coba, MARGIN = 2, convert_counts)

#Prediksi Kelas dari Testing Data dengan Menggunakan Model NB  
predict_testNB.coba <- predict(model_NB, newdata = testNB_binary.coba)
mean(predict_testNB.coba == reviewcoba$Kategori)

#Pembuatan Confusion Matrix untuk Model NB
CrossTable(predict_testNB.coba, reviewcoba$Kategori,
           prop.chisq = FALSE,
           prop.t = FALSE, prop.r = FALSE,
           dnn = c("predicted","actual"))

# Penentuan Kandidat Jumlah Topik 

candidate_alpha <- c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0,4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8,4.9,5) # Kandidat Jumlah Topik

burnin = 1000
iter = 1000
keep = 50
k = 18
delta = 0.01

cluster <- makeCluster(detectCores(logical = TRUE) - 1) # leave one CPU spare...
registerDoParallel(cluster)
clusterEvalQ(cluster, {
  library(topicmodels)
})

#Perhitungan Nilai Perplexity Menggunakan 10 Fold Cross Validation

folds <- 10
splitfolds <- sample(1:folds, nrow(review_new_predicted_technical), replace = TRUE)
clusterExport(cluster, c("dtm.review.technical", "burnin", "iter", 
                         "keep", "splitfolds", "folds", "candidate_alpha"))
system.time({
  results <- foreach(j = 1:length(candidate_alpha), .combine = rbind) %dopar%{
    alpha <- candidate_alpha[j]
    results_1k <- matrix(0, nrow = folds, ncol = 2)
    colnames(results_1k) <- c("alpha", "perplexity")
    for(i in 1:folds){
      train_set <- dtm.review.technical[splitfolds != i , ]
      valid_set <- dtm.review.technical[splitfolds == i, ]
      
      fitted <- LDA(train_set, k = k, method = "Gibbs",
                    control = list(burnin = burnin, iter = iter, alpha = alpha, keep = keep, delta = delta))
      results_1k[i,] <- c(alpha, perplexity(fitted, newdata = valid_set))
    }
    return(results_1k)
  }
})
stopCluster(cluster)

results_df <- as.data.frame(results)

results_df <- ddply(results_df, .(alpha), summarize,  perplexity=mean(perplexity))
perplexity_min <- min(results_df[,"perplexity"])

ggplot(results_df, aes(x = alpha, y = perplexity)) +
  geom_point() +
  geom_smooth(se = FALSE) +
  ggtitle("10-fold cross-validation of topic modelling with the PT X review",
          "(ie 10 different models fit for each candidate number of topics)") +
  labs(x = "Candidate number of topics", 
       y = "Perplexity when fitting the trained model to the hold-out set")

#Menampilkan distribusi kata pada tiap topik

text_topics <- tidy(model_LDA, matrix = "beta")

text_top_terms <- text_topics %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

text_top_terms %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip()