#Ben Waetford
#HX Movielens project
#May 2022


# enable multiple workers for parallel processing

library(doParallel)
nworkers <- makePSOCKcluster(20)
registerDoParallel(nworkers)


# run harvardx code


##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")


# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


#Movielens Project
#load libraries

library(ggplot2)
library(corrplot)
library(dplyr)
library(lubridate)
library(moments)


#load data
# change timestamp into date data type, and create new variables from the date

myedx <- edx %>%
  mutate(reviewdt = as.POSIXct(edx$timestamp,origin = "1970-01-01"),
         review_y = round_date(reviewdt, unit = "year"), #the year the review was made
         INTreview_y = as.integer(year(reviewdt)), #review year as an integer to calculate span later
         review_ym = round_date(reviewdt, unit = "month"), #the yearmonth the review was made
         review_ymd = round_date(reviewdt, unit = "day"), #the yearmonthday the review was made
         release_y = as.integer(sub("\\).*", "", sub(".*\\(", "", edx$title))), #the year the film was released
         span_y = (INTreview_y - release_y), #the approximate age (in years) of the movie when it was rated
         howManyGenres = str_count(edx$genres, "\\|")+1) #the number of genres associated with the film


#create a test set and validation set from edx

set.seed(1976, sample.kind="Rounding") 
test_index <- createDataPartition(y = myedx$rating, times = 1, p = 0.7, list = FALSE)
edxValidate <- myedx[-test_index,] #my validation set
edxTrain <- myedx[test_index,] #my training set


# look at my training data

glimpse(myedx)

#the mean rating
mean(edxTrain$rating)
quantile(edxTrain$rating)

edxTrain %>%
  select(rating, title) %>%
  group_by(title) %>%
  summarise(meanrating = mean(rating), n = n()) %>%
  top_n(10, wt = meanrating)


#data exploration
# correlations

cordata <- edxTrain %>%
  select(-c(title, genres, timestamp, INTreview_y, reviewdt, review_y, review_ym, review_ymd)) #remove non numeric for corellation
C <- cor(cordata)
corrplot(C, method = 'number', type = 'lower') #nothing special noted in correlation plot

#release year and span have the strongest correlation with rating (albeit very small). Explore in more detail. 

# visualizations


# ratings by year

edxTrain %>%
  select(review_y) %>%
  group_by(review_y) %>%
  summarise(n = n()) %>%
  ggplot(aes(review_y, n)) +
  geom_col()

edxTrain %>% 
  select(rating) %>%
  group_by(rating) %>%
  summarize(n = n()) %>%
  ggplot(aes(rating, n)) +
  geom_col() +
  geom_smooth()

edxTrain %>% 
  select(rating) %>%
  mutate(rrating = round(rating, digits = 0)) %>%
  group_by(rrating) %>%
  summarize(n = n()) %>%
  ggplot(aes(rrating, n)) +
  geom_point() +
  geom_smooth()

C <- edxTrain %>%
  select(rating) %>%
  group_by(rating) %>%
  summarize(n = n())

cor(C)

edxTrain %>%
  select(rating, review_y) %>%
  group_by(review_y) %>%
  summarise(rat = mean(rating)) %>%
  ggplot(aes(review_y, rat)) +
  geom_point() + 
  geom_smooth()

edxTrain %>%
  select(rating, review_ym) %>%
  group_by(review_ym) %>%
  summarise(rat = mean(rating)) %>%
  ggplot(aes(review_ym, rat)) +
  geom_point() + 
  geom_smooth()

edxTrain %>%
  select(rating, review_ymd) %>%
  group_by(review_ymd) %>%
  summarise(rat = mean(rating), n = n()) %>%
  ggplot(aes(review_ymd, rat)) +
  geom_point() + 
  geom_smooth()

edxTrain %>% 
  select(span_y, rating) %>%
  group_by(span_y) %>%
  summarize(rating = mean(round(rating, digits = 0)), nratings = n()) %>%
  ggplot(aes(x = span_y, y = rating)) +
  geom_point(aes(size = nratings)) +
  geom_smooth()

C <- edxTrain %>%
  select(span_y, rating) 

cor(C)


#Adapted from (Irizarry, 2019)
#Y_{u, i} = \mu + \epsilon_{u, i}

#epsilon represents independent errors sampled from the same distribution centered at zero,
#mu represents the true rating for all movies and users.

#RMSE
mu <- mean(edxTrain$rating)
naive_rmse <- RMSE(edxTrain$rating, mu)

#results table
rmse_results <- data_frame(method = "Just the Average", RMSE = naive_rmse)

#effects
#movie effect
movie_avg <- edxTrain %>%
  group_by(movieId) %>%
  summarize(bmovie = mean(rating - mu))

predicted_ratings <- mu + edxValidate %>%
  left_join(movie_avg, by = "movieId") %>%
  .$bmovie

model_1_rmse <- RMSE(predicted_ratings, edxValidate$rating, na.rm = TRUE)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie effect Model",
                                     RMSE = model_1_rmse ))
rmse_results %>% knitr::kable()


#user effect
user_avg <- edxTrain %>%
  group_by(userId) %>%
  summarize(buser = mean(rating - mu))

predicted_ratings <- edxValidate %>% 
  left_join(movie_avg, by='movieId') %>%
  left_join(user_avg, by='userId') %>%
  mutate(pred = mu + bmovie + buser) %>%
  .$pred

model_2_rmse <- RMSE(predicted_ratings, edxValidate$rating, na.rm = TRUE)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="+ User Effects Model",  
                                     RMSE = model_2_rmse ))
rmse_results %>% knitr::kable()

#release year effect NOT USED BECUASE OF LOW CORELLATION WITH RATING
# release_y_avg <- edxTrain %>%
#   group_by(release_y) %>%
#   summarize(byear = mean(rating - mu))
# 
# predicted_ratings <- edxValidate %>% 
#   left_join(movie_avg, by='movieId') %>%
#   left_join(user_avg, by='userId') %>%
#   left_join(release_y_avg, by = 'release_y') %>%
#   mutate(pred = mu + bmovie + buser + byear) %>%
#   .$pred
# 
# model_2_rmse <- RMSE(predicted_ratings, edxValidate$rating, na.rm = TRUE)
# rmse_results <- bind_rows(rmse_results,
#                           data_frame(method="+ Release Year Effects Model",  
#                                      RMSE = model_2_rmse ))
# rmse_results %>% knitr::kable()

#review year effect NOT USED BECUASE OF LOW CORELLATION WITH RATING
# review_y_avg <- edxTrain %>%
#   group_by(review_y) %>%
#   summarize(byear1 = mean(rating - mu))
# 
# predicted_ratings <- edxValidate %>% 
#   left_join(movie_avg, by='movieId') %>%
#   left_join(user_avg, by='userId') %>%
#   left_join(release_y_avg, by = 'release_y') %>%
#   left_join(review_y_avg, by = 'review_y') %>%
#   mutate(pred = mu + bmovie + buser + byear + byear1) %>%
#   .$pred
# 
# model_2_rmse <- RMSE(predicted_ratings, edxValidate$rating, na.rm = TRUE)
# rmse_results <- bind_rows(rmse_results,
#                           data_frame(method="+ Review Year Effects Model",  
#                                      RMSE = model_2_rmse ))
# rmse_results %>% knitr::kable()

# #genre effect NOT USED BECUSAE IT INCREASES THE RMSE
# genre_avg <- edxTrain %>%
#   group_by(genres) %>%
#   summarize(bgenre = mean(rating - mu))
# 
# predicted_ratings <- edxValidate %>% 
#   left_join(movie_avg, by='movieId') %>%
#   left_join(user_avg, by='userId') %>%
#   #left_join(release_y_avg, by = 'release_y') %>%
#   #left_join(review_y_avg, by = 'review_y') %>%
#   left_join(genre_avg, by = 'genres') %>%
#   mutate(pred = mu + bmovie + bgenre + buser ) %>%
#   .$pred
# 
# model_2_rmse <- RMSE(predicted_ratings, edxValidate$rating, na.rm = TRUE)
# rmse_results <- bind_rows(rmse_results,
#                           data_frame(method="+ Genre Effects Model",  
#                                      RMSE = model_2_rmse ))
# rmse_results %>% knitr::kable()

#regularization
#adapted from 34.9.2 
#find lambda for movie effect

lambdas <- seq(0, 10, 0.25)

mu <- mean(edxTrain$rating)
summ <- edxTrain %>% 
  group_by(movieId) %>% 
  summarize(s = sum(rating - mu), n_i = n())

rmses <- sapply(lambdas, function(l){
  predicted_ratings <- edxValidate %>% 
    left_join(summ, by='movieId') %>% 
    mutate(bmovie = s/(n_i+l)) %>%
    mutate(pred = mu + bmovie) %>%
    pull(pred)
  return(RMSE(predicted_ratings, edxValidate$rating, na.rm = TRUE))
})

qplot(lambdas, rmses)  
lambdas[which.min(rmses)]

#Compute lambda for all effects (b_m, b_u, b_g)
rmses <- sapply(lambdas, function(l){
  b_m <- edxTrain %>%
    group_by(movieId) %>%
    summarise(b_m = sum(rating - mu)/(n()+l))
  b_u <- edxTrain %>%
    left_join(b_m, by="movieId") %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - b_m - mu)/(n()+l))
  # b_g <- edxTrain %>%
  #   left_join(b_m, by="movieId") %>%
  #   left_join(b_u, by="userId") %>%
  #   group_by(genres) %>%
  #   summarise(b_g = sum(rating - b_m - b_u - mu)/(n()+l))
  # # b_y <- edxTrain %>%
  #   left_join(b_m, by="movieId") %>%
  #   left_join(b_u, by="userId") %>%
  #   left_join(b_g, by = "genres") %>%
  #   group_by(review_y) %>%
  #   summarise(b_y = sum(rating - b_m - b_u - b_g - mu)/(n()+l))
  predicted_ratings <- edxValidate %>%
    left_join(b_m, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    # left_join(b_g, by="genres") %>%
    #left_join(b_y, by = "review_y")
    mutate(pred = mu + b_m + b_u) %>%
    pull(pred)
  return(RMSE(predicted_ratings, edxValidate$rating, na.rm = TRUE))
})

lambda <- lambdas[which.min(rmses)]
qplot(lambdas, rmses)
model3 <-  min(rmses)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="+ Lambda Model",  
                                     RMSE = model3 ))
rmse_results %>% knitr::kable()


##training and testing complete

#Pull Final Results Using Edx DAta


lambdas <- seq(0, 10, 0.25)

mu <- mean(edx$rating)
summ <- edx %>% 
  group_by(movieId) %>% 
  summarize(s = sum(rating - mu), n_i = n())

rmses <- sapply(lambdas, function(l){
  predicted_ratings <- validation %>% 
    left_join(summ, by='movieId') %>% 
    mutate(bmovie = s/(n_i+l)) %>%
    mutate(pred = mu + bmovie) %>%
    pull(pred)
  return(RMSE(predicted_ratings, validation$rating, na.rm = TRUE))
})

qplot(lambdas, rmses)  
lambdas[which.min(rmses)]

rmses <- sapply(lambdas, function(l){
  b_m <- edx %>%
    group_by(movieId) %>%
    summarise(b_m = sum(rating - mu)/(n()+l))
  b_u <- edx %>%
    left_join(b_m, by="movieId") %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - b_m - mu)/(n()+l))
  # b_g <- edxTrain %>%
  #   left_join(b_m, by="movieId") %>%
  #   left_join(b_u, by="userId") %>%
  #   group_by(genres) %>%
  #   summarise(b_g = sum(rating - b_m - b_u - mu)/(n()+l))
  # # b_y <- edxTrain %>%
  #   left_join(b_m, by="movieId") %>%
  #   left_join(b_u, by="userId") %>%
  #   left_join(b_g, by = "genres") %>%
  #   group_by(review_y) %>%
  #   summarise(b_y = sum(rating - b_m - b_u - b_g - mu)/(n()+l))
  predicted_ratings <- validation %>%
    left_join(b_m, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    # left_join(b_g, by="genres") %>%
    #left_join(b_y, by = "review_y")
    mutate(pred = mu + b_m + b_u) %>%
    pull(pred)
  return(RMSE(predicted_ratings, validation$rating, na.rm = TRUE))
})

lambda <- lambdas[which.min(rmses)]
qplot(lambdas, rmses)
model3 <-  min(rmses)

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="+ FINAL Lambda Model",  
                                     RMSE = model3 ))
rmse_results %>% knitr::kable()
