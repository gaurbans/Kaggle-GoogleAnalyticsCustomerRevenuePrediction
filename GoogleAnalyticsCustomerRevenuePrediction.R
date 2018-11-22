# Load packages
library(lubridate)
library(jsonlite)
library(tidyverse)
library(data.table)
library(caret)
library(chron)
library(dplyr)
library(lightgbm)

# Read data
trn <- read_csv("../input/train.csv")
tst <- read_csv("../input/test.csv")

# Convert JSON columns to regular columns 
##Thanks to Erik Bruin for this!
tr_device <- paste("[", paste(trn$device, collapse = ","), "]") %>% fromJSON(flatten = T)
tr_geoNetwork <- paste("[", paste(trn$geoNetwork, collapse = ","), "]") %>% fromJSON(flatten = T)
tr_totals <- paste("[", paste(trn$totals, collapse = ","), "]") %>% fromJSON(flatten = T)
tr_trafficSource <- paste("[", paste(trn$trafficSource, collapse = ","), "]") %>% fromJSON(flatten = T)

te_device <- paste("[", paste(tst$device, collapse = ","), "]") %>% fromJSON(flatten = T)
te_geoNetwork <- paste("[", paste(tst$geoNetwork, collapse = ","), "]") %>% fromJSON(flatten = T)
te_totals <- paste("[", paste(tst$totals, collapse = ","), "]") %>% fromJSON(flatten = T)
te_trafficSource <- paste("[", paste(tst$trafficSource, collapse = ","), "]") %>% fromJSON(flatten = T)

#Combine to make the full training and test sets
trn <- trn %>%
    cbind(tr_device, tr_geoNetwork, tr_totals, tr_trafficSource) %>%
    select(-device, -geoNetwork, -totals, -trafficSource)
    
tst <- tst %>%
    cbind(te_device, te_geoNetwork, te_totals, te_trafficSource) %>%
    select(-device, -geoNetwork, -totals, -trafficSource)

#Remove temporary tr_ and te_ sets
rm(tr_device, tr_geoNetwork, tr_totals, tr_trafficSource, te_device, te_geoNetwork, te_totals, te_trafficSource)

# Remove variables in trn that aren't in tst
names(trn)[which(!names(trn) %in% names(tst))]
##transactionRevenue is the target variable. campaignCode would be a predictor yet it isn't in tst. 
trn <- trn[-which(names(trn) %in% 'campaignCode')]

# See which columns have only one unique value and remove them
nUnique <- sapply(trn, function(x) length(unique(x)))
print(nUnique)

todrop <- names(which(nUnique==1))
trn <- trn[-which(names(trn) %in% todrop)]
tst <- tst[-which(names(tst) %in% todrop)]
                  
# Convert certain variables to numeric
numCols <- c('hits', 'pageviews', 'bounces', 'newVisits')
trn[numCols] <- sapply(trn[numCols], as.integer)
tst[numCols] <- sapply(tst[numCols], as.integer)

# convert date to Date
trn$date <- ymd(trn$date)
tst$date <- ymd(tst$date)

# Convert visitStartTime to POSIXct and extract hour
trn$visitStartTime <- as.POSIXct(trn$visitStartTime, tz="UTC", origin='1970-01-01')
tst$visitStartTime <- as.POSIXct(tst$visitStartTime, tz="UTC", origin='1970-01-01')
                  
# Convert target variable to numeric and set missing values to zero
trn$transactionRevenue <- as.numeric(trn$transactionRevenue)
trn$transactionRevenue[is.na(trn$transactionRevenue)] <- 0

# Take log(transactionRevenue + 1)
trn$transactionRevenue <- log1p(trn$transactionRevenue)
                  
# Checkout missing values in trn
sapply(trn, function(x) sum(is.na(x))/nrow(trn))
sapply(tst, function(x) sum(is.na(x))/nrow(tst))
       
#Separate out character and numeric data
trnnum <- trn[sapply(trn, function(x) !is.character(x))]
tstnum <- tst[sapply(tst, function(x) !is.character(x))]
       
trncat <- trn[sapply(trn, is.character)]
tstcat <- tst[sapply(tst, is.character)]
       
# Set missing values in numeric variables to 0
trnnum[is.na(trnnum)] <- 0
tstnum[is.na(tstnum)] <- 0

# Many non-missing entries are actually missing but coded as "(none)", "(not set)", or something similar
# Replace actually missing entries with NA
actuallyMissing <- c("not available in demo dataset", "(not provided)", "(not set)", "<NA>", "unknown.unknown",  "(none)")
for(i in 1:ncol(trncat)){
    trncat[which(trncat[,i] %in% actuallyMissing), i] <- NA
}

for(i in 1:ncol(tstcat)){
    tstcat[which(trncat[,i] %in% actuallyMissing), i] <- NA
}

# Combine num and cat dataframes
trn <- cbind(trnnum, trncat)
tst <- cbind(tstnum, tstcat)
                     
# Extract week, day, hour, and holiday from date data
trn$week <- factor(week(trn$visitStartTime))
trn$day <- factor(weekdays(trn$date))
trn$hour <- hour(trn$visitStartTime)
trn$holiday <- ifelse(chron::is.holiday(trn$date), 1L, 0L)

tst$week <- factor(week(tst$visitStartTime))
tst$day <- factor(weekdays(tst$visitStartTime))
tst$hour <- hour(tst$visitStartTime)
tst$holiday <- ifelse(chron::is.holiday(tst$date), 1L, 0L)
                     
# See number of unique values for each column
sapply(trn, function(x) length(unique(x)))
       
# Convert binary variables to 0/1
trn$isMobile <- ifelse(trn$isMobile, 1L, 0L)
trn$isTrueDirect <- ifelse(is.na(trn$isTrueDirect), 1L, 0L)
trn$adwordsClickInfo.isVideoAd <- ifelse(is.na(trn$adwordsClickInfo.isVideoAd), 0L, 1L)

tst$isMobile <- ifelse(tst$isMobile, 1L, 0L)
tst$isTrueDirect <- ifelse(is.na(tst$isTrueDirect), 1L, 0L)
tst$adwordsClickInfo.isVideoAd <- ifelse(is.na(tst$adwordsClickInfo.isVideoAd), 0L, 1L)
       
# Take log of totals.pageviews
trn$pageviews <- log1p(trn$pageviews)
tst$pageviews <- log1p(tst$pageviews)
       
# Convert character variables to integer, except sessionId and fullVisitorId
notconvert <- c('sessionId', 'fullVisitorId')

trn[!(names(trn) %in% notconvert)] <- trn[!(names(trn) %in% notconvert)] %>% mutate_if(is.character, as.factor)
tst[!(names(tst) %in% notconvert)] <- tst[!(names(tst) %in% notconvert)] %>% mutate_if(is.character, as.factor)

# Retain names of categorical variables and then convert them to integer
catVars <- names(Filter(is.factor, trn[!(names(trn) %in% notconvert)]))

trn <- trn %>% mutate_if(is.factor, as.integer)
tst <- tst %>% mutate_if(is.factor, as.integer)
       
# Create validation dataset
val <- trn[trn$date>='2017-06-01',]
trn <- trn[trn$date<'2017-06-01',]
       
# Exlude variables not used for modeling
ytrn <- trn$transactionRevenue
yval <- val$transactionRevenue

notmodel <- c('date', 'visitId', 'visitStartTime', 'fullVisitorId', 'sessionId', 'transactionRevenue')
tr <- trn[!(names(trn) %in% notmodel)]
va <- val[!(names(val) %in% notmodel)]
       
# Build LightGBM model
set.seed(123)
lgb.train = lgb.Dataset(data=as.matrix(tr), label=ytrn, categorical_feature=catVars)
lgb.valid = lgb.Dataset(data=as.matrix(va), label=yval, categorical_feature=catVars)

params <- list(objective="regression", metric="rmse", learning_rate=0.01)

lgb.model <- lgb.train(params=params, data=lgb.train, valids=list(val=lgb.valid), learning_rate=0.01, nrounds=1000, verbose=1, early_stopping_rounds=50, eval_freq=100)

lgb.model$best_iter
lgb.model$best_score
       
# Predict on test data
te <- tst[!(names(tst) %in% notmodel)]
te$pred <- predict(lgb.model, as.matrix(te))
te$pred <- expm1(te$pred)
te$pred <- ifelse(te$pred<0, 0, te$pred)
te$pred <- ifelse((te$pageviews<7 | te$bounces==1), 0, te$pred)

# Get sum of revenue by fullVisitorId
pred <- data.frame(fullVisitorId = tst$fullVisitorId, PredictedLogRevenue=te$pred)
pred <- aggregate(PredictedLogRevenue ~ fullVisitorId, data=pred, FUN=function(x) log1p(sum(x)))
pred$PredictedLogRevenue <- round(pred$PredictedLogRevenue, 5)
                  
# Prepare submission
submission <- read.csv("../input/sample_submission.csv", colClasses=c('character', 'numeric'))
submission <- merge(submission['fullVisitorId'], pred, all.x=T)
write.csv(submission, 'lgbsubmission.csv', row.names=F)
