# Load packages
library(data.table)
library(caret)
library(chron)
library(dplyr)
library(lightgbm)

# Function to read data
load_data <- function(filename, json_cols, ...) {
  
  # load json
  json_data <- fread(filename, select=json_cols, ...)
  
  # remove double quotes
  json_data[, (json_cols) := lapply(.SD, function(x) gsub('\"\"', '\"', x))]
  
  # Thank you @ML: https://www.kaggle.com/mrlong/r-flatten-json-columns
  jsontodf <- function(col){
    data.table(rlist::list.stack(lapply(col, function(j)
      as.list(unlist(jsonlite::fromJSON(j)))), fill=T))
  }                                      
  json_data <- do.call(cbind, lapply(json_data, jsontodf))
  
  # load remaing data
  other_data <- fread(filename, drop=json_cols,  ...)
  other_data[, date := lubridate::ymd(date)]
  cbind(other_data, json_data)
}

# Read in training and test data
##Data available at https://www.kaggle.com/c/ga-customer-revenue-prediction/data
trn <- as.data.frame(load_data('../input/train.csv', json_cols <- c('device', 'geoNetwork', 'totals', 'trafficSource')))
tst <- as.data.frame(load_data('../input/test.csv', json_cols <- c('device', 'geoNetwork', 'totals', 'trafficSource')))

# Remove variables in trn that aren't in tst
names(trn)[which(!names(trn) %in% names(tst))]
##totals.transactionRevenue is the target variable. trafficSource.campaignCode would be a predictor yet it isn't in tst. 
trn <- trn[-which(names(trn) %in% 'trafficSource.campaignCode')]

# See which columns have only one unique value and remove them
nUnique <- sapply(trn, function(x) length(unique(x)))
print(nUnique)

todrop <- names(which(nUnique==1))
trn <- trn[-which(names(trn) %in% todrop)]
tst <- tst[-which(names(tst) %in% todrop)]

# Convert totals.hits and totals.pageviews# Convert certain variables to numeric
numCols <- c('totals.hits', 'totals.pageviews', 'totals.bounces', 'totals.newVisits')
trn[numCols] <- sapply(trn[numCols], as.numeric)
tst[numCols] <- sapply(tst[numCols], as.numeric)

# Convert visitStartTime to POSIXct and extract hour
trn$visitStartTime <- as.POSIXct(trn$visitStartTime, tz="UTC", origin='1970-01-01')
tst$visitStartTime <- as.POSIXct(tst$visitStartTime, tz="UTC", origin='1970-01-01')

# Convert target variable to numeric and set missing values to zero
trn$totals.transactionRevenue <- as.numeric(trn$totals.transactionRevenue)
trn$totals.transactionRevenue[is.na(trn$totals.transactionRevenue)] <- 0

# Take log(totals.transactionRevenue + 1)
trn$totals.transactionRevenue <- log1p(trn$totals.transactionRevenue)

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
       
# Fill missing categorical variables with 'missing'
trncat[is.na(trncat)] <- 'missing'
tstcat[is.na(tstcat)] <- 'missing'

# Many non-missing entries are actually missing but coded as "(none)", "(not set)", or something similar
# Replace actually missing entries with 'missing'
actuallyMissing <- c("not available in demo dataset", "(not provided)", "(not set)", "<NA>", "unknown.unknown",  "(none)")
for(i in 1:ncol(trncat)){
    trncat[which(trncat[,i] %in% actuallyMissing), i] <- 'missing'
}

for(i in 1:ncol(tstcat)){
    tstcat[which(trncat[,i] %in% actuallyMissing), i] <- 'missing'
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
trn$device.isMobile <- ifelse(trn$device.isMobile, 1L, 0L)
trn$trafficSource.isTrueDirect <- ifelse(trn$trafficSource.isTrueDirect=='missing', 1L, 0L)
trn$trafficSource.adwordsClickInfo.isVideoAd <- ifelse(trn$trafficSource.adwordsClickInfo.isVideoAd=='missing', 0L, 1L)

tst$device.isMobile <- ifelse(tst$device.isMobile, 1L, 0L)
tst$trafficSource.isTrueDirect <- ifelse(tst$trafficSource.isTrueDirect=='missing', 1L, 0L)
tst$trafficSource.adwordsClickInfo.isVideoAd <- ifelse(tst$trafficSource.adwordsClickInfo.isVideoAd=='missing', 0L, 1L)

# Take log of totals.pageviews
trn$totals.pageviews <- log1p(trn$totals.pageviews)
tst$totals.pageviews <- log1p(tst$totals.pageviews)
       
# Drop unnecessary variables
##Information in geoNetwork.continent is already in geoNetwork.subContinent
##trafficSource.adwordsClickInfo.gclId seems useless
todrop <- c('geoNetwork.continent', 'trafficSource.adwordsClickInfo.gclId')

trn <- trn[-which(names(trn) %in% todrop)]
tst <- tst[-which(names(tst) %in% todrop)]
       
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
ytrn <- trn$totals.transactionRevenue
yval <- val$totals.transactionRevenue

notmodel <- c('visitId', 'visitStartTime', 'fullVisitorId', 'sessionId', 'totals.transactionRevenue')
tr <- trn[!(names(trn) %in% notmodel)]
va <- val[!(names(val) %in% notmodel)]

# Build LightGBM model
set.seed(123)
lgb.train = lgb.Dataset(data=as.matrix(tr),label=ytrn, categorical_feature=catVars)
lgb.valid = lgb.Dataset(data=as.matrix(va),label=yval, categorical_feature=catVars)

params <- list(objective="regression", metric="rmse", learning_rate=0.01)

lgb.model <- lgb.train(params = params, data = lgb.train, valids = list(val = lgb.valid), learning_rate=0.01, nrounds=1000, verbose=1,
    early_stopping_rounds=50, eval_freq=100)

lgb.model$best_iter
lgb.model$best_score
       
# Predict on test data
te <- tst[!(names(tst) %in% notmodel)]
pred <- predict(lgb.model, as.matrix(te))
pred <- expm1(pred)
pred <- ifelse(pred<0, 0, pred)

# Get sum of revenue by fullVisitorId
pred <- data.frame(fullVisitorId=tst$fullVisitorId, PredictedLogRevenue=pred)
pred <- aggregate(PredictedLogRevenue ~ fullVisitorId, data=pred, FUN=function(x) log1p(sum(x)))
                  
# Prepare submission
submission <- as.data.frame(fread("../input/sample_submission.csv", colClasses=c('character', 'numeric')))                  
submission <- merge(submission['fullVisitorId'], pred, all.x=T)
write.csv(submission, 'lgbsubmission.csv', row.names=F)



