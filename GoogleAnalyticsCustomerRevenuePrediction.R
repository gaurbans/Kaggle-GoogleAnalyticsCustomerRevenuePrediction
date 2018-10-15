# Load packages
library(lubridate)
library(jsonlite)
library(rlist)
library(data.table)
library(caret)

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

# Look at columns in trn that are not in tst
names(trn)[which(!names(trn) %in% names(tst))]
##totals.transactionRevenue is the target variable. trafficSource.campaignCode would be a predictor yet it isn't in tst. 
##Remove trafficSource.campaignCode if it has few unique values
table(trn$trafficSource.campaignCode)
##Remove trafficSource.campaignCode because it has only one non-missing value
trn <- trn[-which(names(trn) %in% 'trafficSource.campaignCode')]

# Get rid of columns that have only one unique value since they don't have any predictive power
todrop <- character()
for(i in 1:ncol(trn)){
    if(length(table(trn[,i]))==1){
        todrop <- c(todrop, names(trn[i]))
    }
}
trn <- trn[-which(names(trn) %in% todrop)]
tst <- tst[-which(names(tst) %in% todrop)]

# Convert totals.hits and totals.pageviews to integer
trn$totals.hits <- as.integer(trn$totals.hits)
trn$totals.pageviews <- as.integer(trn$totals.pageviews)
trn$totals.transactionRevenue <- as.integer(trn$totals.transactionRevenue)

tst$totals.hits <- as.integer(tst$totals.hits)
tst$totals.pageviews <- as.integer(tst$totals.pageviews)

# Get number of visits for each visitor
getNumVisits <- function(df){
    numVisits <- as.data.frame(table(df$fullVisitorId))
    names(numVisits) <- c('fullVisitorId', 'num.visits')
    numVisits$fullVisitorId <- as.character(numVisits$fullVisitorId)
    
    df <- merge(df, numVisits, all.x=TRUE)
    
    return(df)
}

trn <- getNumVisits(trn)
tst <- getNumVisits(tst)

# Remove observations in trn where target variable is missing
trn <- trn[!is.na(trn$totals.transactionRevenue),]









