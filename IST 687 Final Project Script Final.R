# Course: IST 687
# Name: Jeremiah Graff
# Final Project
# Due Date: 9/16/19
# Date Submitted: 9/15/19

#-----------------------------------------
library(readxl)
library(tidyverse)
library(dplyr)
library(sqldf)
library(ggplot2)
library(data.table)
library(openintro)
library(imputeTS)
library("arules")
library(arulesViz)
library(kernlab)
library(maps)
library(ggmap)

# the RStudio Import Wizard helped generate the code below to import excel file
fpjDF <- read_excel("C:/Users/GraffJe/Desktop/finalprojectIST687.xlsx")

# finding the # of NAs in the data
sum(length(which(is.na(fpjDF))))

# finding the columns with NAs
colnames(fpjDF)[colSums(is.na(fpjDF)) > 0 ]

# https://stackoverflow.com/questions/8161836/how-do-i-replace-na-values-with-zeros-in-an-r-dataframe
# filling NAs with 0
fpjDF[is.na(fpjDF)] <- 0

# checking for anymore NAs
sum(length(which(is.na(fpjDF))))

str(fpjDF)
summary(fpjDF)

#----------------------------
# everything below will help me determine where i want to focus my efforts on this project
#----------------------------

# creating a data frame to show the avg sat score by airline
avgSatByAirline<-data.frame(tapply(fpjDF$Satisfaction,fpjDF$`Airline Name`,mean))

# renaming the column for AvgSatScore by Airline
# help from https://www.datanovia.com/en/lessons/rename-data-frame-columns-in-r/
names(avgSatByAirline)[1]<-"AvgSatScore"

# trying to see total flight counts to find % of flights by airline
avgSatByAirline$flightCount <- length(fpjDF$`Airline Name`)

# finding the count of flights scheduled by airline
avgSatByAirline$flightsScheduled <- tapply(fpjDF$Satisfaction,fpjDF$`Airline Name`,length)

# finding the % of flights schedule by airline and adding 
avgSatByAirline$percentOfFlightsScheduled <- avgSatByAirline$flightsScheduled/avgSatByAirline$flightCount

# finding the # of successful flights flights
avgSatByAirline$SuccessfulFlights <- sqldf('select count("Flight Cancelled") as SuccessulFlights from fpjDF where "Flight Cancelled" = "No" group by "Airline Name"')

# help from https://stackoverflow.com/questions/12384071/how-to-coerce-a-list-object-to-type-double
# to convert int to numeric
avgSatByAirline$SuccessfulFlights <- as.numeric(unlist(avgSatByAirline$SuccessfulFlights))

# calculating % of flights successfully flown
avgSatByAirline$SuccessfulFlightsPercentage <- avgSatByAirline$SuccessfulFlights/avgSatByAirline$flightsScheduled
avgSatByAirline

# showing avg price sensitivity of airline instances
# below shows a range of price sensitivity with range of 1.256 to 1.292 for all 14 airlines...no outliers in terms of avg satisfaction scores
# don't need the code below but it was helpful to see little variation amongst the airlines
# sqldf('select "Airline Name",avg("Price Sensitivity") as AvgPriceSensitivity from fpjDF group by "Airline Name"')

# sorting data frame by % of flights scheduled
# help from https://www.guru99.com/r-sort-data-frame.html
avgSatByAirlineSorted <- avgSatByAirline[order(-avgSatByAirline$percentOfFlightsScheduled),]
avgSatByAirlineSorted

# sorting by successful flight %
avgSatByAirlineSorted2 <- avgSatByAirline[order(-avgSatByAirline$SuccessfulFlightsPercentage),]
avgSatByAirlineSorted2

setDT(avgSatByAirline, keep.rownames = "Airline")[]
str(avgSatByAirline)

# plot 1
ggplot(avgSatByAirline, aes(x=SuccessfulFlightsPercentage, y=percentOfFlightsScheduled, colour=SuccessfulFlightsPercentage
  ))  + geom_point(aes(size=percentOfFlightsScheduled)) + geom_text(aes(label=Airline),hjust=0,vjust=0
  ) + ggtitle("% Successful Flights by % of Flights")

# plot 2
ggplot(avgSatByAirline, aes(x=SuccessfulFlightsPercentage, y=AvgSatScore, colour=SuccessfulFlightsPercentage
))  + geom_point(aes(size=percentOfFlightsScheduled)) + geom_text(aes(label=Airline),hjust=0,vjust=0
) + ggtitle("% Successful Flights & % of Flights by Avg Sat Score")

# after looking through the data shown in code above, i plan to focus on FlyFast Airways due to the fact that they are 
# considerably lower than the next lowest airline in terms of successful flights (95.7% compared to 96.4% - most are in 98% range)
# with the combination of being the 3rd most frequently flown airline...to me, they are a good target to bring about sow ROI on
# in a consultative relationship

#-----------------------------------------
# below will start seeing my narrowed focus on FlyFastAirWays
#-----------------------------------------

# creating a new data frame that only contains FlyFast Airway Inc. Data
FlyFastData <- subset(fpjDF, fpjDF$'Airline Name' == "FlyFast Airways Inc.")
str(FlyFastData)

FFOriginCityCounts<-data.frame(tapply(FlyFastData$Satisfaction,FlyFastData$`Orgin City` ,length))
FFOriginCityCounts$FlightArriving <- tapply(FlyFastData$Satisfaction,FlyFastData$`Destination City` ,length)

# help from https://stackoverflow.com/questions/29511215/convert-row-names-into-first-column
setDT(FFOriginCityCounts, keep.rownames = "City_State")[]
colnames(FFOriginCityCounts) <- c("City_State", "Flights Leaving", "Flights Arriving")
FFOriginCityCounts$cityStateLower <- tolower(FFOriginCityCounts$City_State)
FFOriginCityCounts$State <- FFOriginCityCounts$City_State

# help from https://stackoverflow.com/questions/7963898/extracting-the-last-n-characters-from-a-string-in-r
substrRight <- function(x, n){
  substr(x, nchar(x)-n+1, nchar(x))
}

FFOriginCityCounts$State <- substrRight(FFOriginCityCounts$State,2)
FFOriginCityCounts$StateFull <- abbr2state(FFOriginCityCounts$State)
FFOriginCityCounts$StateFullLower <- tolower(FFOriginCityCounts$StateFull)

FFOriginCityCounts
dim(FFOriginCityCounts)
FFOriginCityCounts$`Flights Leaving` <- as.numeric(FFOriginCityCounts$`Flights Leaving`)
FFOriginCityCounts$`Flights Arriving` <- as.numeric(FFOriginCityCounts$`Flights Arriving`)
str(FFOriginCityCounts)


StateFlightCounts <- data.frame(tapply(FFOriginCityCounts$`Flights Arriving`,FFOriginCityCounts$StateFull,sum))
StateFlightCounts$FlightsLeaving <- tapply(FFOriginCityCounts$`Flights Leaving`,FFOriginCityCounts$StateFull,sum)
colnames(StateFlightCounts) <- c("FlightsArriving","FlightsLeaving")
setDT(StateFlightCounts, keep.rownames = "State")[]
StateFlightCounts$stateLower <- tolower(StateFlightCounts$State)
StateFlightCounts
str(StateFlightCounts)

LeavingFlightSorted <- StateFlightCounts[order(-StateFlightCounts$FlightsLeaving),]
LeavingFlightSorted

ArrivingFlightSorted <- StateFlightCounts[order(-StateFlightCounts$FlightsArriving),]
ArrivingFlightSorted

us <- map_data("state")

# plot 3
# map showing the flight leaving focus
map.NumOfFlightsLeaving <- ggplot(StateFlightCounts,aes(map_id=stateLower))
map.NumOfFlightsLeaving <- map.NumOfFlightsLeaving + geom_map(map =us,aes(fill=StateFlightCounts$FlightsLeaving))
map.NumOfFlightsLeaving <- map.NumOfFlightsLeaving + expand_limits(x=us$long,y=us$lat)
map.NumOfFlightsLeaving <- map.NumOfFlightsLeaving + coord_map() + ggtitle("Leading Flight Departure States")
map.NumOfFlightsLeaving 

# plot 4
# map showing the flight arriving focus
map.NumOfFlightsArriving <- ggplot(StateFlightCounts,aes(map_id=stateLower))
map.NumOfFlightsArriving <- map.NumOfFlightsArriving + geom_map(map =us,aes(fill=StateFlightCounts$FlightsArriving))
map.NumOfFlightsArriving <- map.NumOfFlightsArriving + expand_limits(x=us$long,y=us$lat)
map.NumOfFlightsArriving <- map.NumOfFlightsArriving + coord_map() + ggtitle("Leading Flight Arrival States")
map.NumOfFlightsArriving 

# ----------------------------------------------------------
# bringing in NewLatLon Script elements
library(jsonlite)
library(tidyverse)

nominatim_osm <- function(address = NULL)
{
  if(suppressWarnings(is.null(address)))
    return(data.frame())
  tryCatch(
    d <- jsonlite::fromJSON( 
      gsub('\\@addr\\@', gsub('\\s+', '\\%20', address), 
           'http://nominatim.openstreetmap.org/search/@addr@?format=json&addressdetails=0&limit=1')
    ), error = function(c) return(data.frame())
  )
  if(length(d) == 0) return(data.frame())
  return(data.frame(lon = as.numeric(d$lon), lat = as.numeric(d$lat)))
}



NewLatLon<-function(addresses){
  d <- suppressWarnings(lapply(addresses, function(address) {
    #set the elapsed time counter to 0
    t <- Sys.time()
    #calling the nominatim OSM API
    api_output <- nominatim_osm(address)
    #get the elapsed time
    t <- difftime(Sys.time(), t, 'secs')
    #return data.frame with the input address, output of the nominatim_osm function and elapsed time
    return(data.frame(address = address, api_output, elapsed_time = t))
  }) %>%
    #stack the list output into data.frame
    bind_rows() %>% data.frame())
  #output the data.frame content into console
  return(d)
}

# https://stackoverflow.com/questions/6347356/creating-a-comma-separated-vector

addys <- c("abilene, tx", "akron, oh", "albany, ga", "albany, ny", "albuquerque, nm", 
           "alexandria, la", "allentown, pa", "amarillo, tx", 
           "appleton, wi", "asheville, nc", "atlanta, ga", "augusta, ga", 
           "austin, tx", "baltimore, md", "bangor, me", "baton rouge, la", 
           "beaumont, tx", "billings, mt", "birmingham, al", 
           "bismarck, nd", "bloomington, il", "boston, ma", 
           "bristol, tn", "brownsville, tx", "brunswick, ga", 
           "buffalo, ny", "burlington, vt", "cedar rapids, ia", 
           "charleston, sc", "charleston, wv", "charlotte, nc", "charlottesville, va", 
           "chattanooga, tn", "chicago, il", "cincinnati, oh", "cleveland, oh", 
           "college station, tx", "colorado springs, co", "columbia, sc", 
           "columbus, ga", "columbus, ms", "columbus, oh", "corpus christi, tx", 
           "dallas, tx", "dayton, oh", "denver, co", 
           "des moines, ia", "detroit, mi", "dickinson, nd", "dothan, al", 
           "durango, co", "el paso, tx", "elmira, ny", "evansville, in", 
           "fargo, nd", "fayetteville, ar", "fayetteville, nc", "flint, mi", 
           "fort myers, fl", "fort smith, ar", "fort wayne, in", "gainesville, fl", 
           "grand junction, co", "grand rapids, mi", "green bay, wi", "greensboro, nc", 
           "greer, sc", "gulfport, ms", "gunnison, co", "harlingen, tx", 
           "harrisburg, pa", "hartford, ct", "hobbs, nm", "houston, tx", 
           "huntsville, al", "indianapolis, in", "jackson, ms", 
           "jacksonville, fl", "jacksonville, nc", "kansas city, mo", 
           "key west, fl", "killeen, tx", "knoxville, tn", "lafayette, la", 
           "lake charles, la", "lansing, mi", "laredo, tx", "lexington, ky", 
           "lincoln, ne", "little rock, ar", "louisville, ky", "lubbock, tx", 
           "madison, wi", "manchester, nh", "memphis, tn", "miami, fl", 
           "midland, tx", "milwaukee, wi", "minneapolis, mn", "minot, nd", 
           "mission, tx", "mobile, al", "moline, il", "monroe, la", 
           "montgomery, al", "montrose, co", "mosinee, wi", "myrtle beach, sc", 
           "nashville, tn", "new bern, nc", "new orleans, la", 
           "new york, ny", "newark, nj", "newport news, va", 
           "norfolk, va", "oklahoma city, ok", "omaha, ne", "orlando, fl", 
           "panama city, fl", "pensacola, fl", "peoria, il", "philadelphia, pa", 
           "pittsburgh, pa", "portland, me", "providence, ri", "raleigh, nc", 
           "rapid city, sd", "richmond, va", "roanoke, va", "rochester, ny", 
           "salt lake city, ut", "san angelo, tx", "san antonio, tx", "santa fe, nm", 
           "savannah, ga", "scranton, pa", "shreveport, la", 
           "sioux falls, sd", "south bend, in", "springfield, mo", "st. louis, mo", 
           "state college, pa", "syracuse, ny", "tallahassee, fl", "tampa, fl", 
           "topeka, ks", "traverse city, mi", "tucson, az", "tulsa, ok", 
           "tyler, tx", "valdosta, ga", "valparaiso, fl", "washington, dc", 
           "west palm beach, fl", "white plains, ny", "wichita falls, tx", 
           "wichita, ks", "williston, nd", "wilmington, nc")



map("state")

# based on this map, why don't they fly to the west/CA?
# plot 5
for (i in 1:length(addys)) {
  g.codes <- NewLatLon(addys[i])  
  #print(addys[i])
  print(g.codes)
  points(g.codes$lon, g.codes$lat, col = "red", cex = 1.5, pch = 16)
}

NewLatLon(addys)
latlon<-NewLatLon(addresses)

# --------------------------------------------------------------------
# exploring the data
# plot 6
SatByGender <- data.frame(tapply(FlyFastData$Satisfaction,FlyFastData$Gender,mean))
colnames(SatByGender) <- "AvgSat"
setDT(SatByGender, keep.rownames = "Gender")[]


ggplot(SatByGender, aes(x=Gender,y=AvgSat)) + geom_col(color="white",fill="black"
              ) + scale_y_continuous(limits = c(0,4)) + ggtitle("AvgSatByGender")

# plot 7
SatByAge<-data.frame(tapply(FlyFastData$Satisfaction,FlyFastData$Age ,mean))
colnames(SatByAge) <- "AvgSat"
setDT(SatByAge, keep.rownames = "Age")[]


ggplot(SatByAge, aes(x=Age,y=AvgSat,group=1)) + geom_line(
  )+ theme(axis.text.x = element_text(size = 10)) + ggtitle("AvgSatByAge")

# price sensitivity plot
SatByPriceSens <- data.frame(tapply(FlyFastData$Satisfaction,FlyFastData$`Price Sensitivity` ,mean))
colnames(SatByPriceSens) <- "AvgSat"
setDT(SatByPriceSens, keep.rownames = "PriceSensitivity")[]

ggplot(SatByPriceSens, aes(x=PriceSensitivity,y=AvgSat)) + geom_col(color="white",fill="black"
) + scale_y_continuous(limits = c(0,5)) + ggtitle("AvgSatByPriceSensitivity")


# plot 
SatByStatus <- data.frame(tapply(FlyFastData$Satisfaction,FlyFastData$`Airline Status` ,mean))
colnames(SatByStatus) <- "AvgSat"
setDT(SatByStatus, keep.rownames = "Status")[]

ggplot(SatByStatus, aes(x=Status,y=AvgSat)) + geom_col(color="white",fill="black"
) + scale_y_continuous(limits = c(0,4)) + ggtitle("AvgSatByStatus")

# plot 
SatByTravelType <- data.frame(tapply(FlyFastData$Satisfaction,FlyFastData$`Type of Travel` ,mean))
colnames(SatByTravelType) <- "AvgSat"
setDT(SatByTravelType, keep.rownames = "TravelType")[]

ggplot(SatByTravelType, aes(x=TravelType,y=AvgSat)) + geom_col(color="white",fill="black"
) + scale_y_continuous(limits = c(0,4)) + ggtitle("AvgSatByTravelType")

# the mean is not as drastic of impact on satisfaction as i would've figured
tapply(FlyFastData$Satisfaction,FlyFastData$`Flight cancelled` ,mean)

# the mean is not as drastic of impact on satisfaction as i would've figured
tapply(FlyFastData$Satisfaction,FlyFastData$`Arrival Delay greater 5 Mins` ,mean)

# plot 
SatByDestState <- data.frame(tapply(FlyFastData$Satisfaction,FlyFastData$`Destination State` ,mean))
colnames(SatByDestState) <- "AvgSat"
setDT(SatByDestState, keep.rownames = "DestState")[]

ggplot(SatByDestState, aes(x=DestState,y=AvgSat)) + geom_col(color="white",fill="black"
) + scale_y_continuous(limits = c(0,4)) + ggtitle("AvgSatByDestState") +theme(
  axis.text.x = element_text(color = "grey20", size = 12, angle = 90, hjust = .5, vjust = .5, face = "plain"
))

# plot 
SatByOriginState <- data.frame(tapply(FlyFastData$Satisfaction,FlyFastData$`Origin State` ,mean))
colnames(SatByOriginState) <- "AvgSat"
setDT(SatByOriginState, keep.rownames = "OriginState")[]

ggplot(SatByOriginState, aes(x=OriginState,y=AvgSat)) + geom_col(color="white",fill="black"
) + scale_y_continuous(limits = c(0,4)) + ggtitle("AvgSatByOriginState") +theme(
  axis.text.x = element_text(color = "grey20", size = 12, angle = 90, hjust = .5, vjust = .5, face = "plain"
  ))

# plot: ok plot, nothing special
SatByDay <- data.frame(tapply(FlyFastData$Satisfaction,FlyFastData$`Day of Month` ,mean))
colnames(SatByDay) <- "AvgSat"
setDT(SatByDay, keep.rownames = "Day")[]
SatByDay$Day <- as.numeric(SatByDay$Day)
str(SatByDay)

ggplot(SatByDay, aes(x=Day,y=AvgSat)) + geom_col(color="white",fill="black"
) + scale_y_continuous(limits = c(0,4)) + ggtitle("AvgSatByDay") +theme(
  axis.text.x = element_text(color = "grey20", size = 12, angle = 90, hjust = .5, vjust = .5, face = "plain"
  ))

ggplot(SatByDay, aes(x=Day,y=AvgSat,group=1)) + geom_line(
)+ theme(axis.text.x = element_text(size = 10)) + ggtitle("AvgSatByDay")

# plot: strong trend as x axis increases
SatByNFPA <- data.frame(tapply(FlyFastData$Satisfaction,FlyFastData$`No of Flights p.a.` ,mean))
colnames(SatByNFPA) <- "AvgSat"
setDT(SatByNFPA, keep.rownames = "NFPA")[]
SatByNFPA$NFPA <- as.numeric(SatByNFPA$NFPA)
str(SatByNFPA)

ggplot(SatByNFPA, aes(x=NFPA,y=AvgSat,group=1)) + geom_line()+ theme(axis.text.x = element_text(size = 15),axis.text.y = element_text(size = 15)) + ggtitle("AvgSatByNFPA")

# good plot
SatByTravelDist <- data.frame(tapply(FlyFastData$Satisfaction,FlyFastData$`Flight Distance` ,mean))
colnames(SatByTravelDist) <- "AvgSat"
setDT(SatByTravelDist, keep.rownames = "TravelDist")[]
SatByTravelDist$TravelDist <- as.numeric(SatByTravelDist$TravelDist)
str(SatByNFPA)

ggplot(SatByTravelDist, aes(x=TravelDist,y=AvgSat,group=1)) + geom_line(
)+ theme(axis.text.x = element_text(size = 25)) + ggtitle("AvgSatByTravelDist")

# plot: create line chart
SatByShopAmnt <- data.frame(tapply(FlyFastData$Satisfaction,FlyFastData$`Shopping Amount at Airport` ,mean))
colnames(SatByShopAmnt) <- "AvgSat"
setDT(SatByShopAmnt, keep.rownames = "ShopAmnt")[]

SatByShopAmnt$ShopAmnt <- as.numeric(SatByShopAmnt$ShopAmnt)

ggplot(SatByShopAmnt, aes(x=ShopAmnt,y=AvgSat,group=1)) + geom_line(
)+ theme(axis.text.x = element_text(size = 15),axis.text.y=element_text(size = 15)) + ggtitle("AvgSatByShopAmnt")

# plot: create line chart
SatByEatDrink <- data.frame(tapply(FlyFastData$Satisfaction,FlyFastData$`Eating and Drinking at Airport` ,mean))
colnames(SatByEatDrink) <- "AvgSat"
setDT(SatByEatDrink, keep.rownames = "EatDrink")[]

SatByEatDrink$EatDrink <- as.numeric(SatByEatDrink$EatDrink)

ggplot(SatByEatDrink, aes(x=EatDrink,y=AvgSat,group=1)) + geom_line(
)+ theme(axis.text.x = element_text(size = 15),axis.text.y=element_text(size = 15)) + ggtitle("AvgSatByEatDrink")

# everything below is aimed at creating numeric values to run through linear model
# to find statistical signifcance of the data in reference to customer satisfaction
FlyFastData2 <- FlyFastData

str(FlyFastData2)

# 1 = Female, 0 = male
FlyFastData2$Gender2 <- ifelse(FlyFastData2$Gender == "Female",1,0)  

# learned about factoring and then applied as.numeric to see if below would work and it did!
# blue = 1
# gold = 2
# platinum = 3
# silver = 4
FlyFastData2$AirlineStatus2 <- as.numeric(factor(FlyFastData2$`Airline Status`))

# business travel =1
# mileage tickets =2
# personal travel =3
FlyFastData2$TypeOfTravel2 <- as.numeric(factor(FlyFastData2$`Type of Travel`))

# business =1
# eco =2
# eco plus =3
FlyFastData2$Class2 <- as.numeric(factor(FlyFastData2$Class))

#1 Alabama             
#2 Arizona             
#3 Arkansas            
#4 Colorado            
#5 Connecticut         
                        #6 District of Columbia
#6 Florida             
#7 Georgia             
#8 Illinois            
#9 Indiana         
#10 Iowa            
#11 Kansas          
#12 Kentucky        
#13 Louisiana       
#14 Maine           
#15 Maryland        
#16 Massachusetts   
#17 Michigan        
#18 Minnesota       
#19 Mississippi     
#20 Missouri        
#21 Montana         
#22 Nebraska        
#23 New Hampshire   
#24 New Jersey      
#25 New Mexico      
#26 New York        
#27 North Carolina  
#28 North Dakota    
#29 Ohio            
#30 Oklahoma        
#31 Pennsylvania    
#32 Rhode Island    
#33 South Carolina  
#34 South Dakota 
#35 Tennessee    
#36 Texas        
#37 Utah         
#38 Vermont      
#39 Virginia     
#40 West Virginia
#41 Wisconsin    

#as.factor(FlyFastData2$`Destination State`)
FlyFastData2$DestinationState2 <- as.numeric(factor(FlyFastData2$`Destination State`))

#as.factor(FlyFastData2$`Origin State`)
FlyFastData2$OriginState2 <- as.numeric(factor(FlyFastData2$`Origin State`))

# yes = 1, no = 0
#as.factor(FlyFastData2$`Flight cancelled`)
FlyFastData2$FlightCancelled2 <- ifelse(FlyFastData2$`Flight cancelled` == "Yes",1,0)

# yes = 1, no = 0
#as.factor(FlyFastData2$`Arrival Delay greater 5 Mins`)
FlyFastData2$ArrDelayGrt5Min2 <- ifelse(FlyFastData2$`Arrival Delay greater 5 Mins` == "Yes",1,0)

FlyFastData2 <- FlyFastData2[,-2]
FlyFastData2 <- FlyFastData2[,-3]
FlyFastData2 <- FlyFastData2[,-7]
FlyFastData2 <- FlyFastData2[,-10]
FlyFastData2 <- FlyFastData2[,-12:-15]
FlyFastData2 <- FlyFastData2[,-12:-13]
FlyFastData2 <- FlyFastData2[,-15]
FlyFastData2 <- FlyFastData2[,-17]
# removing date due to lack of significance
FlyFastData2 <- FlyFastData2[,-11]

str(FlyFastData2)

# linear model time!!!
# starting to predict values
# create the training & test data sets
dim(FlyFastData2)
FlyFastData2[1:5,]
randIndex <- sample(1:dim(FlyFastData2)[1])
summary(randIndex)
length(randIndex)
head(randIndex)
cutPoint2_3 <- floor(2 * dim(FlyFastData2)[1]/3)
cutPoint2_3
trainData <- FlyFastData2[randIndex[1:cutPoint2_3],]
dim(trainData)
head(trainData)
testData <- FlyFastData2[randIndex[(cutPoint2_3+1):dim(FlyFastData2)[1]],]
dim(testData)
head(testData)
str(testData)

# lm equation output
# https://www.dataquest.io/blog/statistical-learning-for-predictive-modeling-r/
# parsimonious code to find statistically significant inputs for predicting Satisfaction
# this code will be used throughout linear modeling
parsModel=lm(formula = Satisfaction ~ .,data=FlyFastData2)
step(parsModel, data=FlyFastData2, direction="backward")


# ksvm model
# ksvm values code was from parsimonious model above
ksvmOutput <- ksvm(Satisfaction ~ Age + `Price Sensitivity` + `Year of First Flight` + 
                     `No of Flights p.a.` + `Shopping Amount at Airport` + `Departure Delay in Minutes` + 
                     `Arrival Delay in Minutes` + `Flight time in minutes` + `Flight Distance` + 
                     Gender2 + AirlineStatus2 + TypeOfTravel2 + Class2 + FlightCancelled2, data = trainData)

ksvmOutput

# creating ksvm predictions from model above
ksvmPred <- predict(ksvmOutput, testData, type="votes")

# comparison dataframe that shows actual vs predicted
KSVMcompTable <- data.frame(testData[,1],ksvmPred[,1])
colnames(KSVMcompTable) <- c("true","pred")
head(KSVMcompTable)
KSVMcompTable

# compute root mean squared error
ksvmRMSE <- sqrt(mean((KSVMcompTable$true - KSVMcompTable$pred)^2))
ksvmRMSE

# compute absolute error for each case
KSVMcompTable$error <- abs(KSVMcompTable$true - KSVMcompTable$pred)
head(KSVMcompTable)

# create new dataframe for plot
ksvmPlot <- data.frame(KSVMcompTable$error,testData$Satisfaction, testData$`No of Flights p.a.`, testData$Age , testData$AirlineStatus2, testData$TypeOfTravel2)
colnames(ksvmPlot) <- c("error","Satisfaction","NoOfFlights", "Age","Status","TravelType")

# plot for ksvm model with errors shown
# color help from http://www.sthda.com/english/wiki/ggplot2-colors-how-to-change-colors-automatically-and-manually
plt1 <- ggplot(ksvmPlot,aes(x=Age,y=NoOfFlights)) + geom_point(aes(size=Satisfaction,color=error)) + ggtitle("ksvm") + scale_color_gradientn(colours = rainbow(5))
plt1

#svm output
# https://cran.r-project.org/web/packages/e1071/vignettes/svmdoc.pdf
# creating svm prediction model
svmModel <- rpart(Satisfaction ~ Age + `Price Sensitivity` + `Year of First Flight` + 
                    `No of Flights p.a.` + `Shopping Amount at Airport` + `Departure Delay in Minutes` + 
                    `Arrival Delay in Minutes` + `Flight time in minutes` + `Flight Distance` + 
                    Gender2 + AirlineStatus2 + TypeOfTravel2 + Class2 + FlightCancelled2, data = trainData)

# actual predictions
svmPred <- predict(svmModel,testData)

# comparison dataframe that shows actual vs predicted
SVMcompTable <- data.frame(pred=svmPred,true=testData$Satisfaction)
head(SVMcompTable)

# compute root mean squared error for svm model to see the spread of residuals
svmRMSE <- sqrt(mean((SVMcompTable$true - SVMcompTable$pred)^2))
svmRMSE

# compute absolute error for each case
SVMcompTable$error <- abs(SVMcompTable$true - SVMcompTable$pred)
head(SVMcompTable)

# create new dataframe to build plot
svmPlot <- data.frame(SVMcompTable$error,testData$Satisfaction, testData$`No of Flights p.a.`, testData$Age , testData$AirlineStatus2, testData$TypeOfTravel2)
colnames(svmPlot) <- c("error","Satisfaction","NoOfFlights", "Age","Status","TravelType")

# plot for svm predictions along with error value
plt2 <- ggplot(svmPlot,aes(x=Age,y=NoOfFlights)) + geom_point(aes(size=Satisfaction,color=error)) + ggtitle("svm") + scale_color_gradientn(colours = rainbow(5))
plt2

# lm output
lmOutput <- lm(formula = Satisfaction ~ Age + `Price Sensitivity` + `Year of First Flight` + 
                 `No of Flights p.a.` + `Shopping Amount at Airport` + `Departure Delay in Minutes` + 
                 `Arrival Delay in Minutes` + `Flight time in minutes` + `Flight Distance` + 
                 Gender2 + AirlineStatus2 + TypeOfTravel2 + Class2 + FlightCancelled2, data = testData)
summary(lmOutput)

# creating the predictions based on model above
predValue <- data.frame(Satisfaction = testData$Satisfaction, Age = testData$Age, NoOfFlights = testData$`No of Flights p.a.`)
predValue$Pred <- predict(lmOutput,pred=predValue)
predValue$error <- abs(predValue$Satisfaction - predValue$Pred)

# compute root mean squared error for lm
lmRMSE <- sqrt(mean((predValue$Satisfaction - predValue$Pred)^2))
lmRMSE

# plot for lm predictions of actual values along with the error
plt3<-ggplot(predValue,aes(x=Age,y=NoOfFlights)) + geom_point(aes(size=Satisfaction,color=error)) + ggtitle("lm") + scale_color_gradientn(colours = rainbow(5))
plt3

# extraGrid for actual value predictions
#https://stackoverflow.com/questions/35634736/consistent-plotting-panel-width-height-when-using-gridextra
#https://cran.r-project.org/web/packages/egg/vignettes/Ecosystem.html
grid.arrange(plt1,plt2,plt3,nrow=2)

# joining the RMSE's into df for step 6
RMSEdf <- data.frame(ksvmRMSE,svmRMSE,lmRMSE)
RMSEdf
