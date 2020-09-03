# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 20:17:01 2020

@author: GRAFFJE
"""

# import packages for analysis and modeling
import pandas as pd  # data frame operations

import numpy as np  # arrays and math functions
from scipy.stats import uniform  # for training-and-test split
#import statsmodels.api as sm  # statistical models (including regression)
import statsmodels.formula.api as smf  # R-like model specification
import matplotlib.pyplot as plt  # 2D plotting
import seaborn as sns  #  PLOTTING
from scipy import stats


# read in data and create data frame (first one is the online file, second is csv file that i added additional info to from online sources)
# coaches3 = pd.read_csv("https://raw.githubusercontent.com/2SUBDA/IST_718/master/Coaches9.csv")
coaches = pd.read_csv("H:/Jeremiah Graff/0-Jeremiah Master's Degree/6-July 2020 Classes/IST 718/Labs/Lab1/coaches.csv")
# the file below changes AAC to Big East to answer one of the questions in assignment
# coaches2 = pd.read_csv("H:/Jeremiah Graff/0-Jeremiah Master's Degree/6-July 2020 Classes/IST 718/Labs/Lab1/coachesv2.csv")

# i hard coded the info below after loading the text file from online into a csv file.
# stadium info from: https://en.wikipedia.org/wiki/List_of_American_football_stadiums_by_capacity & https://en.wikipedia.org/wiki/List_of_NCAA_Division_I_FBS_football_stadiums
# strength of schedule from: http://powerrankingsguru.com/college-football/strength-of-schedule.php
# record data from: https://www.espn.com/college-football/standings
# academic data from: https://www.icpsr.umich.edu/icpsrweb/NCAA/studies/30022/datadocumentation

# just to see if it works
print(coaches)

# just so i don't include it in my model and be one of those folks :)
coaches = coaches.drop('SchoolPay',1)

# all cells below are not needed
coaches = coaches.drop('AssistantPay',1)
coaches = coaches.drop('Bonus',1)
coaches = coaches.drop('BonusPaid',1)
coaches = coaches.drop('Buyout',1)

print(coaches.info())
# https://stackoverflow.com/questions/38516481/trying-to-remove-commas-and-dollars-signs-with-pandas-in-python
coaches['TotalPay'] = coaches['TotalPay'].replace({'\$': '', ',': '', ' ':'','--':''}, regex=True)
print(coaches)

# changing the columns below to a numeric object
coaches["TotalPay"] = pd.to_numeric(coaches['TotalPay'], errors = 'coerce')

print(coaches.info())



# https://stackoverflow.com/questions/29314033/drop-rows-containing-empty-cells-from-a-pandas-dataframe
# dropping rows that have an empty value for TotalPay, includes: Baylor, BYU, Rice, SMU
coaches['TotalPay'].replace('', np.nan, inplace=True)
coaches.dropna(subset=['TotalPay'], inplace=True)

# checking to make sure that the 4 rows dropped as hoped...should end up with 125 entries
print(coaches.info())

coaches['Conference2'] = coaches['Conference'].replace({'AAC': '2AAC', 'ACC': '3ACC', 'Big 12':'4Big 12','Big Ten':'5Big Ten'
                                                       ,'C-USA':'6C-USA','Ind.':'10Ind.','MAC':'7MAC','Mt. West':'8Mt. West'
                                                       ,'Pac-12':'9Pac-12','SEC':'1SEC','Sun Belt':'11Sun Belt'}, regex=True)

sns.boxplot(y="TotalPay", data=coaches)
sns.boxplot(x= "Conference", y = "TotalPay", data = coaches).set_title('Conference TotalPay Boxplot')
# going horizontal with the boxplot to make conferences more readable: https://python-graph-gallery.com/31-horizontal-boxplot-with-seaborn/
sns.boxplot(y= "Conference", x = "TotalPay", data = coaches).set_title('Conference TotalPay Boxplot')

# https://seaborn.pydata.org/generated/seaborn.violinplot.html
sns.violinplot(x= "Conference", y = "TotalPay", data = coaches).set_title('Conference TotalPay Violinplot')
# going horizontal with the boxplot to make conferences more readable
sns.violinplot(y= "Conference", x = "TotalPay", data = coaches).set_title('Conference TotalPay Violinplot')

# https://stackoverflow.com/questions/38309729/count-unique-values-with-pandas-per-groups
ConfCount = coaches.groupby('Conference')['Coach'].nunique()
#print(ConfCount)
# below should equal 125 to match the number of coaches
#print(ConfCount.sum())
ConfSum = coaches.groupby('Conference')['TotalPay'].sum()
#print(ConfSum)
# below should equal 302132595 to match the number of coaches
#print(ConfSum.sum())
ConfAvgPay = ConfSum/ConfCount
# https://stackoverflow.com/questions/3387655/safest-way-to-convert-float-to-integer-in-python
ConfAvgPay = ConfAvgPay.astype(int)
#print(ConfAvgPay)

# https://stackoverflow.com/questions/28503445/assigning-column-names-to-a-pandas-series
ConfAvgPayDF = pd.DataFrame({"Conference":ConfAvgPay.index, "AvgPay":ConfAvgPay.values})
#print(ConfAvgPayDF)

# https://stackoverflow.com/questions/24988873/python-sort-descending-dataframe-with-pandas
SortConfAvgPayDF = ConfAvgPayDF.sort_values('AvgPay', ascending=False)
#print(SortConfAvgPayDF)
sns.barplot(x="AvgPay", y="Conference", color = 'Blue', data=SortConfAvgPayDF).set_title('Avg Conference TotalPay')

# conference attendance capabilities
ConfAttSum = coaches.groupby('Conference')['StadiumSize'].sum()
#print(ConfAttSum)
# https://stackoverflow.com/questions/3387655/safest-way-to-convert-float-to-integer-in-python
ConfAvgAtt = ConfAttSum/ConfCount.astype(int)
#print(ConfAvgAtt)

ConfAvgAttDF = pd.DataFrame({"Conference":ConfAvgAtt.index, "StadiumSize":ConfAvgAtt.values})
#print(ConfAvgAttDF)

SortConfAvgAttDF = ConfAvgAttDF.sort_values('StadiumSize', ascending=False)
#print(SortConfAvgAttDF)
sns.barplot(x="StadiumSize", y="Conference", color = 'Blue', data=SortConfAvgAttDF).set_title('Avg Conference Stadium Size')


# https://stackoverflow.com/questions/43859416/finding-top-10-in-a-dataframe-in-pandas
# https://seaborn.pydata.org/tutorial/categorical.html?highlight=color%20bar
Top10Salaries = coaches.sort_values("TotalPay", ascending = False).head(10)
#print(Top10Salaries)

# run both lines together
sns.barplot(x="TotalPay", y="Coach", hue = 'Conference', data=Top10Salaries).set_title('Top 10 Coach Salaries')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

sns.catplot(x="TotalPay", y="Coach", hue = 'Conference', data=Top10Salaries)

ConfTop10Count = Top10Salaries.groupby('Conference')['TotalPay'].nunique()
#print(ConfTop10Count)
ConfTop10CountDF = pd.DataFrame({"Conference":ConfTop10Count.index, "Count":ConfTop10Count.values})
SortConfTop10Count = ConfTop10CountDF.sort_values('Count', ascending=False)

sns.barplot(x="Count", y="Conference", color = 'Blue', data=SortConfTop10Count).set_title('# of Coaches in top 10 highest paid')

# https://seaborn.pydata.org/generated/seaborn.scatterplot.html & https://stackoverflow.com/questions/30490740/move-legend-outside-figure-in-seaborn-tsplot  
# run both lines together
sns.scatterplot(x="WinPct", y="TotalPay", hue="Conference", data=coaches).set_title('TotalPay Compared to WinPct')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# correlation: https://stackoverflow.com/questions/29432629/plot-correlation-matrix-using-pandas
# run all 4 lines together
f, ax = plt.subplots(figsize=(10, 8))
corr = coaches.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)



# https://seaborn.pydata.org/generated/seaborn.regplot.html
sns.regplot(x="WinPct", y="TotalPay", data=coaches).set_title('TotalPay & WinPct Regression Line Plot')

# https://stackoverflow.com/questions/25579227/seaborn-implot-with-equation-and-r2-text
# https://stackoverflow.com/questions/60358228/how-to-set-title-on-seaborn-jointplot
def r2(x,y):
    return stats.pearsonr(x,y)[0] **2

# run all 3 lines below together
pWinPct = sns.jointplot(x="WinPct", y="TotalPay", data=coaches, kind="reg",stat_func = r2)
pWinPct.fig.suptitle("TotalPay Relationship with WinPct")
pWinPct.fig.subplots_adjust(top=0.95)
# run all 3 lines below together
pGuruRank = sns.jointplot(x="guruRank", y="TotalPay", data=coaches, kind="reg",stat_func = r2)
pGuruRank.fig.suptitle("TotalPay Relationship with guruRank")
pGuruRank.fig.subplots_adjust(top=0.95)
# run all 3 lines below together
pStadiumSize = sns.jointplot(x="StadiumSize", y="TotalPay", data=coaches, kind="reg",stat_func = r2)
pStadiumSize.fig.suptitle("TotalPay Relationship with StadiumSize")
pStadiumSize.fig.subplots_adjust(top=0.95)
# run all 3 lines below together
pOppoRank = sns.jointplot(x="OppoRank", y="TotalPay", data=coaches, kind="reg",stat_func = r2)
pOppoRank.fig.suptitle("TotalPay Relationship with OppoRank")
pOppoRank.fig.subplots_adjust(top=0.95)
# run all 3 lines below together
pStudAth = sns.jointplot(x="StudAth", y="TotalPay", data=coaches, kind="reg",stat_func = r2)
pStudAth.fig.suptitle("TotalPay Relationship with StudAth")
pStudAth.fig.subplots_adjust(top=0.95)
# run all 3 lines below together
pGradSuccRate = sns.jointplot(x="GradSuccRate", y="TotalPay", data=coaches, kind="reg",stat_func = r2)
pGradSuccRate.fig.suptitle("TotalPay Relationship with GradSuccRate")
pGradSuccRate.fig.subplots_adjust(top=0.95)

# StadiumSize r^2 = .64
# OppoRank r^2 = .58
# guruRank r^2 = .51
# GradSuccRate r^2 = .13
# WinPct r^2 = .12
# StudAth r^2 = .027

# totalpay also pops with stadium size in a particularly strong way, strongest r^2 value
# guruRank has a strong negative correlation, this is intuitive because the lower you are, the better your team is
# lastly, TotalPay has a strong negative correlation when measured against OppoRank, which makes sense because the better teams will also be rated lower (meaning more difficult)...the better the team you are, the tougher your schedule will be


#################################################
############ starting to build model ############
#################################################
# employ training-and-test regimen for model validation

np.random.seed(1234)
coaches['runiform'] = uniform.rvs(loc = 0, scale = 1, size = len(coaches))
coaches_train = coaches[coaches['runiform'] >= 0.33]
coaches_test = coaches[coaches['runiform'] < 0.33]
# check training data frame
print('\ncoaches_train data frame (rows, columns): ',coaches_train.shape)
print(coaches_train.head())
# check test data frame
print('\ncoaches_test data frame (rows, columns): ',coaches_test.shape)
print(coaches_test.head())

# specify a simple model with bobblehead entered last
my_model_1 = str('TotalPay ~ StadiumSize + guruRank + OppoRank + WinPct + StudAth + GradSuccRate + Conference2')

# fit the model to the training set
train_model_fit_1 = smf.ols(my_model_1, data = coaches_train).fit()
# summary of model fit to the training set
print(train_model_fit_1.summary())
# training set predictions from the model fit to the training set
coaches_train['predict_totalPay'] = train_model_fit_1.fittedvalues

# test set predictions from the model fit to the training set
coaches_test['predict_totalPay'] = train_model_fit_1.predict(coaches_test)
print(coaches_test)
print(coaches_train)

# https://stackoverflow.com/questions/30787901/how-to-get-a-value-from-a-pandas-dataframe-and-not-the-index-and-object-type
SyracuseSuggestedSalary = coaches_test[coaches_test.School == "Syracuse"]["predict_totalPay"].item()

# https://stackoverflow.com/questions/44176475/printing-float-number-as-integer-in-python/44176556
# https://stackoverflow.com/questions/60610101/string-formatting-dollar-sign-in-python
# https://stackoverflow.com/questions/5180365/python-add-comma-into-number-string
print("The Suggested Salary for the next Syracuse Football coach is: ${:,}".format(int(SyracuseSuggestedSalary)))

# compute the proportion of response variance
# accounted for when predicting out-of-sample
print('\nProportion of Test Set Variance Accounted for: ',    round(np.power(coaches_test['TotalPay'].corr(coaches_test['predict_totalPay']),2),3))

# use the full data set to obtain an estimate of the increase in Salary for each of the variables
my_model_fit_1 = smf.ols(my_model_1, data = coaches).fit()
print(my_model_fit_1.summary())
print(my_model_fit_1.params)
print('\nIntercept of TotalPay is: ${:,}'.format(int(my_model_fit_1.params[0]))) # Intercept
print('\nEstimated Effect on Coach Salary from Sun Belt on TotalPay is: ${:,}'.format(int(my_model_fit_1.params[1])))
print('\nEstimated Effect on Coach Salary from SEC on TotalPay is: ${:,}'.format(int(my_model_fit_1.params[2]))) 
print('\nEstimated Effect on Coach Salary from AAC on TotalPay is: ${:,}'.format(int(my_model_fit_1.params[3]))) # Big East
print('\nEstimated Effect on Coach Salary from ACC on TotalPay is: ${:,}'.format(int(my_model_fit_1.params[4]))) # ACC, Syracuse
print('\nEstimated Effect on Coach Salary from Big 12 on TotalPay is: ${:,}'.format(int(my_model_fit_1.params[5]))) 
print('\nEstimated Effect on Coach Salary from Big Ten on TotalPay is: ${:,}'.format(int(my_model_fit_1.params[6]))) 
print('\nEstimated Effect on Coach Salary from C-USA on TotalPay is: ${:,}'.format(int(my_model_fit_1.params[7]))) 
print('\nEstimated Effect on Coach Salary from MAC on TotalPay is: ${:,}'.format(int(my_model_fit_1.params[8]))) 
print('\nEstimated Effect on Coach Salary from Mt. West on TotalPay is: ${:,}'.format(int(my_model_fit_1.params[9]))) 
print('\nEstimated Effect on Coach Salary from Pac-12 on TotalPay is: ${:,}'.format(int(my_model_fit_1.params[10]))) 
print('\nEstimated Effect on Coach Salary from Stadium Size on TotalPay is: ${:,}'.format(int(my_model_fit_1.params[11]))) 
print('\nEstimated Effect on Coach Salary from guruRank on TotalPay is: ${:,}'.format(int(my_model_fit_1.params[12]))) 
print('\nEstimated Effect on Coach Salary from OppoRank on TotalPay is: ${:,}'.format(int(my_model_fit_1.params[13]))) 
print('\nEstimated Effect on Coach Salary from WinPct on TotalPay is: ${:,}'.format(int(my_model_fit_1.params[14]))) 
print('\nEstimated Effect on Coach Salary from Student Athlete on TotalPay is: ${:,}'.format(int(my_model_fit_1.params[15]))) 
print('\nEstimated Effect on Coach Salary from Grad Success Rate on TotalPay is: ${:,}'.format(int(my_model_fit_1.params[16]))) 

#################################################
############ starting to build 2nd model ########
#################################################
# employ training-and-test regimen for model validation
np.random.seed(1234)
coaches_train2 = coaches[coaches['runiform'] >= 0.33]
coaches_test2 = coaches[coaches['runiform'] < 0.33]
# check training data frame
print('\ncoaches_train2 data frame (rows, columns): ',coaches_train2.shape)
print(coaches_train2.head())
# check test data frame
print('\ncoaches_test2 data frame (rows, columns): ',coaches_test2.shape)
print(coaches_test2.head())

# specify a simple model with bobblehead entered last
my_model_2 = str('TotalPay ~ StadiumSize + guruRank + OppoRank + WinPct + Conference2')

# fit the model to the training set
train_model_fit_2 = smf.ols(my_model_2, data = coaches_train2).fit()
# summary of model fit to the training set
print(train_model_fit_2.summary())
# training set predictions from the model fit to the training set
coaches_train2['predict_totalPay'] = train_model_fit_2.fittedvalues

# test set predictions from the model fit to the training set
coaches_test2['predict_totalPay'] = train_model_fit_2.predict(coaches_test2)
#print(coaches_test2)
#print(coaches_train2)

# https://stackoverflow.com/questions/30787901/how-to-get-a-value-from-a-pandas-dataframe-and-not-the-index-and-object-type
SyracuseSuggestedSalary2 = coaches_test2[coaches_test2.School == "Syracuse"]["predict_totalPay"].item()

# https://stackoverflow.com/questions/44176475/printing-float-number-as-integer-in-python/44176556
# https://stackoverflow.com/questions/60610101/string-formatting-dollar-sign-in-python
# https://stackoverflow.com/questions/5180365/python-add-comma-into-number-string
print("The Suggested Salary for the next Syracuse Football coach is: ${:,}".format(int(SyracuseSuggestedSalary2)))

# compute the proportion of response variance
# accounted for when predicting out-of-sample
print('\nProportion of Test Set Variance Accounted for: ',    round(np.power(coaches_test2['TotalPay'].corr(coaches_test2['predict_totalPay']),2),3))

# use the full data set to obtain an estimate of the increase in Salary for each of the variables
my_model_fit_2 = smf.ols(my_model_2, data = coaches).fit()
print(my_model_fit_2.summary())
print(my_model_fit_2.params)
print('\nIntercept of TotalPay is: ${:,}'.format(int(my_model_fit_2.params[0]))) # Intercept
print('\nEstimated Effect on Coach Salary from Sun Belt on TotalPay is: ${:,}'.format(int(my_model_fit_2.params[1])))
print('\nEstimated Effect on Coach Salary from SEC on TotalPay is: ${:,}'.format(int(my_model_fit_2.params[2]))) 
print('\nEstimated Effect on Coach Salary from AAC on TotalPay is: ${:,}'.format(int(my_model_fit_2.params[3]))) # Big East
print('\nEstimated Effect on Coach Salary from ACC on TotalPay is: ${:,}'.format(int(my_model_fit_2.params[4]))) # ACC, Syracuse
print('\nEstimated Effect on Coach Salary from Big 12 on TotalPay is: ${:,}'.format(int(my_model_fit_2.params[5]))) 
print('\nEstimated Effect on Coach Salary from Big Ten on TotalPay is: ${:,}'.format(int(my_model_fit_2.params[6]))) 
print('\nEstimated Effect on Coach Salary from C-USA on TotalPay is: ${:,}'.format(int(my_model_fit_2.params[7]))) 
print('\nEstimated Effect on Coach Salary from MAC on TotalPay is: ${:,}'.format(int(my_model_fit_2.params[8]))) 
print('\nEstimated Effect on Coach Salary from Mt. West on TotalPay is: ${:,}'.format(int(my_model_fit_2.params[9]))) 
print('\nEstimated Effect on Coach Salary from Pac-12 on TotalPay is: ${:,}'.format(int(my_model_fit_2.params[10]))) 
print('\nEstimated Effect on Coach Salary from Stadium Size on TotalPay is: ${:,}'.format(int(my_model_fit_2.params[11]))) 
print('\nEstimated Effect on Coach Salary from guruRank on TotalPay is: ${:,}'.format(int(my_model_fit_2.params[12]))) 
print('\nEstimated Effect on Coach Salary from OppoRank on TotalPay is: ${:,}'.format(int(my_model_fit_2.params[13]))) 
print('\nEstimated Effect on Coach Salary from WinPct on TotalPay is: ${:,}'.format(int(my_model_fit_2.params[14]))) 


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.reset_index.html
modelparams = pd.DataFrame(data = my_model_fit_1.params).reset_index(0)
modelparams.columns = ["Variable","Model_1_Value"]
model2params = pd.DataFrame(my_model_fit_2.params).reset_index(0)
model2params.columns = ["Variable","Model_2_Value"]

# https://stackoverflow.com/questions/58774232/is-there-a-vlookup-function-in-python-that-allows-you-to-return-a-value-from-a-c
modelparams_merged = modelparams.merge(model2params, left_on = ['Variable'], right_on = ['Variable'], how = "left")

# https://datascience.stackexchange.com/questions/45314/dataframe-has-no-column-names-how-to-add-a-header
#print(modelparams)
#print(model2params)
#print(modelparams_merged)
modelparams_merged.dropna(inplace=True)
#print(modelparams_merged)

# https://stackoverflow.com/questions/45393123/adding-calculated-column-in-pandas
# https://www.tutorialspoint.com/How-to-calculate-absolute-value-in-Python
modelparams_merged["Model_Abs_Difference"] = abs(modelparams_merged.Model_1_Value - modelparams_merged.Model_2_Value)
#print(modelparams_merged)
# https://stackoverflow.com/questions/21291259/convert-floats-to-ints-in-pandas
modelparams_merged = modelparams_merged.astype(int, errors = 'ignore')
print(modelparams_merged)

sns.barplot(y= "Variable", x = "Model_Abs_Difference", data = modelparams_merged).set_title('Variable Difference Boxplot')



# answers to questions
# What is the recommended salary for the Syracuse football coach?
print("Model 1 says the Suggested Salary for the next Syracuse Football coach is: ${:,}".format(int(SyracuseSuggestedSalary)))
print("Model 2 says the Suggested Salary for the next Syracuse Football coach is: ${:,}".format(int(SyracuseSuggestedSalary2)))
print("The range for Models 1 & 2 is: ${:,}".format(int(SyracuseSuggestedSalary - SyracuseSuggestedSalary2)))

# What would his salary be if we were still in the Big East? 
# Model 1
print('\nBig East (AAC) minus ACC coefficients = ${:,}'.format(int(my_model_fit_1.params[3] - my_model_fit_1.params[4]))) # this number shows difference of Big East - ACC intercept...we add this number to the suggested salary for model 1 to see what the Syracuse coach should make in the Big East
print("\nThe Suggested Salary for the next Syracuse Football coach is: ${:,}".format(int(SyracuseSuggestedSalary)))
# the sum of the 2 above numbers
print("\nModel 1 says the Suggested Salary for the next Syracuse Football coach in the Big East is: ${:,}".format(int(SyracuseSuggestedSalary + (my_model_fit_1.params[3] - my_model_fit_1.params[4]))))


# Model 2
print('\nBig East (AAC) minus ACC coefficients = ${:,}'.format(int(my_model_fit_2.params[3] - my_model_fit_2.params[4]))) # this number shows difference of Big East - ACC intercept...we add this number to the suggested salary for model 2 to see what the Syracuse coach should make in the Big East
print("\nThe Suggested Salary for the next Syracuse Football coach is: ${:,}".format(int(SyracuseSuggestedSalary2)))
# the sum of the 2 above numbers
print("\nModel 2 says the Suggested Salary for the next Syracuse Football coach in the Big East is: ${:,}".format(int(SyracuseSuggestedSalary2 + (my_model_fit_2.params[3] - my_model_fit_2.params[4]))))


# What if we went to the Big Ten?
# Model 1: param 6 references big 10, param 3 references ACC
print('\nBig 10 minus ACC coefficients = ${:,}'.format(int(my_model_fit_1.params[6] - my_model_fit_1.params[4]))) # this number shows difference of Big 10 - ACC intercept...we add this number to the suggested salary for model 1 to see what the Syracuse coach should make in the Big 10
print("\nThe Suggested Salary for the next Syracuse Football coach is: ${:,}".format(int(SyracuseSuggestedSalary)))
# the sum of the 2 above numbers
print("\nModel 1 says the Suggested Salary for the next Syracuse Football coach in the Big 10 is: ${:,}".format(int(SyracuseSuggestedSalary + (my_model_fit_1.params[6] - my_model_fit_1.params[4]))))


# Model 2: param 6 references big 10, param 3 references ACC
print('\nBig 10 minus ACC coefficients = ${:,}'.format(int(my_model_fit_2.params[6] - my_model_fit_2.params[4]))) # this number shows difference of Big 10 - ACC intercept...we add this number to the suggested salary for model 2 to see what the Syracuse coach should make in the Big 10
print("\nThe Suggested Salary for the next Syracuse Football coach is: ${:,}".format(int(SyracuseSuggestedSalary2)))
# the sum of the 2 above numbers
print("\nModel 2 says the Suggested Salary for the next Syracuse Football coach in the Big 10 is: ${:,}".format(int(SyracuseSuggestedSalary2+(my_model_fit_2.params[6] - my_model_fit_2.params[4]))))


# What schools did we drop from our data, and why?
# The schools that were dropped from the data were Baylor, BYU, Rice & SMU because they had no data for TotalPay in the original data file. Lines 56 & 57 above show the code that was used to drop these schools (any schools that had NA values).


# What effect does graduation rate have on the projected salary?
# Graduation Rate has an r^2 value of 0.13 when regressed against TotalPay (as seen in lines 170-172), which means that 13% of TotalPay is accounted for by using Grad Success Rate as a predictive variable.
# This is not a high r^2 value (highest is 1) as it does not account for very much of the TotalPay variable.
# Additionally, the p-value of Grad Success Rate was higher than 0.05 in the 2 regression models that it was included for, which means it is not a significant variable in the TotalPay variable prediction.
# Lastly, the output of the line of code below suggests that each point of Grad Success Rate is worth $9,133.
print('\nEstimated Effect on Coach Salary from Grad Success Rate on TotalPay is: ${:,}'.format(int(my_model_fit_1.params[16]))) # Grad Success Rate


# How good is our model?
# Model 1
print(my_model_fit_1.summary())
# Adjusted R Squared of 0.784

# Model 2
print(my_model_fit_2.summary())
# Adjusted R Squared of 0.787


# The two models have a similar adjust r^2 of ~0.785. This suggests that 78% of variance is accounted for within the models. These are moderately good results.

# What is the single biggest impact on salary size?
# Outside of the Intercept, Models 1 & 2 both suggest that being a coach in the SEC makes the largest impact on salary size. The coefficient for the SEC had the largest value (~$76,000 higher than being a coach in the Big 10 for model 1, ~$10,000 higher than being a coach in the Big 10 for model 2) of the predictive variables.
print(modelparams_merged)
modelparamsimpact = modelparams_merged.drop([0])
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.max.html
# The values from line 390 need to be compared to the list in line 391 to find which variable they correspond with. The max values for Models 1 & 2 both show that the SEC is the largest variable.
print(modelparamsimpact.max())
print(modelparamsimpact)
