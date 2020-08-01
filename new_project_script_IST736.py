# -*- coding: utf-8 -*-
"""
Created on Tue May 19 08:52:42 2020

@author: GRAFFJE
"""

############################################################


#import nltk
import pandas as pd
import numpy as np
import sklearn
#import re  

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#from nltk.tokenize import word_tokenize
#from nltk.probability import FreqDist
#from nltk.corpus import stopwords

import matplotlib.pyplot as plt

## For Stemming
#from nltk.stem import PorterStemmer
#from nltk.tokenize import sent_tokenize, word_tokenize

#import os
#import os.path

#from nltk.stem.wordnet import WordNetLemmatizer
#from nltk.stem.porter import PorterStemmer
#import string

# for vis
import seaborn as sns
#from wordcloud import WordCloud

tempfile = pd.ExcelFile("H:/Jeremiah Graff/0-Jeremiah Master's Degree/5-April 2020 Classes/IST 736/project/new_files_from_david/master_speech.xlsx")
rawfile = tempfile.parse(0)
print(rawfile)
rawfile['comb_text'] = rawfile[rawfile.columns[19:]].apply(
    lambda x: ' '.join(x.dropna().astype(str)), axis = 1)

## party
# 0 = democrat, 1 = republican
df_party = rawfile.copy(deep=True)
df_party = df_party.filter(['id', 'party_num', 'comb_text'])
print(df_party)

PartyList = []
TextList = []
IndexList = []

# https://www.geeksforgeeks.org/create-a-list-from-rows-in-pandas-dataframe/
for row in df_party.itertuples():
    partylabel = row.party_num
    textlabel = row.comb_text
    textlabel = textlabel.replace('\n',' ') # remove \n
    textlabel = textlabel.replace("\'","'")
    textlabel = textlabel.replace('\t',' ') # remove \t
    #textlabel = textlabel.replace(',',' ') # remove ,
    #textlabel = textlabel.replace('.',' ') # remove .
    #textlabel = textlabel.replace('"',' ') # remove "
    #textlabel = textlabel.replace("'",' ') # remove '
    textlabel = textlabel.replace('nan',' ') # remove nan
    #textlabel = textlabel.replace('-',' ') # remove -
    #textlabel = textlabel.replace(' - ',' ') # remove -
    #textlabel = textlabel.replace('?',' ') # remove ?
    #textlabel = textlabel.replace('!',' ') # remove !
    #textlabel = textlabel.replace(':',' ') # remove :
    #textlabel = textlabel.replace(';',' ') # remove ;
    #textlabel = textlabel.replace('  ',' ') # remove double spaces
    indexlabel = row.id
    TextList.append(textlabel)
    PartyList.append(partylabel)
    IndexList.append(indexlabel)


print(IndexList)
print(PartyList)
print(TextList)

## next label
## winner
# 1 = won presidency, 0 = did not win
df_winner = rawfile.copy(deep=True)
df_winner = df_winner.filter(['id', 'winner', 'comb_text'])
print(df_winner)

WinnerList = []

for row in df_winner.itertuples():
    winnerlabel = row.winner
    WinnerList.append(winnerlabel)
    
print(WinnerList)

## next label
## sentiment
# 1 = positive, 0 = negative
df_sent = rawfile.copy(deep=True)
df_sent = df_sent.filter(['id', 'sentiment', 'comb_text'])
print(df_sent)

SentimentList = []

for row in df_sent.itertuples():
    sentimentlabel = row.sentiment
    SentimentList.append(sentimentlabel)
    
print(SentimentList)

## next label
## incumbent_candidate
# 1 = positive, 0 = negative
df_incum_cand = rawfile.copy(deep=True)
df_incum_cand = df_incum_cand.filter(['id', 'incumbent_candidate', 'comb_text'])
print(df_incum_cand)

IncumCandList = []

for row in df_incum_cand.itertuples():
    incum_candlabel = row.incumbent_candidate
    IncumCandList.append(incum_candlabel)
    
print(IncumCandList)

## next label
## incumbent_party
# 1 = positive, 0 = negative
df_incum_party = rawfile.copy(deep=True)
df_incum_party = df_incum_party.filter(['id', 'incumbent_party', 'comb_text'])
print(df_incum_party)

IncumPartyList = []

for row in df_incum_party.itertuples():
    incum_partylabel = row.incumbent_party
    IncumPartyList.append(incum_partylabel)
    
print(IncumPartyList)

## next label
## unemployment
# 1 = positive, 0 = negative
df_unemployment = rawfile.copy(deep=True)
df_unemployment = df_unemployment.filter(['id', 'unemployment', 'comb_text'])
print(df_unemployment)

UnemploymentList = []

for row in df_unemployment.itertuples():
    unemploymentlabel = row.unemployment
    UnemploymentList.append(unemploymentlabel)
    
print(UnemploymentList)

## next label
## GDP
# 1 = positive, 0 = negative
df_GDP = rawfile.copy(deep=True)
df_GDP = df_GDP.filter(['id', 'GDP', 'comb_text'])
print(df_GDP)

GDPList = []

for row in df_GDP.itertuples():
    GDPlabel = row.GDP
    GDPList.append(GDPlabel)
    
print(GDPList)

## next label
## Inflation
# 1 = positive, 0 = negative
df_Inflation = rawfile.copy(deep=True)
df_Inflation = df_Inflation.filter(['id', 'Inflation', 'comb_text'])
print(df_Inflation)

InflationList = []

for row in df_Inflation.itertuples():
    Inflationlabel = row.Inflation
    InflationList.append(Inflationlabel)
    
print(InflationList)

## next label
## satisfaction
# 1 = positive, 0 = negative
df_satisfaction = rawfile.copy(deep=True)
df_satisfaction = df_satisfaction.filter(['id', 'satisfaction', 'comb_text'])
df_satisfaction = df_satisfaction[df_satisfaction['satisfaction'].notna()]
print(df_satisfaction)

SatisfactionList = []

for row in df_satisfaction.itertuples():
    satisfactionlabel = row.satisfaction
    SatisfactionList.append(satisfactionlabel)

SatisfactionList = [ int(x) for x in SatisfactionList ]
print(SatisfactionList)

## next label
## real_income_growth
# 1 = positive, 0 = negative
df_real_inc_growth = rawfile.copy(deep=True)
df_real_inc_growth = df_real_inc_growth.filter(['id', 'real_income_growth', 'comb_text'])
df_real_inc_growth = df_real_inc_growth[df_real_inc_growth['real_income_growth'].notna()]
print(df_real_inc_growth)

RealIncomeGrowthList = []

for row in df_real_inc_growth.itertuples():
    real_inc_growthlabel = row.real_income_growth
    RealIncomeGrowthList.append(real_inc_growthlabel)

RealIncomeGrowthList = [ int(x) for x in RealIncomeGrowthList ]
print(RealIncomeGrowthList)

## next label
## pres_approval
# 1 = positive, 0 = negative
df_pres_approval = rawfile.copy(deep=True)
df_pres_approval = df_pres_approval.filter(['id', 'pres_approval', 'comb_text'])
df_pres_approval = df_pres_approval[df_pres_approval['pres_approval'].notna()]
print(df_pres_approval)

PresApprovalList = []

for row in df_pres_approval.itertuples():
    pres_approvallabel = row.pres_approval
    PresApprovalList.append(pres_approvallabel)

PresApprovalList = [ int(x) for x in PresApprovalList ]
print(PresApprovalList)

## next label
## income_tax
# 1 = positive, 0 = negative
df_income_tax = rawfile.copy(deep=True)
df_income_tax = df_income_tax.filter(['id', 'income_tax', 'comb_text'])
df_income_tax = df_income_tax[df_income_tax['income_tax'].notna()]
print(df_income_tax)

IncomeTaxList = []

for row in df_income_tax.itertuples():
    income_taxlabel = row.income_tax
    IncomeTaxList.append(income_taxlabel)

IncomeTaxList = [ int(x) for x in IncomeTaxList ]
print(IncomeTaxList)

## next label
## djia_volume
# 1 = positive, 0 = negative
df_djia_volume = rawfile.copy(deep=True)
df_djia_volume = df_djia_volume.filter(['id', 'djia_volume', 'comb_text'])
df_djia_volume = df_djia_volume[df_djia_volume['djia_volume'].notna()]
print(df_djia_volume)

DjiaVolumeList = []

for row in df_djia_volume.itertuples():
    djia_volumelabel = row.djia_volume
    DjiaVolumeList.append(djia_volumelabel)

DjiaVolumeList = [ int(x) for x in DjiaVolumeList ]
print(DjiaVolumeList)

## next label
## cpi
# 1 = positive, 0 = negative
df_cpi = rawfile.copy(deep=True)
df_cpi = df_cpi.filter(['id', 'cpi', 'comb_text'])
df_cpi = df_cpi[df_cpi['cpi'].notna()]
print(df_cpi)

CpiList = []

for row in df_cpi.itertuples():
    cpilabel = row.cpi
    CpiList.append(cpilabel)

CpiList = [ int(x) for x in CpiList ]
print(CpiList)

## next label
## cci_index
# 1 = positive, 0 = negative
df_cci_index = rawfile.copy(deep=True)
df_cci_index = df_cci_index.filter(['id', 'cci_index', 'comb_text'])
df_cci_index = df_cci_index[df_cci_index['cci_index'].notna()]
print(df_cci_index)

CciIndexList = []

for row in df_cci_index.itertuples():
    cci_indexlabel = row.cci_index
    CciIndexList.append(cci_indexlabel)

CciIndexList = [ int(x) for x in CciIndexList ]
print(CciIndexList)


#######################################################################################
### all of the Lists in one spot for building out labeled dataframes for model building

print(IndexList)
print(TextList)
print(PartyList)
print(WinnerList)
print(SentimentList)
print(IncumCandList)
print(IncumPartyList)
print(UnemploymentList)
print(GDPList)
print(InflationList)
print(SatisfactionList)
print(RealIncomeGrowthList)
print(PresApprovalList)
print(IncomeTaxList)
print(DjiaVolumeList)
print(CpiList)
print(CciIndexList)
############# Now - use CountVectorizer.....................

MyVect=CountVectorizer(input='content', stop_words='english', token_pattern='(?u)[a-zA-Z]+')

MyVect_bool=CountVectorizer(input='content', stop_words='english', token_pattern='(?u)[a-zA-Z]+', binary = True)

MyVect_tf=TfidfVectorizer(input='content', stop_words='english', token_pattern='(?u)[a-zA-Z]+')

## NOw I can vectorize using my list of complete paths to my files
X_text=MyVect.fit_transform(TextList)

Bool_X_text=MyVect.fit_transform(TextList)

tf_X_text=MyVect.fit_transform(TextList)

## Hmm - that's not quite what we want...
## Let's get the feature names which ARE the words
ColumnNamesText=MyVect.get_feature_names()
print(ColumnNamesText)

## OK good - but we want a document topic model A DTM (matrix of counts)
VectDF=pd.DataFrame(X_text.toarray(), columns=ColumnNamesText, index = IndexList)
print(VectDF)

bool_VectDF=pd.DataFrame(Bool_X_text.toarray(), columns=ColumnNamesText, index = IndexList)
print(bool_VectDF)

tf_VectDF=pd.DataFrame(tf_X_text.toarray(), columns=ColumnNamesText, index = IndexList)
print(tf_VectDF)

#################################################################################################
#################################################################################################
### everything from this point down is a matter of find and replace for each of the variables ###
#################################################################################################
#################################################################################################

### starting to build a model for Party
PartyVectDF = VectDF.copy(deep=True)
PartyVectDF.insert(loc=0, column='LABEL', value=PartyList)
print(PartyVectDF)

bool_PartyVectDF = bool_VectDF.copy(deep=True)
bool_PartyVectDF.insert(loc=0, column='LABEL', value=PartyList)
print(bool_PartyVectDF)

tf_PartyVectDF = tf_VectDF.copy(deep=True)
tf_PartyVectDF.insert(loc=0, column='LABEL', value=PartyList)
print(tf_PartyVectDF)


##--------------------------------------------------------
####################################################################
###################### model Data Build ############################
####################################################################
##--------------------------------------------------------
from sklearn.model_selection import train_test_split

## Copy everything below here to rebuild for sentiment data
TrainDF, TestDF = train_test_split(PartyVectDF, test_size=0.3)
print(TrainDF.head())

## Now we have a training set and a testing set. 
print("\nThe training set is:")
print(TrainDF)
print("\nThe testing set is:")
print(TestDF)

## IMPORTANT - YOU CANNOT LEAVE LABELS ON THE TEST SET
## Save labels
TestLabels=TestDF["LABEL"]
print(TestLabels)
## remove labels
TestDF_nolabels = TestDF.drop(["LABEL"], axis=1)
print(TestDF_nolabels)

TrainDF_nolabels=TrainDF.drop(["LABEL"], axis=1)
print(TrainDF_nolabels)

TrainLabels=TrainDF["LABEL"]
print(TrainLabels)

############# Do the same for the binary dataframe
## Create Train/Test for TrainDFB
TrainDFB, TestDFB = train_test_split(bool_PartyVectDF, test_size=0.3)
print(TrainDFB.head())

TrainDFB_nolabels=TrainDFB.drop(["LABEL"], axis=1)
print(TrainDFB_nolabels)
TrainLabelsB=TrainDFB["LABEL"]
print(TrainLabelsB)

TestDFB_nolabels=TestDFB.drop(["LABEL"], axis=1)
print(TestDFB_nolabels)
TestLabelsB=TestDFB["LABEL"]
print(TestLabelsB)

############# Do the same for the tfidf dataframe
## Create Train/Test for TrainDF tfidf
TrainDF_tf, TestDF_tf = train_test_split(tf_PartyVectDF, test_size=0.3)
print(TrainDF_tf.head())

TrainDF_tf_nolabels=TrainDF_tf.drop(["LABEL"], axis=1)
print(TrainDF_tf_nolabels)
TrainLabels_tf=TrainDF_tf["LABEL"]
print(TrainLabels_tf)

TestDF_tf_nolabels=TestDF_tf.drop(["LABEL"], axis=1)
print(TestDF_tf_nolabels)
TestLabels_tf=TestDF_tf["LABEL"]
print(TestLabels_tf)

####################################################################
########################### Naive Bayes ############################
####################################################################
from sklearn.naive_bayes import MultinomialNB
#https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB.fit
#Create the modeler
ModelNB= MultinomialNB()
## When you look up this model, you learn that it wants the 
## DF seperate from the labels

ModelNB.fit(TrainDF_nolabels, TrainLabels)
Prediction = ModelNB.predict(TestDF_nolabels)
print("\nThe prediction from NB is:")
print(Prediction)
print("\nThe actual labels are:")
print(TestLabels)
## confusion matrix
from sklearn.metrics import confusion_matrix
## The confusion matrix is square and is labels X labels
## We ahve two labels, so ours will be 2X2
#The matrix shows
## rows are the true labels
## columns are predicted
## it is al[habetical
## The numbers are how many 
cnf_matrix = confusion_matrix(TestLabels, Prediction)
print("\nThe confusion matrix is:")
print(cnf_matrix)
### prediction probabilities
## columns are the labels in alphabetical order
## The decinal in the matrix are the prob of being
## that label
print(np.round(ModelNB.predict_proba(TestDF_nolabels),2))

from sklearn.metrics import classification_report
target_names = ['0','1']
print(classification_report(TestLabels, Prediction, target_names=target_names))

#######################################################
### Bernoulli #########################################
#######################################################
### NOTE TO CLASS: This should use the Binary


from sklearn.naive_bayes import BernoulliNB
BernModel = BernoulliNB()
BernModel.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
B_Prediction = BernModel.predict(TestDFB_nolabels)
print("\nBernoulli prediction is:\n")
print(B_Prediction)
print("\nActual:")
print(TestLabelsB)

Bern_cnf_matrix = confusion_matrix(TestLabelsB, B_Prediction)
print("\nThe confusion matrix is:")
print(Bern_cnf_matrix)

B_target_names = ['0','1']
print(classification_report(TestLabelsB, B_Prediction, target_names=B_target_names))

####################################################################
########################### Naive Bayes on tfidf ###################
####################################################################
tf_ModelNB= MultinomialNB()
## When you look up this model, you learn that it wants the 
## DF seperate from the labels

tf_ModelNB.fit(TrainDF_tf_nolabels, TrainLabels_tf)
tf_Prediction = tf_ModelNB.predict(TestDF_tf_nolabels)
print("\nThe tf prediction from NB is:")
print(tf_Prediction)
print("\nThe actual tf labels are:")
print(TestLabels_tf)
## confusion matrix
tf_cnf_matrix = confusion_matrix(TestLabels_tf, tf_Prediction)
print("\nThe tf confusion matrix is:")
print(tf_cnf_matrix)
### prediction probabilities
## columns are the labels in alphabetical order
## The decinal in the matrix are the prob of being
## that label
print(np.round(ModelNB.predict_proba(TestDF_tf_nolabels),2))

tf_target_names = ['0','1']
print(classification_report(TestLabels_tf, tf_Prediction, target_names=tf_target_names))



#############################################
###########  SVM  for regular model #1 ######
#############################################
from sklearn.svm import LinearSVC
print(TrainDF_nolabels.head())
print(TrainLabels.head())

Party_SVM_Model=LinearSVC(C=.01)
Party_SVM_Model.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Party_svm_predict = Party_SVM_Model.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(Party_svm_predict)
print("Actual:")
print(TestLabels)

from sklearn.metrics import confusion_matrix
Party_SVM_matrix = confusion_matrix(TestLabels, Party_svm_predict)
print("\nThe confusion matrix is:")
print(Party_SVM_matrix)
print("\n\n")

from sklearn.metrics import classification_report
Party_svm_target_names = ['0','1']
print(classification_report(TestLabels, Party_svm_predict, target_names = Party_svm_target_names))

Party_SVM_reg_FP = Party_SVM_matrix[0][1] 
Party_SVM_reg_FN = Party_SVM_matrix[1][0]
Party_SVM_reg_TP = Party_SVM_matrix[1][1]
Party_SVM_reg_TN = Party_SVM_matrix[0][0]

# Overall accuracy
Party_SVM_reg_ACC = (Party_SVM_reg_TP + Party_SVM_reg_TN)/(Party_SVM_reg_TP + Party_SVM_reg_FP + Party_SVM_reg_FN + Party_SVM_reg_TN)
print(Party_SVM_reg_ACC)

PartyAccuracyDict = {}
PartyAccuracyDict.update({'Party_SVM_reg_ACC': Party_SVM_reg_ACC})
print(PartyAccuracyDict)

#############################################
###########  SVM  for regular model #2 ######
#############################################

Party_SVM_Model2=LinearSVC(C=1)
Party_SVM_Model2.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Party_svm_predict2 = Party_SVM_Model2.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(Party_svm_predict2)
print("Actual:")
print(TestLabels)

Party_SVM_matrix2 = confusion_matrix(TestLabels, Party_svm_predict2)
print("\nThe confusion matrix is:")
print(Party_SVM_matrix2)
print("\n\n")

Party_svm_target_names2 = ['0','1']
print(classification_report(TestLabels, Party_svm_predict2, target_names = Party_svm_target_names2))

Party_SVM_reg2_FP = Party_SVM_matrix2[0][1] 
Party_SVM_reg2_FN = Party_SVM_matrix2[1][0]
Party_SVM_reg2_TP = Party_SVM_matrix2[1][1]
Party_SVM_reg2_TN = Party_SVM_matrix2[0][0]

# Overall accuracy
Party_SVM_reg2_ACC = (Party_SVM_reg2_TP + Party_SVM_reg2_TN)/(Party_SVM_reg2_TP + Party_SVM_reg2_FP + Party_SVM_reg2_FN + Party_SVM_reg2_TN)
print(Party_SVM_reg2_ACC)

PartyAccuracyDict.update({'Party_SVM_reg2_ACC': Party_SVM_reg2_ACC})
print(PartyAccuracyDict)

#############################################
###########  SVM  for regular model #3 ######
#############################################

Party_SVM_Model3=LinearSVC(C=100)
Party_SVM_Model3.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Party_svm_predict3 = Party_SVM_Model3.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(Party_svm_predict3)
print("Actual:")
print(TestLabels)

Party_SVM_matrix3 = confusion_matrix(TestLabels, Party_svm_predict3)
print("\nThe confusion matrix is:")
print(Party_SVM_matrix3)
print("\n\n")

from sklearn.metrics import classification_report
Party_svm_target_names3 = ['0','1']
print(classification_report(TestLabels, Party_svm_predict3, target_names = Party_svm_target_names3))

Party_SVM_reg3_FP = Party_SVM_matrix3[0][1] 
Party_SVM_reg3_FN = Party_SVM_matrix3[1][0]
Party_SVM_reg3_TP = Party_SVM_matrix3[1][1]
Party_SVM_reg3_TN = Party_SVM_matrix3[0][0]

# Overall accuracy
Party_SVM_reg3_ACC = (Party_SVM_reg3_TP + Party_SVM_reg3_TN)/(Party_SVM_reg3_TP + Party_SVM_reg3_FP + Party_SVM_reg3_FN + Party_SVM_reg3_TN)
print(Party_SVM_reg3_ACC)

PartyAccuracyDict.update({'Party_SVM_reg3_ACC': Party_SVM_reg3_ACC})
print(PartyAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #1 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

Party_B_SVM_Model=LinearSVC(C=100)
Party_B_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Party_b_svm_predict = Party_B_SVM_Model.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(Party_b_svm_predict)
print("\nActual:")
print(TestLabelsB)

Party_B_SVM_matrix = confusion_matrix(TestLabelsB, Party_b_svm_predict)
print("\nThe confusion matrix is:")
print(Party_B_SVM_matrix)
print("\n\n")

Party_svm_B_target_names = ['0','1']
print(classification_report(TestLabelsB, Party_b_svm_predict, target_names = Party_svm_B_target_names))

Party_SVM_bool_FP = Party_B_SVM_matrix[0][1] 
Party_SVM_bool_FN = Party_B_SVM_matrix[1][0]
Party_SVM_bool_TP = Party_B_SVM_matrix[1][1]
Party_SVM_bool_TN = Party_B_SVM_matrix[0][0]

# Overall accuracy
Party_SVM_bool_ACC = (Party_SVM_bool_TP + Party_SVM_bool_TN)/(Party_SVM_bool_TP + Party_SVM_bool_FP + Party_SVM_bool_FN + Party_SVM_bool_TN)
print(Party_SVM_bool_ACC)

PartyAccuracyDict.update({'Party_SVM_bool_ACC': Party_SVM_bool_ACC})
print(PartyAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #2 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

Party_B_SVM_Model2=LinearSVC(C=1)
Party_B_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Party_b_svm_predict2 = Party_B_SVM_Model2.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(Party_b_svm_predict2)
print("\nActual:")
print(TestLabelsB)

Party_B_SVM_matrix2 = confusion_matrix(TestLabelsB, Party_b_svm_predict2)
print("\nThe confusion matrix is:")
print(Party_B_SVM_matrix2)
print("\n\n")

Party_svm_B_target_names2 = ['0','1']
print(classification_report(TestLabelsB, Party_b_svm_predict2, target_names = Party_svm_B_target_names2))

Party_SVM_bool2_FP = Party_B_SVM_matrix2[0][1] 
Party_SVM_bool2_FN = Party_B_SVM_matrix2[1][0]
Party_SVM_bool2_TP = Party_B_SVM_matrix2[1][1]
Party_SVM_bool2_TN = Party_B_SVM_matrix2[0][0]

# Overall accuracy
Party_SVM_bool2_ACC = (Party_SVM_bool2_TP + Party_SVM_bool2_TN)/(Party_SVM_bool2_TP + Party_SVM_bool2_FP + Party_SVM_bool2_FN + Party_SVM_bool2_TN)
print(Party_SVM_bool2_ACC)

PartyAccuracyDict.update({'Party_SVM_bool2_ACC': Party_SVM_bool2_ACC})
print(PartyAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #3 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

Party_B_SVM_Model3=LinearSVC(C=.01)
Party_B_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Party_b_svm_predict3 = Party_B_SVM_Model3.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(Party_b_svm_predict3)
print("\nActual:")
print(TestLabelsB)

Party_B_SVM_matrix3 = confusion_matrix(TestLabelsB, Party_b_svm_predict3)
print("\nThe confusion matrix is:")
print(Party_B_SVM_matrix3)
print("\n\n")

Party_svm_B_target_names3 = ['0','1']
print(classification_report(TestLabelsB, Party_b_svm_predict3, target_names = Party_svm_B_target_names3))

Party_SVM_bool3_FP = Party_B_SVM_matrix3[0][1] 
Party_SVM_bool3_FN = Party_B_SVM_matrix3[1][0]
Party_SVM_bool3_TP = Party_B_SVM_matrix3[1][1]
Party_SVM_bool3_TN = Party_B_SVM_matrix3[0][0]

# Overall accuracy
Party_SVM_bool3_ACC = (Party_SVM_bool3_TP + Party_SVM_bool3_TN)/(Party_SVM_bool3_TP + Party_SVM_bool3_FP + Party_SVM_bool3_FN + Party_SVM_bool3_TN)
print(Party_SVM_bool3_ACC)

PartyAccuracyDict.update({'Party_SVM_bool3_ACC': Party_SVM_bool3_ACC})
print(PartyAccuracyDict)

#############################################
###########  SVM  for tfidf model #1 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

Party_tf_SVM_Model=LinearSVC(C=.001)
Party_tf_SVM_Model.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Party_tf_svm_predict = Party_tf_SVM_Model.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(Party_tf_svm_predict)
print("\nActual:")
print(TestLabels_tf)

Party_tf_SVM_matrix = confusion_matrix(TestLabels_tf, Party_tf_svm_predict)
print("\nThe confusion matrix is:")
print(Party_tf_SVM_matrix)
print("\n\n")

Party_svm_tf_target_names = ['0','1']
print(classification_report(TestLabels_tf, Party_tf_svm_predict, target_names = Party_svm_tf_target_names))

Party_SVM_tf_FP = Party_tf_SVM_matrix[0][1] 
Party_SVM_tf_FN = Party_tf_SVM_matrix[1][0]
Party_SVM_tf_TP = Party_tf_SVM_matrix[1][1]
Party_SVM_tf_TN = Party_tf_SVM_matrix[0][0]

# Overall accuracy
Party_SVM_tf_ACC = (Party_SVM_tf_TP + Party_SVM_tf_TN)/(Party_SVM_tf_TP + Party_SVM_tf_FP + Party_SVM_tf_FN + Party_SVM_tf_TN)
print(Party_SVM_tf_ACC)

PartyAccuracyDict.update({'Party_SVM_tf_ACC': Party_SVM_tf_ACC})
print(PartyAccuracyDict)

#############################################
###########  SVM  for tfidf model #2 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

Party_tf_SVM_Model2=LinearSVC(C=1)
Party_tf_SVM_Model2.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Party_tf_svm_predict2 = Party_tf_SVM_Model2.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(Party_tf_svm_predict2)
print("\nActual:")
print(TestLabels_tf)

Party_tf_SVM_matrix2 = confusion_matrix(TestLabels_tf, Party_tf_svm_predict2)
print("\nThe confusion matrix is:")
print(Party_tf_SVM_matrix2)
print("\n\n")

Party_svm_tf_target_names2 = ['0','1']
print(classification_report(TestLabels_tf, Party_tf_svm_predict2, target_names = Party_svm_tf_target_names2))

Party_SVM_tf2_FP = Party_tf_SVM_matrix2[0][1] 
Party_SVM_tf2_FN = Party_tf_SVM_matrix2[1][0]
Party_SVM_tf2_TP = Party_tf_SVM_matrix2[1][1]
Party_SVM_tf2_TN = Party_tf_SVM_matrix2[0][0]

# Overall accuracy
Party_SVM_tf2_ACC = (Party_SVM_tf2_TP + Party_SVM_tf2_TN)/(Party_SVM_tf2_TP + Party_SVM_tf2_FP + Party_SVM_tf2_FN + Party_SVM_tf2_TN)
print(Party_SVM_tf2_ACC)

PartyAccuracyDict.update({'Party_SVM_tf2_ACC': Party_SVM_tf2_ACC})
print(PartyAccuracyDict)

#############################################
###########  SVM  for tfidf model #3 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

Party_tf_SVM_Model3=LinearSVC(C=1)
Party_tf_SVM_Model3.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Party_tf_svm_predict3 = Party_tf_SVM_Model3.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(Party_tf_svm_predict3)
print("\nActual:")
print(TestLabels_tf)

Party_tf_SVM_matrix3 = confusion_matrix(TestLabels_tf, Party_tf_svm_predict3)
print("\nThe confusion matrix is:")
print(Party_tf_SVM_matrix3)
print("\n\n")

Party_svm_tf_target_names3 = ['0','1']
print(classification_report(TestLabels_tf, Party_tf_svm_predict3, target_names = Party_svm_tf_target_names3))

Party_SVM_tf3_FP = Party_tf_SVM_matrix3[0][1] 
Party_SVM_tf3_FN = Party_tf_SVM_matrix3[1][0]
Party_SVM_tf3_TP = Party_tf_SVM_matrix3[1][1]
Party_SVM_tf3_TN = Party_tf_SVM_matrix3[0][0]

# Overall accuracy
Party_SVM_tf3_ACC = (Party_SVM_tf3_TP + Party_SVM_tf3_TN)/(Party_SVM_tf3_TP + Party_SVM_tf3_FP + Party_SVM_tf3_FN + Party_SVM_tf3_TN)
print(Party_SVM_tf3_ACC)

PartyAccuracyDict.update({'Party_SVM_tf3_ACC': Party_SVM_tf3_ACC})
print(PartyAccuracyDict)

#############################################
#######  SVM  with different kernels ########
#############################################
## running on Bernoulli model since it had best accuracy above
## i want to see if the accuracy gets improved in the different kernels

#############################################
###############  Sigmoid #1 #################
#############################################

Party_sig_SVM_Model=sklearn.svm.SVC(C=10, kernel='sigmoid', 
                           verbose=True, gamma="scale")
Party_sig_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
Party_sig_svm_predict = Party_sig_SVM_Model.predict(TestDFB_nolabels)
print("\nsig SVM prediction:\n")
print(Party_sig_svm_predict)
print("\nActual:")
print(TestLabelsB)

Party_sig_SVM_matrix = confusion_matrix(TestLabelsB, Party_sig_svm_predict)
print("\nThe sig confusion matrix is:")
print(Party_sig_SVM_matrix)
print("\n\n")

Party_svm_sig_target_names = ['0','1']
print(classification_report(TestLabelsB, Party_sig_svm_predict, target_names = Party_svm_sig_target_names))

Party_SVM_sig_FP = Party_sig_SVM_matrix[0][1] 
Party_SVM_sig_FN = Party_sig_SVM_matrix[1][0]
Party_SVM_sig_TP = Party_sig_SVM_matrix[1][1]
Party_SVM_sig_TN = Party_sig_SVM_matrix[0][0]

# Overall accuracy
Party_SVM_sig_ACC = (Party_SVM_sig_TP + Party_SVM_sig_TN)/(Party_SVM_sig_TP + Party_SVM_sig_FP + Party_SVM_sig_FN + Party_SVM_sig_TN)
print(Party_SVM_sig_ACC)

PartyAccuracyDict.update({'Party_SVM_sig_ACC': Party_SVM_sig_ACC})
print(PartyAccuracyDict)

#############################################
###############  Sigmoid #2 #################
#############################################

Party_sig_SVM_Model2=sklearn.svm.SVC(C=1000, kernel='sigmoid', 
                           verbose=True, gamma="scale")
Party_sig_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
Party_sig_svm_predict2 = Party_sig_SVM_Model2.predict(TestDFB_nolabels)
print("\nsig SVM prediction 2:\n")
print(Party_sig_svm_predict2)
print("\nActual:")
print(TestLabelsB)

Party_sig_SVM_matrix2 = confusion_matrix(TestLabelsB, Party_sig_svm_predict2)
print("\nThe sig 2 confusion matrix is:")
print(Party_sig_SVM_matrix2)
print("\n\n")

Party_svm_sig_target_names2 = ['0','1']
print(classification_report(TestLabelsB, Party_sig_svm_predict2, target_names = Party_svm_sig_target_names2))

Party_SVM_sig2_FP = Party_sig_SVM_matrix2[0][1] 
Party_SVM_sig2_FN = Party_sig_SVM_matrix2[1][0]
Party_SVM_sig2_TP = Party_sig_SVM_matrix2[1][1]
Party_SVM_sig2_TN = Party_sig_SVM_matrix2[0][0]

# Overall accuracy
Party_SVM_sig2_ACC = (Party_SVM_sig2_TP + Party_SVM_sig2_TN)/(Party_SVM_sig2_TP + Party_SVM_sig2_FP + Party_SVM_sig2_FN + Party_SVM_sig2_TN)
print(Party_SVM_sig2_ACC)

PartyAccuracyDict.update({'Party_SVM_sig2_ACC': Party_SVM_sig2_ACC})
print(PartyAccuracyDict)

#############################################
###############  Sigmoid #3 #################
#############################################

Party_sig_SVM_Model3=sklearn.svm.SVC(C=.001, kernel='sigmoid', 
                           verbose=True, gamma="scale")
Party_sig_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
Party_sig_svm_predict3 = Party_sig_SVM_Model3.predict(TestDFB_nolabels)
print("\nsig SVM prediction 3:\n")
print(Party_sig_svm_predict3)
print("\nActual:")
print(TestLabelsB)

Party_sig_SVM_matrix3 = confusion_matrix(TestLabelsB, Party_sig_svm_predict3)
print("\nThe sig 3 confusion matrix is:")
print(Party_sig_SVM_matrix3)
print("\n\n")

Party_svm_sig_target_names3 = ['0','1']
print(classification_report(TestLabelsB, Party_sig_svm_predict3, target_names = Party_svm_sig_target_names3))

Party_SVM_sig3_FP = Party_sig_SVM_matrix3[0][1] 
Party_SVM_sig3_FN = Party_sig_SVM_matrix3[1][0]
Party_SVM_sig3_TP = Party_sig_SVM_matrix3[1][1]
Party_SVM_sig3_TN = Party_sig_SVM_matrix3[0][0]

# Overall accuracy
Party_SVM_sig3_ACC = (Party_SVM_sig3_TP + Party_SVM_sig3_TN)/(Party_SVM_sig3_TP + Party_SVM_sig3_FP + Party_SVM_sig3_FN + Party_SVM_sig3_TN)
print(Party_SVM_sig3_ACC)

PartyAccuracyDict.update({'Party_SVM_sig3_ACC': Party_SVM_sig3_ACC})
print(PartyAccuracyDict)

#############################################
################  Poly #1 ###################
#############################################

Party_poly_SVM_Model=sklearn.svm.SVC(C=1000, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(Party_poly_SVM_Model)
Party_poly_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
Party_poly_svm_predict = Party_poly_SVM_Model.predict(TestDFB_nolabels)
print("\npoly SVM prediction:\n")
print(Party_poly_svm_predict)
print("\nActual:")
print(TestLabelsB)

Party_poly_SVM_matrix = confusion_matrix(TestLabelsB, Party_poly_svm_predict)
print("\nThe poly confusion matrix is:")
print(Party_poly_SVM_matrix)
print("\n\n")

Party_svm_poly_target_names = ['0','1']
print(classification_report(TestLabelsB, Party_poly_svm_predict, target_names = Party_svm_poly_target_names))

Party_SVM_poly_FP = Party_poly_SVM_matrix[0][1] 
Party_SVM_poly_FN = Party_poly_SVM_matrix[1][0]
Party_SVM_poly_TP = Party_poly_SVM_matrix[1][1]
Party_SVM_poly_TN = Party_poly_SVM_matrix[0][0]

# Overall accuracy
Party_SVM_poly_ACC = (Party_SVM_poly_TP + Party_SVM_poly_TN)/(Party_SVM_poly_TP + Party_SVM_poly_FP + Party_SVM_poly_FN + Party_SVM_poly_TN)
print(Party_SVM_poly_ACC)

PartyAccuracyDict.update({'Party_SVM_poly_ACC': Party_SVM_poly_ACC})
print(PartyAccuracyDict)

#############################################
################  Poly #2 ###################
#############################################

Party_poly_SVM_Model2=sklearn.svm.SVC(C=10, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(Party_poly_SVM_Model2)
Party_poly_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
Party_poly_svm_predict2 = Party_poly_SVM_Model2.predict(TestDFB_nolabels)
print("\npoly SVM 2 prediction:\n")
print(Party_poly_svm_predict2)
print("\nActual:")
print(TestLabelsB)

Party_poly_SVM_matrix2 = confusion_matrix(TestLabelsB, Party_poly_svm_predict2)
print("\nThe poly 2 confusion matrix is:")
print(Party_poly_SVM_matrix2)
print("\n\n")

Party_svm_poly_target_names2 = ['0','1']
print(classification_report(TestLabelsB, Party_poly_svm_predict2, target_names = Party_svm_poly_target_names2))

Party_SVM_poly2_FP = Party_poly_SVM_matrix2[0][1] 
Party_SVM_poly2_FN = Party_poly_SVM_matrix2[1][0]
Party_SVM_poly2_TP = Party_poly_SVM_matrix2[1][1]
Party_SVM_poly2_TN = Party_poly_SVM_matrix2[0][0]

# Overall accuracy
Party_SVM_poly2_ACC = (Party_SVM_poly2_TP + Party_SVM_poly2_TN)/(Party_SVM_poly2_TP + Party_SVM_poly2_FP + Party_SVM_poly2_FN + Party_SVM_poly2_TN)
print(Party_SVM_poly2_ACC)

PartyAccuracyDict.update({'Party_SVM_poly2_ACC': Party_SVM_poly2_ACC})
print(PartyAccuracyDict)

#############################################
################  Poly #3 ###################
#############################################

Party_poly_SVM_Model3=sklearn.svm.SVC(C=.001, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(Party_poly_SVM_Model3)
Party_poly_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
Party_poly_svm_predict3 = Party_poly_SVM_Model3.predict(TestDFB_nolabels)
print("\npoly SVM 3 prediction:\n")
print(Party_poly_svm_predict3)
print("\nActual:")
print(TestLabelsB)

Party_poly_SVM_matrix3 = confusion_matrix(TestLabelsB, Party_poly_svm_predict3)
print("\nThe poly 3 confusion matrix is:")
print(Party_poly_SVM_matrix3)
print("\n\n")

Party_svm_poly_target_names3 = ['0','1']
print(classification_report(TestLabelsB, Party_poly_svm_predict3, target_names = Party_svm_poly_target_names3))

Party_SVM_poly3_FP = Party_poly_SVM_matrix3[0][1] 
Party_SVM_poly3_FN = Party_poly_SVM_matrix3[1][0]
Party_SVM_poly3_TP = Party_poly_SVM_matrix3[1][1]
Party_SVM_poly3_TN = Party_poly_SVM_matrix3[0][0]

# Overall accuracy
Party_SVM_poly3_ACC = (Party_SVM_poly3_TP + Party_SVM_poly3_TN)/(Party_SVM_poly3_TP + Party_SVM_poly3_FP + Party_SVM_poly3_FN + Party_SVM_poly3_TN)
print(Party_SVM_poly3_ACC)

PartyAccuracyDict.update({'Party_SVM_poly3_ACC': Party_SVM_poly3_ACC})
print(PartyAccuracyDict)

PartyVisDF = pd.DataFrame(PartyAccuracyDict.items(), index = PartyAccuracyDict.keys(), columns=['Model','Accuracy'])
print(PartyVisDF)
SortedPartyVisDF = PartyVisDF.sort_values('Accuracy', ascending = [True])
print(SortedPartyVisDF)

SortedPartyVisDF.plot.barh(y='Accuracy')

#print(IndexList)
#print(TextList)
#print(PartyList)
print(WinnerList)
print(SentimentList)
print(IncumCandList)
print(IncumPartyList)
print(UnemploymentList)
print(GDPList)
print(InflationList)
print(SatisfactionList)
print(RealIncomeGrowthList)
print(PresApprovalList)
print(IncomeTaxList)
print(DjiaVolumeList)
print(CpiList)
print(CciIndexList)

#################################################################################################
#################################################################################################
### everything from this point down is a matter of find and replace for each of the variables ###
#################################################################################################
#################################################################################################

### starting to build a model for Winner
WinnerVectDF = VectDF.copy(deep=True)
WinnerVectDF.insert(loc=0, column='LABEL', value=WinnerList)
print(WinnerVectDF)

bool_WinnerVectDF = bool_VectDF.copy(deep=True)
bool_WinnerVectDF.insert(loc=0, column='LABEL', value=WinnerList)
print(bool_WinnerVectDF)

tf_WinnerVectDF = tf_VectDF.copy(deep=True)
tf_WinnerVectDF.insert(loc=0, column='LABEL', value=WinnerList)
print(tf_WinnerVectDF)


##--------------------------------------------------------
####################################################################
###################### model Data Build ############################
####################################################################
##--------------------------------------------------------
from sklearn.model_selection import train_test_split

## Copy everything below here to rebuild for sentiment data
TrainDF, TestDF = train_test_split(WinnerVectDF, test_size=0.3)
print(TrainDF.head())

## Now we have a training set and a testing set. 
print("\nThe training set is:")
print(TrainDF)
print("\nThe testing set is:")
print(TestDF)

## IMPORTANT - YOU CANNOT LEAVE LABELS ON THE TEST SET
## Save labels
TestLabels=TestDF["LABEL"]
print(TestLabels)
## remove labels
TestDF_nolabels = TestDF.drop(["LABEL"], axis=1)
print(TestDF_nolabels)

TrainDF_nolabels=TrainDF.drop(["LABEL"], axis=1)
print(TrainDF_nolabels)

TrainLabels=TrainDF["LABEL"]
print(TrainLabels)

############# Do the same for the binary dataframe
## Create Train/Test for TrainDFB
TrainDFB, TestDFB = train_test_split(bool_WinnerVectDF, test_size=0.3)
print(TrainDFB.head())

TrainDFB_nolabels=TrainDFB.drop(["LABEL"], axis=1)
print(TrainDFB_nolabels)
TrainLabelsB=TrainDFB["LABEL"]
print(TrainLabelsB)

TestDFB_nolabels=TestDFB.drop(["LABEL"], axis=1)
print(TestDFB_nolabels)
TestLabelsB=TestDFB["LABEL"]
print(TestLabelsB)

############# Do the same for the tfidf dataframe
## Create Train/Test for TrainDF tfidf
TrainDF_tf, TestDF_tf = train_test_split(tf_WinnerVectDF, test_size=0.3)
print(TrainDF_tf.head())

TrainDF_tf_nolabels=TrainDF_tf.drop(["LABEL"], axis=1)
print(TrainDF_tf_nolabels)
TrainLabels_tf=TrainDF_tf["LABEL"]
print(TrainLabels_tf)

TestDF_tf_nolabels=TestDF_tf.drop(["LABEL"], axis=1)
print(TestDF_tf_nolabels)
TestLabels_tf=TestDF_tf["LABEL"]
print(TestLabels_tf)



#############################################
###########  SVM  for regular model #1 ######
#############################################
from sklearn.svm import LinearSVC
print(TrainDF_nolabels.head())
print(TrainLabels.head())

Winner_SVM_Model=LinearSVC(C=.01)
Winner_SVM_Model.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Winner_svm_predict = Winner_SVM_Model.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(Winner_svm_predict)
print("Actual:")
print(TestLabels)

from sklearn.metrics import confusion_matrix
Winner_SVM_matrix = confusion_matrix(TestLabels, Winner_svm_predict)
print("\nThe confusion matrix is:")
print(Winner_SVM_matrix)
print("\n\n")

from sklearn.metrics import classification_report
Winner_svm_target_names = ['0','1']
print(classification_report(TestLabels, Winner_svm_predict, target_names = Winner_svm_target_names))

Winner_SVM_reg_FP = Winner_SVM_matrix[0][1] 
Winner_SVM_reg_FN = Winner_SVM_matrix[1][0]
Winner_SVM_reg_TP = Winner_SVM_matrix[1][1]
Winner_SVM_reg_TN = Winner_SVM_matrix[0][0]

# Overall accuracy
Winner_SVM_reg_ACC = (Winner_SVM_reg_TP + Winner_SVM_reg_TN)/(Winner_SVM_reg_TP + Winner_SVM_reg_FP + Winner_SVM_reg_FN + Winner_SVM_reg_TN)
print(Winner_SVM_reg_ACC)

WinnerAccuracyDict = {}
WinnerAccuracyDict.update({'Winner_SVM_reg_ACC': Winner_SVM_reg_ACC})
print(WinnerAccuracyDict)

#############################################
###########  SVM  for regular model #2 ######
#############################################

Winner_SVM_Model2=LinearSVC(C=1)
Winner_SVM_Model2.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Winner_svm_predict2 = Winner_SVM_Model2.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(Winner_svm_predict2)
print("Actual:")
print(TestLabels)

Winner_SVM_matrix2 = confusion_matrix(TestLabels, Winner_svm_predict2)
print("\nThe confusion matrix is:")
print(Winner_SVM_matrix2)
print("\n\n")

Winner_svm_target_names2 = ['0','1']
print(classification_report(TestLabels, Winner_svm_predict2, target_names = Winner_svm_target_names2))

Winner_SVM_reg2_FP = Winner_SVM_matrix2[0][1] 
Winner_SVM_reg2_FN = Winner_SVM_matrix2[1][0]
Winner_SVM_reg2_TP = Winner_SVM_matrix2[1][1]
Winner_SVM_reg2_TN = Winner_SVM_matrix2[0][0]

# Overall accuracy
Winner_SVM_reg2_ACC = (Winner_SVM_reg2_TP + Winner_SVM_reg2_TN)/(Winner_SVM_reg2_TP + Winner_SVM_reg2_FP + Winner_SVM_reg2_FN + Winner_SVM_reg2_TN)
print(Winner_SVM_reg2_ACC)

WinnerAccuracyDict.update({'Winner_SVM_reg2_ACC': Winner_SVM_reg2_ACC})
print(WinnerAccuracyDict)

#############################################
###########  SVM  for regular model #3 ######
#############################################

Winner_SVM_Model3=LinearSVC(C=100)
Winner_SVM_Model3.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Winner_svm_predict3 = Winner_SVM_Model3.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(Winner_svm_predict3)
print("Actual:")
print(TestLabels)

Winner_SVM_matrix3 = confusion_matrix(TestLabels, Winner_svm_predict3)
print("\nThe confusion matrix is:")
print(Winner_SVM_matrix3)
print("\n\n")

from sklearn.metrics import classification_report
Winner_svm_target_names3 = ['0','1']
print(classification_report(TestLabels, Winner_svm_predict3, target_names = Winner_svm_target_names3))

Winner_SVM_reg3_FP = Winner_SVM_matrix3[0][1] 
Winner_SVM_reg3_FN = Winner_SVM_matrix3[1][0]
Winner_SVM_reg3_TP = Winner_SVM_matrix3[1][1]
Winner_SVM_reg3_TN = Winner_SVM_matrix3[0][0]

# Overall accuracy
Winner_SVM_reg3_ACC = (Winner_SVM_reg3_TP + Winner_SVM_reg3_TN)/(Winner_SVM_reg3_TP + Winner_SVM_reg3_FP + Winner_SVM_reg3_FN + Winner_SVM_reg3_TN)
print(Winner_SVM_reg3_ACC)

WinnerAccuracyDict.update({'Winner_SVM_reg3_ACC': Winner_SVM_reg3_ACC})
print(WinnerAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #1 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

Winner_B_SVM_Model=LinearSVC(C=100)
Winner_B_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Winner_b_svm_predict = Winner_B_SVM_Model.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(Winner_b_svm_predict)
print("\nActual:")
print(TestLabelsB)

Winner_B_SVM_matrix = confusion_matrix(TestLabelsB, Winner_b_svm_predict)
print("\nThe confusion matrix is:")
print(Winner_B_SVM_matrix)
print("\n\n")

Winner_svm_B_target_names = ['0','1']
print(classification_report(TestLabelsB, Winner_b_svm_predict, target_names = Winner_svm_B_target_names))

Winner_SVM_bool_FP = Winner_B_SVM_matrix[0][1] 
Winner_SVM_bool_FN = Winner_B_SVM_matrix[1][0]
Winner_SVM_bool_TP = Winner_B_SVM_matrix[1][1]
Winner_SVM_bool_TN = Winner_B_SVM_matrix[0][0]

# Overall accuracy
Winner_SVM_bool_ACC = (Winner_SVM_bool_TP + Winner_SVM_bool_TN)/(Winner_SVM_bool_TP + Winner_SVM_bool_FP + Winner_SVM_bool_FN + Winner_SVM_bool_TN)
print(Winner_SVM_bool_ACC)

WinnerAccuracyDict.update({'Winner_SVM_bool_ACC': Winner_SVM_bool_ACC})
print(WinnerAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #2 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

Winner_B_SVM_Model2=LinearSVC(C=1)
Winner_B_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Winner_b_svm_predict2 = Winner_B_SVM_Model2.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(Winner_b_svm_predict2)
print("\nActual:")
print(TestLabelsB)

Winner_B_SVM_matrix2 = confusion_matrix(TestLabelsB, Winner_b_svm_predict2)
print("\nThe confusion matrix is:")
print(Winner_B_SVM_matrix2)
print("\n\n")

Winner_svm_B_target_names2 = ['0','1']
print(classification_report(TestLabelsB, Winner_b_svm_predict2, target_names = Winner_svm_B_target_names2))

Winner_SVM_bool2_FP = Winner_B_SVM_matrix2[0][1] 
Winner_SVM_bool2_FN = Winner_B_SVM_matrix2[1][0]
Winner_SVM_bool2_TP = Winner_B_SVM_matrix2[1][1]
Winner_SVM_bool2_TN = Winner_B_SVM_matrix2[0][0]

# Overall accuracy
Winner_SVM_bool2_ACC = (Winner_SVM_bool2_TP + Winner_SVM_bool2_TN)/(Winner_SVM_bool2_TP + Winner_SVM_bool2_FP + Winner_SVM_bool2_FN + Winner_SVM_bool2_TN)
print(Winner_SVM_bool2_ACC)

WinnerAccuracyDict.update({'Winner_SVM_bool2_ACC': Winner_SVM_bool2_ACC})
print(WinnerAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #3 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

Winner_B_SVM_Model3=LinearSVC(C=.01)
Winner_B_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Winner_b_svm_predict3 = Winner_B_SVM_Model3.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(Winner_b_svm_predict3)
print("\nActual:")
print(TestLabelsB)

Winner_B_SVM_matrix3 = confusion_matrix(TestLabelsB, Winner_b_svm_predict3)
print("\nThe confusion matrix is:")
print(Winner_B_SVM_matrix3)
print("\n\n")

Winner_svm_B_target_names3 = ['0','1']
print(classification_report(TestLabelsB, Winner_b_svm_predict3, target_names = Winner_svm_B_target_names3))

Winner_SVM_bool3_FP = Winner_B_SVM_matrix3[0][1] 
Winner_SVM_bool3_FN = Winner_B_SVM_matrix3[1][0]
Winner_SVM_bool3_TP = Winner_B_SVM_matrix3[1][1]
Winner_SVM_bool3_TN = Winner_B_SVM_matrix3[0][0]

# Overall accuracy
Winner_SVM_bool3_ACC = (Winner_SVM_bool3_TP + Winner_SVM_bool3_TN)/(Winner_SVM_bool3_TP + Winner_SVM_bool3_FP + Winner_SVM_bool3_FN + Winner_SVM_bool3_TN)
print(Winner_SVM_bool3_ACC)

WinnerAccuracyDict.update({'Winner_SVM_bool3_ACC': Winner_SVM_bool3_ACC})
print(WinnerAccuracyDict)

#############################################
###########  SVM  for tfidf model #1 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

Winner_tf_SVM_Model=LinearSVC(C=.001)
Winner_tf_SVM_Model.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Winner_tf_svm_predict = Winner_tf_SVM_Model.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(Winner_tf_svm_predict)
print("\nActual:")
print(TestLabels_tf)

Winner_tf_SVM_matrix = confusion_matrix(TestLabels_tf, Winner_tf_svm_predict)
print("\nThe confusion matrix is:")
print(Winner_tf_SVM_matrix)
print("\n\n")

Winner_svm_tf_target_names = ['0','1']
print(classification_report(TestLabels_tf, Winner_tf_svm_predict, target_names = Winner_svm_tf_target_names))

Winner_SVM_tf_FP = Winner_tf_SVM_matrix[0][1] 
Winner_SVM_tf_FN = Winner_tf_SVM_matrix[1][0]
Winner_SVM_tf_TP = Winner_tf_SVM_matrix[1][1]
Winner_SVM_tf_TN = Winner_tf_SVM_matrix[0][0]

# Overall accuracy
Winner_SVM_tf_ACC = (Winner_SVM_tf_TP + Winner_SVM_tf_TN)/(Winner_SVM_tf_TP + Winner_SVM_tf_FP + Winner_SVM_tf_FN + Winner_SVM_tf_TN)
print(Winner_SVM_tf_ACC)

WinnerAccuracyDict.update({'Winner_SVM_tf_ACC': Winner_SVM_tf_ACC})
print(WinnerAccuracyDict)

#############################################
###########  SVM  for tfidf model #2 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

Winner_tf_SVM_Model2=LinearSVC(C=1)
Winner_tf_SVM_Model2.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Winner_tf_svm_predict2 = Winner_tf_SVM_Model2.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(Winner_tf_svm_predict2)
print("\nActual:")
print(TestLabels_tf)

Winner_tf_SVM_matrix2 = confusion_matrix(TestLabels_tf, Winner_tf_svm_predict2)
print("\nThe confusion matrix is:")
print(Winner_tf_SVM_matrix2)
print("\n\n")

Winner_svm_tf_target_names2 = ['0','1']
print(classification_report(TestLabels_tf, Winner_tf_svm_predict2, target_names = Winner_svm_tf_target_names2))

Winner_SVM_tf2_FP = Winner_tf_SVM_matrix2[0][1] 
Winner_SVM_tf2_FN = Winner_tf_SVM_matrix2[1][0]
Winner_SVM_tf2_TP = Winner_tf_SVM_matrix2[1][1]
Winner_SVM_tf2_TN = Winner_tf_SVM_matrix2[0][0]

# Overall accuracy
Winner_SVM_tf2_ACC = (Winner_SVM_tf2_TP + Winner_SVM_tf2_TN)/(Winner_SVM_tf2_TP + Winner_SVM_tf2_FP + Winner_SVM_tf2_FN + Winner_SVM_tf2_TN)
print(Winner_SVM_tf2_ACC)

WinnerAccuracyDict.update({'Winner_SVM_tf2_ACC': Winner_SVM_tf2_ACC})
print(WinnerAccuracyDict)

#############################################
###########  SVM  for tfidf model #3 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

Winner_tf_SVM_Model3=LinearSVC(C=1)
Winner_tf_SVM_Model3.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Winner_tf_svm_predict3 = Winner_tf_SVM_Model3.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(Winner_tf_svm_predict3)
print("\nActual:")
print(TestLabels_tf)

Winner_tf_SVM_matrix3 = confusion_matrix(TestLabels_tf, Winner_tf_svm_predict3)
print("\nThe confusion matrix is:")
print(Winner_tf_SVM_matrix3)
print("\n\n")

Winner_svm_tf_target_names3 = ['0','1']
print(classification_report(TestLabels_tf, Winner_tf_svm_predict3, target_names = Winner_svm_tf_target_names3))

Winner_SVM_tf3_FP = Winner_tf_SVM_matrix3[0][1] 
Winner_SVM_tf3_FN = Winner_tf_SVM_matrix3[1][0]
Winner_SVM_tf3_TP = Winner_tf_SVM_matrix3[1][1]
Winner_SVM_tf3_TN = Winner_tf_SVM_matrix3[0][0]

# Overall accuracy
Winner_SVM_tf3_ACC = (Winner_SVM_tf3_TP + Winner_SVM_tf3_TN)/(Winner_SVM_tf3_TP + Winner_SVM_tf3_FP + Winner_SVM_tf3_FN + Winner_SVM_tf3_TN)
print(Winner_SVM_tf3_ACC)

WinnerAccuracyDict.update({'Winner_SVM_tf3_ACC': Winner_SVM_tf3_ACC})
print(WinnerAccuracyDict)

#############################################
#######  SVM  with different kernels ########
#############################################
## running on Bernoulli model since it had best accuracy above
## i want to see if the accuracy gets improved in the different kernels

#############################################
###############  Sigmoid #1 #################
#############################################

Winner_sig_SVM_Model=sklearn.svm.SVC(C=10, kernel='sigmoid', 
                           verbose=True, gamma="scale")
Winner_sig_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
Winner_sig_svm_predict = Winner_sig_SVM_Model.predict(TestDFB_nolabels)
print("\nsig SVM prediction:\n")
print(Winner_sig_svm_predict)
print("\nActual:")
print(TestLabelsB)

Winner_sig_SVM_matrix = confusion_matrix(TestLabelsB, Winner_sig_svm_predict)
print("\nThe sig confusion matrix is:")
print(Winner_sig_SVM_matrix)
print("\n\n")

Winner_svm_sig_target_names = ['0','1']
print(classification_report(TestLabelsB, Winner_sig_svm_predict, target_names = Winner_svm_sig_target_names))

Winner_SVM_sig_FP = Winner_sig_SVM_matrix[0][1] 
Winner_SVM_sig_FN = Winner_sig_SVM_matrix[1][0]
Winner_SVM_sig_TP = Winner_sig_SVM_matrix[1][1]
Winner_SVM_sig_TN = Winner_sig_SVM_matrix[0][0]

# Overall accuracy
Winner_SVM_sig_ACC = (Winner_SVM_sig_TP + Winner_SVM_sig_TN)/(Winner_SVM_sig_TP + Winner_SVM_sig_FP + Winner_SVM_sig_FN + Winner_SVM_sig_TN)
print(Winner_SVM_sig_ACC)

WinnerAccuracyDict.update({'Winner_SVM_sig_ACC': Winner_SVM_sig_ACC})
print(WinnerAccuracyDict)

#############################################
###############  Sigmoid #2 #################
#############################################

Winner_sig_SVM_Model2=sklearn.svm.SVC(C=1000, kernel='sigmoid', 
                           verbose=True, gamma="scale")
Winner_sig_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
Winner_sig_svm_predict2 = Winner_sig_SVM_Model2.predict(TestDFB_nolabels)
print("\nsig SVM prediction 2:\n")
print(Winner_sig_svm_predict2)
print("\nActual:")
print(TestLabelsB)

Winner_sig_SVM_matrix2 = confusion_matrix(TestLabelsB, Winner_sig_svm_predict2)
print("\nThe sig 2 confusion matrix is:")
print(Winner_sig_SVM_matrix2)
print("\n\n")

Winner_svm_sig_target_names2 = ['0','1']
print(classification_report(TestLabelsB, Winner_sig_svm_predict2, target_names = Winner_svm_sig_target_names2))

Winner_SVM_sig2_FP = Winner_sig_SVM_matrix2[0][1] 
Winner_SVM_sig2_FN = Winner_sig_SVM_matrix2[1][0]
Winner_SVM_sig2_TP = Winner_sig_SVM_matrix2[1][1]
Winner_SVM_sig2_TN = Winner_sig_SVM_matrix2[0][0]

# Overall accuracy
Winner_SVM_sig2_ACC = (Winner_SVM_sig2_TP + Winner_SVM_sig2_TN)/(Winner_SVM_sig2_TP + Winner_SVM_sig2_FP + Winner_SVM_sig2_FN + Winner_SVM_sig2_TN)
print(Winner_SVM_sig2_ACC)

WinnerAccuracyDict.update({'Winner_SVM_sig2_ACC': Winner_SVM_sig2_ACC})
print(WinnerAccuracyDict)

#############################################
###############  Sigmoid #3 #################
#############################################

Winner_sig_SVM_Model3=sklearn.svm.SVC(C=.001, kernel='sigmoid', 
                           verbose=True, gamma="scale")
Winner_sig_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
Winner_sig_svm_predict3 = Winner_sig_SVM_Model3.predict(TestDFB_nolabels)
print("\nsig SVM prediction 3:\n")
print(Winner_sig_svm_predict3)
print("\nActual:")
print(TestLabelsB)

Winner_sig_SVM_matrix3 = confusion_matrix(TestLabelsB, Winner_sig_svm_predict3)
print("\nThe sig 3 confusion matrix is:")
print(Winner_sig_SVM_matrix3)
print("\n\n")

Winner_svm_sig_target_names3 = ['0','1']
print(classification_report(TestLabelsB, Winner_sig_svm_predict3, target_names = Winner_svm_sig_target_names3))

Winner_SVM_sig3_FP = Winner_sig_SVM_matrix3[0][1] 
Winner_SVM_sig3_FN = Winner_sig_SVM_matrix3[1][0]
Winner_SVM_sig3_TP = Winner_sig_SVM_matrix3[1][1]
Winner_SVM_sig3_TN = Winner_sig_SVM_matrix3[0][0]

# Overall accuracy
Winner_SVM_sig3_ACC = (Winner_SVM_sig3_TP + Winner_SVM_sig3_TN)/(Winner_SVM_sig3_TP + Winner_SVM_sig3_FP + Winner_SVM_sig3_FN + Winner_SVM_sig3_TN)
print(Winner_SVM_sig3_ACC)

WinnerAccuracyDict.update({'Winner_SVM_sig3_ACC': Winner_SVM_sig3_ACC})
print(WinnerAccuracyDict)

#############################################
################  Poly #1 ###################
#############################################

Winner_poly_SVM_Model=sklearn.svm.SVC(C=1000, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(Winner_poly_SVM_Model)
Winner_poly_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
Winner_poly_svm_predict = Winner_poly_SVM_Model.predict(TestDFB_nolabels)
print("\npoly SVM prediction:\n")
print(Winner_poly_svm_predict)
print("\nActual:")
print(TestLabelsB)

Winner_poly_SVM_matrix = confusion_matrix(TestLabelsB, Winner_poly_svm_predict)
print("\nThe poly confusion matrix is:")
print(Winner_poly_SVM_matrix)
print("\n\n")

Winner_svm_poly_target_names = ['0','1']
print(classification_report(TestLabelsB, Winner_poly_svm_predict, target_names = Winner_svm_poly_target_names))

Winner_SVM_poly_FP = Winner_poly_SVM_matrix[0][1] 
Winner_SVM_poly_FN = Winner_poly_SVM_matrix[1][0]
Winner_SVM_poly_TP = Winner_poly_SVM_matrix[1][1]
Winner_SVM_poly_TN = Winner_poly_SVM_matrix[0][0]

# Overall accuracy
Winner_SVM_poly_ACC = (Winner_SVM_poly_TP + Winner_SVM_poly_TN)/(Winner_SVM_poly_TP + Winner_SVM_poly_FP + Winner_SVM_poly_FN + Winner_SVM_poly_TN)
print(Winner_SVM_poly_ACC)

WinnerAccuracyDict.update({'Winner_SVM_poly_ACC': Winner_SVM_poly_ACC})
print(WinnerAccuracyDict)

#############################################
################  Poly #2 ###################
#############################################

Winner_poly_SVM_Model2=sklearn.svm.SVC(C=10, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(Winner_poly_SVM_Model2)
Winner_poly_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
Winner_poly_svm_predict2 = Winner_poly_SVM_Model2.predict(TestDFB_nolabels)
print("\npoly SVM 2 prediction:\n")
print(Winner_poly_svm_predict2)
print("\nActual:")
print(TestLabelsB)

Winner_poly_SVM_matrix2 = confusion_matrix(TestLabelsB, Winner_poly_svm_predict2)
print("\nThe poly 2 confusion matrix is:")
print(Winner_poly_SVM_matrix2)
print("\n\n")

Winner_svm_poly_target_names2 = ['0','1']
print(classification_report(TestLabelsB, Winner_poly_svm_predict2, target_names = Winner_svm_poly_target_names2))

Winner_SVM_poly2_FP = Winner_poly_SVM_matrix2[0][1] 
Winner_SVM_poly2_FN = Winner_poly_SVM_matrix2[1][0]
Winner_SVM_poly2_TP = Winner_poly_SVM_matrix2[1][1]
Winner_SVM_poly2_TN = Winner_poly_SVM_matrix2[0][0]

# Overall accuracy
Winner_SVM_poly2_ACC = (Winner_SVM_poly2_TP + Winner_SVM_poly2_TN)/(Winner_SVM_poly2_TP + Winner_SVM_poly2_FP + Winner_SVM_poly2_FN + Winner_SVM_poly2_TN)
print(Winner_SVM_poly2_ACC)

WinnerAccuracyDict.update({'Winner_SVM_poly2_ACC': Winner_SVM_poly2_ACC})
print(WinnerAccuracyDict)

#############################################
################  Poly #3 ###################
#############################################

Winner_poly_SVM_Model3=sklearn.svm.SVC(C=.001, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(Winner_poly_SVM_Model3)
Winner_poly_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
Winner_poly_svm_predict3 = Winner_poly_SVM_Model3.predict(TestDFB_nolabels)
print("\npoly SVM 3 prediction:\n")
print(Winner_poly_svm_predict3)
print("\nActual:")
print(TestLabelsB)

Winner_poly_SVM_matrix3 = confusion_matrix(TestLabelsB, Winner_poly_svm_predict3)
print("\nThe poly 3 confusion matrix is:")
print(Winner_poly_SVM_matrix3)
print("\n\n")

Winner_svm_poly_target_names3 = ['0','1']
print(classification_report(TestLabelsB, Winner_poly_svm_predict3, target_names = Winner_svm_poly_target_names3))

Winner_SVM_poly3_FP = Winner_poly_SVM_matrix3[0][1] 
Winner_SVM_poly3_FN = Winner_poly_SVM_matrix3[1][0]
Winner_SVM_poly3_TP = Winner_poly_SVM_matrix3[1][1]
Winner_SVM_poly3_TN = Winner_poly_SVM_matrix3[0][0]

# Overall accuracy
Winner_SVM_poly3_ACC = (Winner_SVM_poly3_TP + Winner_SVM_poly3_TN)/(Winner_SVM_poly3_TP + Winner_SVM_poly3_FP + Winner_SVM_poly3_FN + Winner_SVM_poly3_TN)
print(Winner_SVM_poly3_ACC)

WinnerAccuracyDict.update({'Winner_SVM_poly3_ACC': Winner_SVM_poly3_ACC})
print(WinnerAccuracyDict)

WinnerVisDF = pd.DataFrame(WinnerAccuracyDict.items(), index = WinnerAccuracyDict.keys(), columns=['Model','Accuracy'])
print(WinnerVisDF)
SortedWinnerVisDF = WinnerVisDF.sort_values('Accuracy', ascending = [True])
print(SortedWinnerVisDF)

print(PartyAccuracyDict)
print(WinnerAccuracyDict)

SortedPartyVisDF.plot.barh(y='Accuracy')
SortedWinnerVisDF.plot.barh(y='Accuracy')

#print(IndexList)
#print(TextList)
#print(PartyList)
#print(WinnerList)
print(SentimentList)
print(IncumCandList)
print(IncumPartyList)
print(UnemploymentList)
print(GDPList)
print(InflationList)
print(SatisfactionList)
print(RealIncomeGrowthList)
print(PresApprovalList)
print(IncomeTaxList)
print(DjiaVolumeList)
print(CpiList)
print(CciIndexList)

#################################################################################################
#################################################################################################
### everything from this point down is a matter of find and replace for each of the variables ###
#################################################################################################
#################################################################################################

### starting to build a model for Sentiment
SentimentVectDF = VectDF.copy(deep=True)
SentimentVectDF.insert(loc=0, column='LABEL', value=SentimentList)
print(SentimentVectDF)

bool_SentimentVectDF = bool_VectDF.copy(deep=True)
bool_SentimentVectDF.insert(loc=0, column='LABEL', value=SentimentList)
print(bool_SentimentVectDF)

tf_SentimentVectDF = tf_VectDF.copy(deep=True)
tf_SentimentVectDF.insert(loc=0, column='LABEL', value=SentimentList)
print(tf_SentimentVectDF)


##--------------------------------------------------------
####################################################################
###################### model Data Build ############################
####################################################################
##--------------------------------------------------------
from sklearn.model_selection import train_test_split

## Copy everything below here to rebuild for sentiment data
TrainDF, TestDF = train_test_split(SentimentVectDF, test_size=0.3)
print(TrainDF.head())

## Now we have a training set and a testing set. 
print("\nThe training set is:")
print(TrainDF)
print("\nThe testing set is:")
print(TestDF)

## IMPORTANT - YOU CANNOT LEAVE LABELS ON THE TEST SET
## Save labels
TestLabels=TestDF["LABEL"]
print(TestLabels)
## remove labels
TestDF_nolabels = TestDF.drop(["LABEL"], axis=1)
print(TestDF_nolabels)

TrainDF_nolabels=TrainDF.drop(["LABEL"], axis=1)
print(TrainDF_nolabels)

TrainLabels=TrainDF["LABEL"]
print(TrainLabels)

############# Do the same for the binary dataframe
## Create Train/Test for TrainDFB
TrainDFB, TestDFB = train_test_split(bool_SentimentVectDF, test_size=0.3)
print(TrainDFB.head())

TrainDFB_nolabels=TrainDFB.drop(["LABEL"], axis=1)
print(TrainDFB_nolabels)
TrainLabelsB=TrainDFB["LABEL"]
print(TrainLabelsB)

TestDFB_nolabels=TestDFB.drop(["LABEL"], axis=1)
print(TestDFB_nolabels)
TestLabelsB=TestDFB["LABEL"]
print(TestLabelsB)

############# Do the same for the tfidf dataframe
## Create Train/Test for TrainDF tfidf
TrainDF_tf, TestDF_tf = train_test_split(tf_SentimentVectDF, test_size=0.3)
print(TrainDF_tf.head())

TrainDF_tf_nolabels=TrainDF_tf.drop(["LABEL"], axis=1)
print(TrainDF_tf_nolabels)
TrainLabels_tf=TrainDF_tf["LABEL"]
print(TrainLabels_tf)

TestDF_tf_nolabels=TestDF_tf.drop(["LABEL"], axis=1)
print(TestDF_tf_nolabels)
TestLabels_tf=TestDF_tf["LABEL"]
print(TestLabels_tf)



#############################################
###########  SVM  for regular model #1 ######
#############################################
from sklearn.svm import LinearSVC
print(TrainDF_nolabels.head())
print(TrainLabels.head())

Sentiment_SVM_Model=LinearSVC(C=.01)
Sentiment_SVM_Model.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Sentiment_svm_predict = Sentiment_SVM_Model.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(Sentiment_svm_predict)
print("Actual:")
print(TestLabels)

from sklearn.metrics import confusion_matrix
Sentiment_SVM_matrix = confusion_matrix(TestLabels, Sentiment_svm_predict)
print("\nThe confusion matrix is:")
print(Sentiment_SVM_matrix)
print("\n\n")

from sklearn.metrics import classification_report
Sentiment_svm_target_names = ['0','1']
print(classification_report(TestLabels, Sentiment_svm_predict, target_names = Sentiment_svm_target_names))

Sentiment_SVM_reg_FP = Sentiment_SVM_matrix[0][1] 
Sentiment_SVM_reg_FN = Sentiment_SVM_matrix[1][0]
Sentiment_SVM_reg_TP = Sentiment_SVM_matrix[1][1]
Sentiment_SVM_reg_TN = Sentiment_SVM_matrix[0][0]

# Overall accuracy
Sentiment_SVM_reg_ACC = (Sentiment_SVM_reg_TP + Sentiment_SVM_reg_TN)/(Sentiment_SVM_reg_TP + Sentiment_SVM_reg_FP + Sentiment_SVM_reg_FN + Sentiment_SVM_reg_TN)
print(Sentiment_SVM_reg_ACC)

SentimentAccuracyDict = {}
SentimentAccuracyDict.update({'Sentiment_SVM_reg_ACC': Sentiment_SVM_reg_ACC})
print(SentimentAccuracyDict)

#############################################
###########  SVM  for regular model #2 ######
#############################################

Sentiment_SVM_Model2=LinearSVC(C=1)
Sentiment_SVM_Model2.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Sentiment_svm_predict2 = Sentiment_SVM_Model2.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(Sentiment_svm_predict2)
print("Actual:")
print(TestLabels)

Sentiment_SVM_matrix2 = confusion_matrix(TestLabels, Sentiment_svm_predict2)
print("\nThe confusion matrix is:")
print(Sentiment_SVM_matrix2)
print("\n\n")

Sentiment_svm_target_names2 = ['0','1']
print(classification_report(TestLabels, Sentiment_svm_predict2, target_names = Sentiment_svm_target_names2))

Sentiment_SVM_reg2_FP = Sentiment_SVM_matrix2[0][1] 
Sentiment_SVM_reg2_FN = Sentiment_SVM_matrix2[1][0]
Sentiment_SVM_reg2_TP = Sentiment_SVM_matrix2[1][1]
Sentiment_SVM_reg2_TN = Sentiment_SVM_matrix2[0][0]

# Overall accuracy
Sentiment_SVM_reg2_ACC = (Sentiment_SVM_reg2_TP + Sentiment_SVM_reg2_TN)/(Sentiment_SVM_reg2_TP + Sentiment_SVM_reg2_FP + Sentiment_SVM_reg2_FN + Sentiment_SVM_reg2_TN)
print(Sentiment_SVM_reg2_ACC)

SentimentAccuracyDict.update({'Sentiment_SVM_reg2_ACC': Sentiment_SVM_reg2_ACC})
print(SentimentAccuracyDict)

#############################################
###########  SVM  for regular model #3 ######
#############################################

Sentiment_SVM_Model3=LinearSVC(C=100)
Sentiment_SVM_Model3.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Sentiment_svm_predict3 = Sentiment_SVM_Model3.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(Sentiment_svm_predict3)
print("Actual:")
print(TestLabels)

Sentiment_SVM_matrix3 = confusion_matrix(TestLabels, Sentiment_svm_predict3)
print("\nThe confusion matrix is:")
print(Sentiment_SVM_matrix3)
print("\n\n")

from sklearn.metrics import classification_report
Sentiment_svm_target_names3 = ['0','1']
print(classification_report(TestLabels, Sentiment_svm_predict3, target_names = Sentiment_svm_target_names3))

Sentiment_SVM_reg3_FP = Sentiment_SVM_matrix3[0][1] 
Sentiment_SVM_reg3_FN = Sentiment_SVM_matrix3[1][0]
Sentiment_SVM_reg3_TP = Sentiment_SVM_matrix3[1][1]
Sentiment_SVM_reg3_TN = Sentiment_SVM_matrix3[0][0]

# Overall accuracy
Sentiment_SVM_reg3_ACC = (Sentiment_SVM_reg3_TP + Sentiment_SVM_reg3_TN)/(Sentiment_SVM_reg3_TP + Sentiment_SVM_reg3_FP + Sentiment_SVM_reg3_FN + Sentiment_SVM_reg3_TN)
print(Sentiment_SVM_reg3_ACC)

SentimentAccuracyDict.update({'Sentiment_SVM_reg3_ACC': Sentiment_SVM_reg3_ACC})
print(SentimentAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #1 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

Sentiment_B_SVM_Model=LinearSVC(C=100)
Sentiment_B_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Sentiment_b_svm_predict = Sentiment_B_SVM_Model.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(Sentiment_b_svm_predict)
print("\nActual:")
print(TestLabelsB)

Sentiment_B_SVM_matrix = confusion_matrix(TestLabelsB, Sentiment_b_svm_predict)
print("\nThe confusion matrix is:")
print(Sentiment_B_SVM_matrix)
print("\n\n")

Sentiment_svm_B_target_names = ['0','1']
print(classification_report(TestLabelsB, Sentiment_b_svm_predict, target_names = Sentiment_svm_B_target_names))

Sentiment_SVM_bool_FP = Sentiment_B_SVM_matrix[0][1] 
Sentiment_SVM_bool_FN = Sentiment_B_SVM_matrix[1][0]
Sentiment_SVM_bool_TP = Sentiment_B_SVM_matrix[1][1]
Sentiment_SVM_bool_TN = Sentiment_B_SVM_matrix[0][0]

# Overall accuracy
Sentiment_SVM_bool_ACC = (Sentiment_SVM_bool_TP + Sentiment_SVM_bool_TN)/(Sentiment_SVM_bool_TP + Sentiment_SVM_bool_FP + Sentiment_SVM_bool_FN + Sentiment_SVM_bool_TN)
print(Sentiment_SVM_bool_ACC)

SentimentAccuracyDict.update({'Sentiment_SVM_bool_ACC': Sentiment_SVM_bool_ACC})
print(SentimentAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #2 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

Sentiment_B_SVM_Model2=LinearSVC(C=1)
Sentiment_B_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Sentiment_b_svm_predict2 = Sentiment_B_SVM_Model2.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(Sentiment_b_svm_predict2)
print("\nActual:")
print(TestLabelsB)

Sentiment_B_SVM_matrix2 = confusion_matrix(TestLabelsB, Sentiment_b_svm_predict2)
print("\nThe confusion matrix is:")
print(Sentiment_B_SVM_matrix2)
print("\n\n")

Sentiment_svm_B_target_names2 = ['0','1']
print(classification_report(TestLabelsB, Sentiment_b_svm_predict2, target_names = Sentiment_svm_B_target_names2))

Sentiment_SVM_bool2_FP = Sentiment_B_SVM_matrix2[0][1] 
Sentiment_SVM_bool2_FN = Sentiment_B_SVM_matrix2[1][0]
Sentiment_SVM_bool2_TP = Sentiment_B_SVM_matrix2[1][1]
Sentiment_SVM_bool2_TN = Sentiment_B_SVM_matrix2[0][0]

# Overall accuracy
Sentiment_SVM_bool2_ACC = (Sentiment_SVM_bool2_TP + Sentiment_SVM_bool2_TN)/(Sentiment_SVM_bool2_TP + Sentiment_SVM_bool2_FP + Sentiment_SVM_bool2_FN + Sentiment_SVM_bool2_TN)
print(Sentiment_SVM_bool2_ACC)

SentimentAccuracyDict.update({'Sentiment_SVM_bool2_ACC': Sentiment_SVM_bool2_ACC})
print(SentimentAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #3 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

Sentiment_B_SVM_Model3=LinearSVC(C=.01)
Sentiment_B_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Sentiment_b_svm_predict3 = Sentiment_B_SVM_Model3.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(Sentiment_b_svm_predict3)
print("\nActual:")
print(TestLabelsB)

Sentiment_B_SVM_matrix3 = confusion_matrix(TestLabelsB, Sentiment_b_svm_predict3)
print("\nThe confusion matrix is:")
print(Sentiment_B_SVM_matrix3)
print("\n\n")

Sentiment_svm_B_target_names3 = ['0','1']
print(classification_report(TestLabelsB, Sentiment_b_svm_predict3, target_names = Sentiment_svm_B_target_names3))

Sentiment_SVM_bool3_FP = Sentiment_B_SVM_matrix3[0][1] 
Sentiment_SVM_bool3_FN = Sentiment_B_SVM_matrix3[1][0]
Sentiment_SVM_bool3_TP = Sentiment_B_SVM_matrix3[1][1]
Sentiment_SVM_bool3_TN = Sentiment_B_SVM_matrix3[0][0]

# Overall accuracy
Sentiment_SVM_bool3_ACC = (Sentiment_SVM_bool3_TP + Sentiment_SVM_bool3_TN)/(Sentiment_SVM_bool3_TP + Sentiment_SVM_bool3_FP + Sentiment_SVM_bool3_FN + Sentiment_SVM_bool3_TN)
print(Sentiment_SVM_bool3_ACC)

SentimentAccuracyDict.update({'Sentiment_SVM_bool3_ACC': Sentiment_SVM_bool3_ACC})
print(SentimentAccuracyDict)

#############################################
###########  SVM  for tfidf model #1 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

Sentiment_tf_SVM_Model=LinearSVC(C=.001)
Sentiment_tf_SVM_Model.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Sentiment_tf_svm_predict = Sentiment_tf_SVM_Model.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(Sentiment_tf_svm_predict)
print("\nActual:")
print(TestLabels_tf)

Sentiment_tf_SVM_matrix = confusion_matrix(TestLabels_tf, Sentiment_tf_svm_predict)
print("\nThe confusion matrix is:")
print(Sentiment_tf_SVM_matrix)
print("\n\n")

Sentiment_svm_tf_target_names = ['0','1']
print(classification_report(TestLabels_tf, Sentiment_tf_svm_predict, target_names = Sentiment_svm_tf_target_names))

Sentiment_SVM_tf_FP = Sentiment_tf_SVM_matrix[0][1] 
Sentiment_SVM_tf_FN = Sentiment_tf_SVM_matrix[1][0]
Sentiment_SVM_tf_TP = Sentiment_tf_SVM_matrix[1][1]
Sentiment_SVM_tf_TN = Sentiment_tf_SVM_matrix[0][0]

# Overall accuracy
Sentiment_SVM_tf_ACC = (Sentiment_SVM_tf_TP + Sentiment_SVM_tf_TN)/(Sentiment_SVM_tf_TP + Sentiment_SVM_tf_FP + Sentiment_SVM_tf_FN + Sentiment_SVM_tf_TN)
print(Sentiment_SVM_tf_ACC)

SentimentAccuracyDict.update({'Sentiment_SVM_tf_ACC': Sentiment_SVM_tf_ACC})
print(SentimentAccuracyDict)

#############################################
###########  SVM  for tfidf model #2 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

Sentiment_tf_SVM_Model2=LinearSVC(C=1)
Sentiment_tf_SVM_Model2.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Sentiment_tf_svm_predict2 = Sentiment_tf_SVM_Model2.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(Sentiment_tf_svm_predict2)
print("\nActual:")
print(TestLabels_tf)

Sentiment_tf_SVM_matrix2 = confusion_matrix(TestLabels_tf, Sentiment_tf_svm_predict2)
print("\nThe confusion matrix is:")
print(Sentiment_tf_SVM_matrix2)
print("\n\n")

Sentiment_svm_tf_target_names2 = ['0','1']
print(classification_report(TestLabels_tf, Sentiment_tf_svm_predict2, target_names = Sentiment_svm_tf_target_names2))

Sentiment_SVM_tf2_FP = Sentiment_tf_SVM_matrix2[0][1] 
Sentiment_SVM_tf2_FN = Sentiment_tf_SVM_matrix2[1][0]
Sentiment_SVM_tf2_TP = Sentiment_tf_SVM_matrix2[1][1]
Sentiment_SVM_tf2_TN = Sentiment_tf_SVM_matrix2[0][0]

# Overall accuracy
Sentiment_SVM_tf2_ACC = (Sentiment_SVM_tf2_TP + Sentiment_SVM_tf2_TN)/(Sentiment_SVM_tf2_TP + Sentiment_SVM_tf2_FP + Sentiment_SVM_tf2_FN + Sentiment_SVM_tf2_TN)
print(Sentiment_SVM_tf2_ACC)

SentimentAccuracyDict.update({'Sentiment_SVM_tf2_ACC': Sentiment_SVM_tf2_ACC})
print(SentimentAccuracyDict)

#############################################
###########  SVM  for tfidf model #3 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

Sentiment_tf_SVM_Model3=LinearSVC(C=1)
Sentiment_tf_SVM_Model3.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Sentiment_tf_svm_predict3 = Sentiment_tf_SVM_Model3.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(Sentiment_tf_svm_predict3)
print("\nActual:")
print(TestLabels_tf)

Sentiment_tf_SVM_matrix3 = confusion_matrix(TestLabels_tf, Sentiment_tf_svm_predict3)
print("\nThe confusion matrix is:")
print(Sentiment_tf_SVM_matrix3)
print("\n\n")

Sentiment_svm_tf_target_names3 = ['0','1']
print(classification_report(TestLabels_tf, Sentiment_tf_svm_predict3, target_names = Sentiment_svm_tf_target_names3))

Sentiment_SVM_tf3_FP = Sentiment_tf_SVM_matrix3[0][1] 
Sentiment_SVM_tf3_FN = Sentiment_tf_SVM_matrix3[1][0]
Sentiment_SVM_tf3_TP = Sentiment_tf_SVM_matrix3[1][1]
Sentiment_SVM_tf3_TN = Sentiment_tf_SVM_matrix3[0][0]

# Overall accuracy
Sentiment_SVM_tf3_ACC = (Sentiment_SVM_tf3_TP + Sentiment_SVM_tf3_TN)/(Sentiment_SVM_tf3_TP + Sentiment_SVM_tf3_FP + Sentiment_SVM_tf3_FN + Sentiment_SVM_tf3_TN)
print(Sentiment_SVM_tf3_ACC)

SentimentAccuracyDict.update({'Sentiment_SVM_tf3_ACC': Sentiment_SVM_tf3_ACC})
print(SentimentAccuracyDict)

#############################################
#######  SVM  with different kernels ########
#############################################
## running on Bernoulli model since it had best accuracy above
## i want to see if the accuracy gets improved in the different kernels

#############################################
###############  Sigmoid #1 #################
#############################################

Sentiment_sig_SVM_Model=sklearn.svm.SVC(C=10, kernel='sigmoid', 
                           verbose=True, gamma="scale")
Sentiment_sig_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
Sentiment_sig_svm_predict = Sentiment_sig_SVM_Model.predict(TestDFB_nolabels)
print("\nsig SVM prediction:\n")
print(Sentiment_sig_svm_predict)
print("\nActual:")
print(TestLabelsB)

Sentiment_sig_SVM_matrix = confusion_matrix(TestLabelsB, Sentiment_sig_svm_predict)
print("\nThe sig confusion matrix is:")
print(Sentiment_sig_SVM_matrix)
print("\n\n")

Sentiment_svm_sig_target_names = ['0','1']
print(classification_report(TestLabelsB, Sentiment_sig_svm_predict, target_names = Sentiment_svm_sig_target_names))

Sentiment_SVM_sig_FP = Sentiment_sig_SVM_matrix[0][1] 
Sentiment_SVM_sig_FN = Sentiment_sig_SVM_matrix[1][0]
Sentiment_SVM_sig_TP = Sentiment_sig_SVM_matrix[1][1]
Sentiment_SVM_sig_TN = Sentiment_sig_SVM_matrix[0][0]

# Overall accuracy
Sentiment_SVM_sig_ACC = (Sentiment_SVM_sig_TP + Sentiment_SVM_sig_TN)/(Sentiment_SVM_sig_TP + Sentiment_SVM_sig_FP + Sentiment_SVM_sig_FN + Sentiment_SVM_sig_TN)
print(Sentiment_SVM_sig_ACC)

SentimentAccuracyDict.update({'Sentiment_SVM_sig_ACC': Sentiment_SVM_sig_ACC})
print(SentimentAccuracyDict)

#############################################
###############  Sigmoid #2 #################
#############################################

Sentiment_sig_SVM_Model2=sklearn.svm.SVC(C=1000, kernel='sigmoid', 
                           verbose=True, gamma="scale")
Sentiment_sig_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
Sentiment_sig_svm_predict2 = Sentiment_sig_SVM_Model2.predict(TestDFB_nolabels)
print("\nsig SVM prediction 2:\n")
print(Sentiment_sig_svm_predict2)
print("\nActual:")
print(TestLabelsB)

Sentiment_sig_SVM_matrix2 = confusion_matrix(TestLabelsB, Sentiment_sig_svm_predict2)
print("\nThe sig 2 confusion matrix is:")
print(Sentiment_sig_SVM_matrix2)
print("\n\n")

Sentiment_svm_sig_target_names2 = ['0','1']
print(classification_report(TestLabelsB, Sentiment_sig_svm_predict2, target_names = Sentiment_svm_sig_target_names2))

Sentiment_SVM_sig2_FP = Sentiment_sig_SVM_matrix2[0][1] 
Sentiment_SVM_sig2_FN = Sentiment_sig_SVM_matrix2[1][0]
Sentiment_SVM_sig2_TP = Sentiment_sig_SVM_matrix2[1][1]
Sentiment_SVM_sig2_TN = Sentiment_sig_SVM_matrix2[0][0]

# Overall accuracy
Sentiment_SVM_sig2_ACC = (Sentiment_SVM_sig2_TP + Sentiment_SVM_sig2_TN)/(Sentiment_SVM_sig2_TP + Sentiment_SVM_sig2_FP + Sentiment_SVM_sig2_FN + Sentiment_SVM_sig2_TN)
print(Sentiment_SVM_sig2_ACC)

SentimentAccuracyDict.update({'Sentiment_SVM_sig2_ACC': Sentiment_SVM_sig2_ACC})
print(SentimentAccuracyDict)

#############################################
###############  Sigmoid #3 #################
#############################################

Sentiment_sig_SVM_Model3=sklearn.svm.SVC(C=.001, kernel='sigmoid', 
                           verbose=True, gamma="scale")
Sentiment_sig_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
Sentiment_sig_svm_predict3 = Sentiment_sig_SVM_Model3.predict(TestDFB_nolabels)
print("\nsig SVM prediction 3:\n")
print(Sentiment_sig_svm_predict3)
print("\nActual:")
print(TestLabelsB)

Sentiment_sig_SVM_matrix3 = confusion_matrix(TestLabelsB, Sentiment_sig_svm_predict3)
print("\nThe sig 3 confusion matrix is:")
print(Sentiment_sig_SVM_matrix3)
print("\n\n")

Sentiment_svm_sig_target_names3 = ['0','1']
print(classification_report(TestLabelsB, Sentiment_sig_svm_predict3, target_names = Sentiment_svm_sig_target_names3))

Sentiment_SVM_sig3_FP = Sentiment_sig_SVM_matrix3[0][1] 
Sentiment_SVM_sig3_FN = Sentiment_sig_SVM_matrix3[1][0]
Sentiment_SVM_sig3_TP = Sentiment_sig_SVM_matrix3[1][1]
Sentiment_SVM_sig3_TN = Sentiment_sig_SVM_matrix3[0][0]

# Overall accuracy
Sentiment_SVM_sig3_ACC = (Sentiment_SVM_sig3_TP + Sentiment_SVM_sig3_TN)/(Sentiment_SVM_sig3_TP + Sentiment_SVM_sig3_FP + Sentiment_SVM_sig3_FN + Sentiment_SVM_sig3_TN)
print(Sentiment_SVM_sig3_ACC)

SentimentAccuracyDict.update({'Sentiment_SVM_sig3_ACC': Sentiment_SVM_sig3_ACC})
print(SentimentAccuracyDict)

#############################################
################  Poly #1 ###################
#############################################

Sentiment_poly_SVM_Model=sklearn.svm.SVC(C=1000, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(Sentiment_poly_SVM_Model)
Sentiment_poly_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
Sentiment_poly_svm_predict = Sentiment_poly_SVM_Model.predict(TestDFB_nolabels)
print("\npoly SVM prediction:\n")
print(Sentiment_poly_svm_predict)
print("\nActual:")
print(TestLabelsB)

Sentiment_poly_SVM_matrix = confusion_matrix(TestLabelsB, Sentiment_poly_svm_predict)
print("\nThe poly confusion matrix is:")
print(Sentiment_poly_SVM_matrix)
print("\n\n")

Sentiment_svm_poly_target_names = ['0','1']
print(classification_report(TestLabelsB, Sentiment_poly_svm_predict, target_names = Sentiment_svm_poly_target_names))

Sentiment_SVM_poly_FP = Sentiment_poly_SVM_matrix[0][1] 
Sentiment_SVM_poly_FN = Sentiment_poly_SVM_matrix[1][0]
Sentiment_SVM_poly_TP = Sentiment_poly_SVM_matrix[1][1]
Sentiment_SVM_poly_TN = Sentiment_poly_SVM_matrix[0][0]

# Overall accuracy
Sentiment_SVM_poly_ACC = (Sentiment_SVM_poly_TP + Sentiment_SVM_poly_TN)/(Sentiment_SVM_poly_TP + Sentiment_SVM_poly_FP + Sentiment_SVM_poly_FN + Sentiment_SVM_poly_TN)
print(Sentiment_SVM_poly_ACC)

SentimentAccuracyDict.update({'Sentiment_SVM_poly_ACC': Sentiment_SVM_poly_ACC})
print(SentimentAccuracyDict)

#############################################
################  Poly #2 ###################
#############################################

Sentiment_poly_SVM_Model2=sklearn.svm.SVC(C=10, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(Sentiment_poly_SVM_Model2)
Sentiment_poly_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
Sentiment_poly_svm_predict2 = Sentiment_poly_SVM_Model2.predict(TestDFB_nolabels)
print("\npoly SVM 2 prediction:\n")
print(Sentiment_poly_svm_predict2)
print("\nActual:")
print(TestLabelsB)

Sentiment_poly_SVM_matrix2 = confusion_matrix(TestLabelsB, Sentiment_poly_svm_predict2)
print("\nThe poly 2 confusion matrix is:")
print(Sentiment_poly_SVM_matrix2)
print("\n\n")

Sentiment_svm_poly_target_names2 = ['0','1']
print(classification_report(TestLabelsB, Sentiment_poly_svm_predict2, target_names = Sentiment_svm_poly_target_names2))

Sentiment_SVM_poly2_FP = Sentiment_poly_SVM_matrix2[0][1] 
Sentiment_SVM_poly2_FN = Sentiment_poly_SVM_matrix2[1][0]
Sentiment_SVM_poly2_TP = Sentiment_poly_SVM_matrix2[1][1]
Sentiment_SVM_poly2_TN = Sentiment_poly_SVM_matrix2[0][0]

# Overall accuracy
Sentiment_SVM_poly2_ACC = (Sentiment_SVM_poly2_TP + Sentiment_SVM_poly2_TN)/(Sentiment_SVM_poly2_TP + Sentiment_SVM_poly2_FP + Sentiment_SVM_poly2_FN + Sentiment_SVM_poly2_TN)
print(Sentiment_SVM_poly2_ACC)

SentimentAccuracyDict.update({'Sentiment_SVM_poly2_ACC': Sentiment_SVM_poly2_ACC})
print(SentimentAccuracyDict)

#############################################
################  Poly #3 ###################
#############################################

Sentiment_poly_SVM_Model3=sklearn.svm.SVC(C=.001, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(Sentiment_poly_SVM_Model3)
Sentiment_poly_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
Sentiment_poly_svm_predict3 = Sentiment_poly_SVM_Model3.predict(TestDFB_nolabels)
print("\npoly SVM 3 prediction:\n")
print(Sentiment_poly_svm_predict3)
print("\nActual:")
print(TestLabelsB)

Sentiment_poly_SVM_matrix3 = confusion_matrix(TestLabelsB, Sentiment_poly_svm_predict3)
print("\nThe poly 3 confusion matrix is:")
print(Sentiment_poly_SVM_matrix3)
print("\n\n")

Sentiment_svm_poly_target_names3 = ['0','1']
print(classification_report(TestLabelsB, Sentiment_poly_svm_predict3, target_names = Sentiment_svm_poly_target_names3))

Sentiment_SVM_poly3_FP = Sentiment_poly_SVM_matrix3[0][1] 
Sentiment_SVM_poly3_FN = Sentiment_poly_SVM_matrix3[1][0]
Sentiment_SVM_poly3_TP = Sentiment_poly_SVM_matrix3[1][1]
Sentiment_SVM_poly3_TN = Sentiment_poly_SVM_matrix3[0][0]

# Overall accuracy
Sentiment_SVM_poly3_ACC = (Sentiment_SVM_poly3_TP + Sentiment_SVM_poly3_TN)/(Sentiment_SVM_poly3_TP + Sentiment_SVM_poly3_FP + Sentiment_SVM_poly3_FN + Sentiment_SVM_poly3_TN)
print(Sentiment_SVM_poly3_ACC)

SentimentAccuracyDict.update({'Sentiment_SVM_poly3_ACC': Sentiment_SVM_poly3_ACC})
print(SentimentAccuracyDict)

SentimentVisDF = pd.DataFrame(SentimentAccuracyDict.items(), index = SentimentAccuracyDict.keys(), columns=['Model','Accuracy'])
print(SentimentVisDF)
SortedSentimentVisDF = SentimentVisDF.sort_values('Accuracy', ascending = [True])
print(SortedSentimentVisDF)

print(PartyAccuracyDict)
print(WinnerAccuracyDict)
print(SentimentAccuracyDict)

SortedPartyVisDF.plot.barh(y='Accuracy')
SortedWinnerVisDF.plot.barh(y='Accuracy')
SortedSentimentVisDF.plot.barh(y='Accuracy')

#print(IndexList)
#print(TextList)
#print(PartyList)
#print(WinnerList)
#print(SentimentList)
print(IncumCandList)
print(IncumPartyList)
print(UnemploymentList)
print(GDPList)
print(InflationList)
print(SatisfactionList)
print(RealIncomeGrowthList)
print(PresApprovalList)
print(IncomeTaxList)
print(DjiaVolumeList)
print(CpiList)
print(CciIndexList)

#################################################################################################
#################################################################################################
### everything from this point down is a matter of find and replace for each of the variables ###
#################################################################################################
#################################################################################################

### starting to build a model for IncumCand
IncumCandVectDF = VectDF.copy(deep=True)
IncumCandVectDF.insert(loc=0, column='LABEL', value=IncumCandList)
print(IncumCandVectDF)

bool_IncumCandVectDF = bool_VectDF.copy(deep=True)
bool_IncumCandVectDF.insert(loc=0, column='LABEL', value=IncumCandList)
print(bool_IncumCandVectDF)

tf_IncumCandVectDF = tf_VectDF.copy(deep=True)
tf_IncumCandVectDF.insert(loc=0, column='LABEL', value=IncumCandList)
print(tf_IncumCandVectDF)


##--------------------------------------------------------
####################################################################
###################### model Data Build ############################
####################################################################
##--------------------------------------------------------
from sklearn.model_selection import train_test_split

## Copy everything below here to rebuild for IncumCand data
TrainDF, TestDF = train_test_split(IncumCandVectDF, test_size=0.3)
print(TrainDF.head())

## Now we have a training set and a testing set. 
print("\nThe training set is:")
print(TrainDF)
print("\nThe testing set is:")
print(TestDF)

## IMPORTANT - YOU CANNOT LEAVE LABELS ON THE TEST SET
## Save labels
TestLabels=TestDF["LABEL"]
print(TestLabels)
## remove labels
TestDF_nolabels = TestDF.drop(["LABEL"], axis=1)
print(TestDF_nolabels)

TrainDF_nolabels=TrainDF.drop(["LABEL"], axis=1)
print(TrainDF_nolabels)

TrainLabels=TrainDF["LABEL"]
print(TrainLabels)

############# Do the same for the binary dataframe
## Create Train/Test for TrainDFB
TrainDFB, TestDFB = train_test_split(bool_IncumCandVectDF, test_size=0.3)
print(TrainDFB.head())

TrainDFB_nolabels=TrainDFB.drop(["LABEL"], axis=1)
print(TrainDFB_nolabels)
TrainLabelsB=TrainDFB["LABEL"]
print(TrainLabelsB)

TestDFB_nolabels=TestDFB.drop(["LABEL"], axis=1)
print(TestDFB_nolabels)
TestLabelsB=TestDFB["LABEL"]
print(TestLabelsB)

############# Do the same for the tfidf dataframe
## Create Train/Test for TrainDF tfidf
TrainDF_tf, TestDF_tf = train_test_split(tf_IncumCandVectDF, test_size=0.3)
print(TrainDF_tf.head())

TrainDF_tf_nolabels=TrainDF_tf.drop(["LABEL"], axis=1)
print(TrainDF_tf_nolabels)
TrainLabels_tf=TrainDF_tf["LABEL"]
print(TrainLabels_tf)

TestDF_tf_nolabels=TestDF_tf.drop(["LABEL"], axis=1)
print(TestDF_tf_nolabels)
TestLabels_tf=TestDF_tf["LABEL"]
print(TestLabels_tf)



#############################################
###########  SVM  for regular model #1 ######
#############################################
from sklearn.svm import LinearSVC
print(TrainDF_nolabels.head())
print(TrainLabels.head())

IncumCand_SVM_Model=LinearSVC(C=.01)
IncumCand_SVM_Model.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
IncumCand_svm_predict = IncumCand_SVM_Model.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(IncumCand_svm_predict)
print("Actual:")
print(TestLabels)

from sklearn.metrics import confusion_matrix
IncumCand_SVM_matrix = confusion_matrix(TestLabels, IncumCand_svm_predict)
print("\nThe confusion matrix is:")
print(IncumCand_SVM_matrix)
print("\n\n")

from sklearn.metrics import classification_report
IncumCand_svm_target_names = ['0','1']
print(classification_report(TestLabels, IncumCand_svm_predict, target_names = IncumCand_svm_target_names))

IncumCand_SVM_reg_FP = IncumCand_SVM_matrix[0][1] 
IncumCand_SVM_reg_FN = IncumCand_SVM_matrix[1][0]
IncumCand_SVM_reg_TP = IncumCand_SVM_matrix[1][1]
IncumCand_SVM_reg_TN = IncumCand_SVM_matrix[0][0]

# Overall accuracy
IncumCand_SVM_reg_ACC = (IncumCand_SVM_reg_TP + IncumCand_SVM_reg_TN)/(IncumCand_SVM_reg_TP + IncumCand_SVM_reg_FP + IncumCand_SVM_reg_FN + IncumCand_SVM_reg_TN)
print(IncumCand_SVM_reg_ACC)

IncumCandAccuracyDict = {}
IncumCandAccuracyDict.update({'IncumCand_SVM_reg_ACC': IncumCand_SVM_reg_ACC})
print(IncumCandAccuracyDict)

#############################################
###########  SVM  for regular model #2 ######
#############################################

IncumCand_SVM_Model2=LinearSVC(C=1)
IncumCand_SVM_Model2.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
IncumCand_svm_predict2 = IncumCand_SVM_Model2.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(IncumCand_svm_predict2)
print("Actual:")
print(TestLabels)

IncumCand_SVM_matrix2 = confusion_matrix(TestLabels, IncumCand_svm_predict2)
print("\nThe confusion matrix is:")
print(IncumCand_SVM_matrix2)
print("\n\n")

IncumCand_svm_target_names2 = ['0','1']
print(classification_report(TestLabels, IncumCand_svm_predict2, target_names = IncumCand_svm_target_names2))

IncumCand_SVM_reg2_FP = IncumCand_SVM_matrix2[0][1] 
IncumCand_SVM_reg2_FN = IncumCand_SVM_matrix2[1][0]
IncumCand_SVM_reg2_TP = IncumCand_SVM_matrix2[1][1]
IncumCand_SVM_reg2_TN = IncumCand_SVM_matrix2[0][0]

# Overall accuracy
IncumCand_SVM_reg2_ACC = (IncumCand_SVM_reg2_TP + IncumCand_SVM_reg2_TN)/(IncumCand_SVM_reg2_TP + IncumCand_SVM_reg2_FP + IncumCand_SVM_reg2_FN + IncumCand_SVM_reg2_TN)
print(IncumCand_SVM_reg2_ACC)

IncumCandAccuracyDict.update({'IncumCand_SVM_reg2_ACC': IncumCand_SVM_reg2_ACC})
print(IncumCandAccuracyDict)

#############################################
###########  SVM  for regular model #3 ######
#############################################

IncumCand_SVM_Model3=LinearSVC(C=100)
IncumCand_SVM_Model3.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
IncumCand_svm_predict3 = IncumCand_SVM_Model3.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(IncumCand_svm_predict3)
print("Actual:")
print(TestLabels)

IncumCand_SVM_matrix3 = confusion_matrix(TestLabels, IncumCand_svm_predict3)
print("\nThe confusion matrix is:")
print(IncumCand_SVM_matrix3)
print("\n\n")

from sklearn.metrics import classification_report
IncumCand_svm_target_names3 = ['0','1']
print(classification_report(TestLabels, IncumCand_svm_predict3, target_names = IncumCand_svm_target_names3))

IncumCand_SVM_reg3_FP = IncumCand_SVM_matrix3[0][1] 
IncumCand_SVM_reg3_FN = IncumCand_SVM_matrix3[1][0]
IncumCand_SVM_reg3_TP = IncumCand_SVM_matrix3[1][1]
IncumCand_SVM_reg3_TN = IncumCand_SVM_matrix3[0][0]

# Overall accuracy
IncumCand_SVM_reg3_ACC = (IncumCand_SVM_reg3_TP + IncumCand_SVM_reg3_TN)/(IncumCand_SVM_reg3_TP + IncumCand_SVM_reg3_FP + IncumCand_SVM_reg3_FN + IncumCand_SVM_reg3_TN)
print(IncumCand_SVM_reg3_ACC)

IncumCandAccuracyDict.update({'IncumCand_SVM_reg3_ACC': IncumCand_SVM_reg3_ACC})
print(IncumCandAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #1 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

IncumCand_B_SVM_Model=LinearSVC(C=100)
IncumCand_B_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
IncumCand_b_svm_predict = IncumCand_B_SVM_Model.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(IncumCand_b_svm_predict)
print("\nActual:")
print(TestLabelsB)

IncumCand_B_SVM_matrix = confusion_matrix(TestLabelsB, IncumCand_b_svm_predict)
print("\nThe confusion matrix is:")
print(IncumCand_B_SVM_matrix)
print("\n\n")

IncumCand_svm_B_target_names = ['0','1']
print(classification_report(TestLabelsB, IncumCand_b_svm_predict, target_names = IncumCand_svm_B_target_names))

IncumCand_SVM_bool_FP = IncumCand_B_SVM_matrix[0][1] 
IncumCand_SVM_bool_FN = IncumCand_B_SVM_matrix[1][0]
IncumCand_SVM_bool_TP = IncumCand_B_SVM_matrix[1][1]
IncumCand_SVM_bool_TN = IncumCand_B_SVM_matrix[0][0]

# Overall accuracy
IncumCand_SVM_bool_ACC = (IncumCand_SVM_bool_TP + IncumCand_SVM_bool_TN)/(IncumCand_SVM_bool_TP + IncumCand_SVM_bool_FP + IncumCand_SVM_bool_FN + IncumCand_SVM_bool_TN)
print(IncumCand_SVM_bool_ACC)

IncumCandAccuracyDict.update({'IncumCand_SVM_bool_ACC': IncumCand_SVM_bool_ACC})
print(IncumCandAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #2 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

IncumCand_B_SVM_Model2=LinearSVC(C=1)
IncumCand_B_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
IncumCand_b_svm_predict2 = IncumCand_B_SVM_Model2.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(IncumCand_b_svm_predict2)
print("\nActual:")
print(TestLabelsB)

IncumCand_B_SVM_matrix2 = confusion_matrix(TestLabelsB, IncumCand_b_svm_predict2)
print("\nThe confusion matrix is:")
print(IncumCand_B_SVM_matrix2)
print("\n\n")

IncumCand_svm_B_target_names2 = ['0','1']
print(classification_report(TestLabelsB, IncumCand_b_svm_predict2, target_names = IncumCand_svm_B_target_names2))

IncumCand_SVM_bool2_FP = IncumCand_B_SVM_matrix2[0][1] 
IncumCand_SVM_bool2_FN = IncumCand_B_SVM_matrix2[1][0]
IncumCand_SVM_bool2_TP = IncumCand_B_SVM_matrix2[1][1]
IncumCand_SVM_bool2_TN = IncumCand_B_SVM_matrix2[0][0]

# Overall accuracy
IncumCand_SVM_bool2_ACC = (IncumCand_SVM_bool2_TP + IncumCand_SVM_bool2_TN)/(IncumCand_SVM_bool2_TP + IncumCand_SVM_bool2_FP + IncumCand_SVM_bool2_FN + IncumCand_SVM_bool2_TN)
print(IncumCand_SVM_bool2_ACC)

IncumCandAccuracyDict.update({'IncumCand_SVM_bool2_ACC': IncumCand_SVM_bool2_ACC})
print(IncumCandAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #3 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

IncumCand_B_SVM_Model3=LinearSVC(C=.01)
IncumCand_B_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
IncumCand_b_svm_predict3 = IncumCand_B_SVM_Model3.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(IncumCand_b_svm_predict3)
print("\nActual:")
print(TestLabelsB)

IncumCand_B_SVM_matrix3 = confusion_matrix(TestLabelsB, IncumCand_b_svm_predict3)
print("\nThe confusion matrix is:")
print(IncumCand_B_SVM_matrix3)
print("\n\n")

IncumCand_svm_B_target_names3 = ['0','1']
print(classification_report(TestLabelsB, IncumCand_b_svm_predict3, target_names = IncumCand_svm_B_target_names3))

IncumCand_SVM_bool3_FP = IncumCand_B_SVM_matrix3[0][1] 
IncumCand_SVM_bool3_FN = IncumCand_B_SVM_matrix3[1][0]
IncumCand_SVM_bool3_TP = IncumCand_B_SVM_matrix3[1][1]
IncumCand_SVM_bool3_TN = IncumCand_B_SVM_matrix3[0][0]

# Overall accuracy
IncumCand_SVM_bool3_ACC = (IncumCand_SVM_bool3_TP + IncumCand_SVM_bool3_TN)/(IncumCand_SVM_bool3_TP + IncumCand_SVM_bool3_FP + IncumCand_SVM_bool3_FN + IncumCand_SVM_bool3_TN)
print(IncumCand_SVM_bool3_ACC)

IncumCandAccuracyDict.update({'IncumCand_SVM_bool3_ACC': IncumCand_SVM_bool3_ACC})
print(IncumCandAccuracyDict)

#############################################
###########  SVM  for tfidf model #1 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

IncumCand_tf_SVM_Model=LinearSVC(C=.001)
IncumCand_tf_SVM_Model.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
IncumCand_tf_svm_predict = IncumCand_tf_SVM_Model.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(IncumCand_tf_svm_predict)
print("\nActual:")
print(TestLabels_tf)

IncumCand_tf_SVM_matrix = confusion_matrix(TestLabels_tf, IncumCand_tf_svm_predict)
print("\nThe confusion matrix is:")
print(IncumCand_tf_SVM_matrix)
print("\n\n")

IncumCand_svm_tf_target_names = ['0','1']
print(classification_report(TestLabels_tf, IncumCand_tf_svm_predict, target_names = IncumCand_svm_tf_target_names))

IncumCand_SVM_tf_FP = IncumCand_tf_SVM_matrix[0][1] 
IncumCand_SVM_tf_FN = IncumCand_tf_SVM_matrix[1][0]
IncumCand_SVM_tf_TP = IncumCand_tf_SVM_matrix[1][1]
IncumCand_SVM_tf_TN = IncumCand_tf_SVM_matrix[0][0]

# Overall accuracy
IncumCand_SVM_tf_ACC = (IncumCand_SVM_tf_TP + IncumCand_SVM_tf_TN)/(IncumCand_SVM_tf_TP + IncumCand_SVM_tf_FP + IncumCand_SVM_tf_FN + IncumCand_SVM_tf_TN)
print(IncumCand_SVM_tf_ACC)

IncumCandAccuracyDict.update({'IncumCand_SVM_tf_ACC': IncumCand_SVM_tf_ACC})
print(IncumCandAccuracyDict)

#############################################
###########  SVM  for tfidf model #2 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

IncumCand_tf_SVM_Model2=LinearSVC(C=1)
IncumCand_tf_SVM_Model2.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
IncumCand_tf_svm_predict2 = IncumCand_tf_SVM_Model2.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(IncumCand_tf_svm_predict2)
print("\nActual:")
print(TestLabels_tf)

IncumCand_tf_SVM_matrix2 = confusion_matrix(TestLabels_tf, IncumCand_tf_svm_predict2)
print("\nThe confusion matrix is:")
print(IncumCand_tf_SVM_matrix2)
print("\n\n")

IncumCand_svm_tf_target_names2 = ['0','1']
print(classification_report(TestLabels_tf, IncumCand_tf_svm_predict2, target_names = IncumCand_svm_tf_target_names2))

IncumCand_SVM_tf2_FP = IncumCand_tf_SVM_matrix2[0][1] 
IncumCand_SVM_tf2_FN = IncumCand_tf_SVM_matrix2[1][0]
IncumCand_SVM_tf2_TP = IncumCand_tf_SVM_matrix2[1][1]
IncumCand_SVM_tf2_TN = IncumCand_tf_SVM_matrix2[0][0]

# Overall accuracy
IncumCand_SVM_tf2_ACC = (IncumCand_SVM_tf2_TP + IncumCand_SVM_tf2_TN)/(IncumCand_SVM_tf2_TP + IncumCand_SVM_tf2_FP + IncumCand_SVM_tf2_FN + IncumCand_SVM_tf2_TN)
print(IncumCand_SVM_tf2_ACC)

IncumCandAccuracyDict.update({'IncumCand_SVM_tf2_ACC': IncumCand_SVM_tf2_ACC})
print(IncumCandAccuracyDict)

#############################################
###########  SVM  for tfidf model #3 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

IncumCand_tf_SVM_Model3=LinearSVC(C=1)
IncumCand_tf_SVM_Model3.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
IncumCand_tf_svm_predict3 = IncumCand_tf_SVM_Model3.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(IncumCand_tf_svm_predict3)
print("\nActual:")
print(TestLabels_tf)

IncumCand_tf_SVM_matrix3 = confusion_matrix(TestLabels_tf, IncumCand_tf_svm_predict3)
print("\nThe confusion matrix is:")
print(IncumCand_tf_SVM_matrix3)
print("\n\n")

IncumCand_svm_tf_target_names3 = ['0','1']
print(classification_report(TestLabels_tf, IncumCand_tf_svm_predict3, target_names = IncumCand_svm_tf_target_names3))

IncumCand_SVM_tf3_FP = IncumCand_tf_SVM_matrix3[0][1] 
IncumCand_SVM_tf3_FN = IncumCand_tf_SVM_matrix3[1][0]
IncumCand_SVM_tf3_TP = IncumCand_tf_SVM_matrix3[1][1]
IncumCand_SVM_tf3_TN = IncumCand_tf_SVM_matrix3[0][0]

# Overall accuracy
IncumCand_SVM_tf3_ACC = (IncumCand_SVM_tf3_TP + IncumCand_SVM_tf3_TN)/(IncumCand_SVM_tf3_TP + IncumCand_SVM_tf3_FP + IncumCand_SVM_tf3_FN + IncumCand_SVM_tf3_TN)
print(IncumCand_SVM_tf3_ACC)

IncumCandAccuracyDict.update({'IncumCand_SVM_tf3_ACC': IncumCand_SVM_tf3_ACC})
print(IncumCandAccuracyDict)

#############################################
#######  SVM  with different kernels ########
#############################################
## running on Bernoulli model since it had best accuracy above
## i want to see if the accuracy gets improved in the different kernels

#############################################
###############  Sigmoid #1 #################
#############################################

IncumCand_sig_SVM_Model=sklearn.svm.SVC(C=10, kernel='sigmoid', 
                           verbose=True, gamma="scale")
IncumCand_sig_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
IncumCand_sig_svm_predict = IncumCand_sig_SVM_Model.predict(TestDFB_nolabels)
print("\nsig SVM prediction:\n")
print(IncumCand_sig_svm_predict)
print("\nActual:")
print(TestLabelsB)

IncumCand_sig_SVM_matrix = confusion_matrix(TestLabelsB, IncumCand_sig_svm_predict)
print("\nThe sig confusion matrix is:")
print(IncumCand_sig_SVM_matrix)
print("\n\n")

IncumCand_svm_sig_target_names = ['0','1']
print(classification_report(TestLabelsB, IncumCand_sig_svm_predict, target_names = IncumCand_svm_sig_target_names))

IncumCand_SVM_sig_FP = IncumCand_sig_SVM_matrix[0][1] 
IncumCand_SVM_sig_FN = IncumCand_sig_SVM_matrix[1][0]
IncumCand_SVM_sig_TP = IncumCand_sig_SVM_matrix[1][1]
IncumCand_SVM_sig_TN = IncumCand_sig_SVM_matrix[0][0]

# Overall accuracy
IncumCand_SVM_sig_ACC = (IncumCand_SVM_sig_TP + IncumCand_SVM_sig_TN)/(IncumCand_SVM_sig_TP + IncumCand_SVM_sig_FP + IncumCand_SVM_sig_FN + IncumCand_SVM_sig_TN)
print(IncumCand_SVM_sig_ACC)

IncumCandAccuracyDict.update({'IncumCand_SVM_sig_ACC': IncumCand_SVM_sig_ACC})
print(IncumCandAccuracyDict)

#############################################
###############  Sigmoid #2 #################
#############################################

IncumCand_sig_SVM_Model2=sklearn.svm.SVC(C=1000, kernel='sigmoid', 
                           verbose=True, gamma="scale")
IncumCand_sig_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
IncumCand_sig_svm_predict2 = IncumCand_sig_SVM_Model2.predict(TestDFB_nolabels)
print("\nsig SVM prediction 2:\n")
print(IncumCand_sig_svm_predict2)
print("\nActual:")
print(TestLabelsB)

IncumCand_sig_SVM_matrix2 = confusion_matrix(TestLabelsB, IncumCand_sig_svm_predict2)
print("\nThe sig 2 confusion matrix is:")
print(IncumCand_sig_SVM_matrix2)
print("\n\n")

IncumCand_svm_sig_target_names2 = ['0','1']
print(classification_report(TestLabelsB, IncumCand_sig_svm_predict2, target_names = IncumCand_svm_sig_target_names2))

IncumCand_SVM_sig2_FP = IncumCand_sig_SVM_matrix2[0][1] 
IncumCand_SVM_sig2_FN = IncumCand_sig_SVM_matrix2[1][0]
IncumCand_SVM_sig2_TP = IncumCand_sig_SVM_matrix2[1][1]
IncumCand_SVM_sig2_TN = IncumCand_sig_SVM_matrix2[0][0]

# Overall accuracy
IncumCand_SVM_sig2_ACC = (IncumCand_SVM_sig2_TP + IncumCand_SVM_sig2_TN)/(IncumCand_SVM_sig2_TP + IncumCand_SVM_sig2_FP + IncumCand_SVM_sig2_FN + IncumCand_SVM_sig2_TN)
print(IncumCand_SVM_sig2_ACC)

IncumCandAccuracyDict.update({'IncumCand_SVM_sig2_ACC': IncumCand_SVM_sig2_ACC})
print(IncumCandAccuracyDict)

#############################################
###############  Sigmoid #3 #################
#############################################

IncumCand_sig_SVM_Model3=sklearn.svm.SVC(C=.001, kernel='sigmoid', 
                           verbose=True, gamma="scale")
IncumCand_sig_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
IncumCand_sig_svm_predict3 = IncumCand_sig_SVM_Model3.predict(TestDFB_nolabels)
print("\nsig SVM prediction 3:\n")
print(IncumCand_sig_svm_predict3)
print("\nActual:")
print(TestLabelsB)

IncumCand_sig_SVM_matrix3 = confusion_matrix(TestLabelsB, IncumCand_sig_svm_predict3)
print("\nThe sig 3 confusion matrix is:")
print(IncumCand_sig_SVM_matrix3)
print("\n\n")

IncumCand_svm_sig_target_names3 = ['0','1']
print(classification_report(TestLabelsB, IncumCand_sig_svm_predict3, target_names = IncumCand_svm_sig_target_names3))

IncumCand_SVM_sig3_FP = IncumCand_sig_SVM_matrix3[0][1] 
IncumCand_SVM_sig3_FN = IncumCand_sig_SVM_matrix3[1][0]
IncumCand_SVM_sig3_TP = IncumCand_sig_SVM_matrix3[1][1]
IncumCand_SVM_sig3_TN = IncumCand_sig_SVM_matrix3[0][0]

# Overall accuracy
IncumCand_SVM_sig3_ACC = (IncumCand_SVM_sig3_TP + IncumCand_SVM_sig3_TN)/(IncumCand_SVM_sig3_TP + IncumCand_SVM_sig3_FP + IncumCand_SVM_sig3_FN + IncumCand_SVM_sig3_TN)
print(IncumCand_SVM_sig3_ACC)

IncumCandAccuracyDict.update({'IncumCand_SVM_sig3_ACC': IncumCand_SVM_sig3_ACC})
print(IncumCandAccuracyDict)

#############################################
################  Poly #1 ###################
#############################################

IncumCand_poly_SVM_Model=sklearn.svm.SVC(C=1000, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(IncumCand_poly_SVM_Model)
IncumCand_poly_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
IncumCand_poly_svm_predict = IncumCand_poly_SVM_Model.predict(TestDFB_nolabels)
print("\npoly SVM prediction:\n")
print(IncumCand_poly_svm_predict)
print("\nActual:")
print(TestLabelsB)

IncumCand_poly_SVM_matrix = confusion_matrix(TestLabelsB, IncumCand_poly_svm_predict)
print("\nThe poly confusion matrix is:")
print(IncumCand_poly_SVM_matrix)
print("\n\n")

IncumCand_svm_poly_target_names = ['0','1']
print(classification_report(TestLabelsB, IncumCand_poly_svm_predict, target_names = IncumCand_svm_poly_target_names))

IncumCand_SVM_poly_FP = IncumCand_poly_SVM_matrix[0][1] 
IncumCand_SVM_poly_FN = IncumCand_poly_SVM_matrix[1][0]
IncumCand_SVM_poly_TP = IncumCand_poly_SVM_matrix[1][1]
IncumCand_SVM_poly_TN = IncumCand_poly_SVM_matrix[0][0]

# Overall accuracy
IncumCand_SVM_poly_ACC = (IncumCand_SVM_poly_TP + IncumCand_SVM_poly_TN)/(IncumCand_SVM_poly_TP + IncumCand_SVM_poly_FP + IncumCand_SVM_poly_FN + IncumCand_SVM_poly_TN)
print(IncumCand_SVM_poly_ACC)

IncumCandAccuracyDict.update({'IncumCand_SVM_poly_ACC': IncumCand_SVM_poly_ACC})
print(IncumCandAccuracyDict)

#############################################
################  Poly #2 ###################
#############################################

IncumCand_poly_SVM_Model2=sklearn.svm.SVC(C=10, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(IncumCand_poly_SVM_Model2)
IncumCand_poly_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
IncumCand_poly_svm_predict2 = IncumCand_poly_SVM_Model2.predict(TestDFB_nolabels)
print("\npoly SVM 2 prediction:\n")
print(IncumCand_poly_svm_predict2)
print("\nActual:")
print(TestLabelsB)

IncumCand_poly_SVM_matrix2 = confusion_matrix(TestLabelsB, IncumCand_poly_svm_predict2)
print("\nThe poly 2 confusion matrix is:")
print(IncumCand_poly_SVM_matrix2)
print("\n\n")

IncumCand_svm_poly_target_names2 = ['0','1']
print(classification_report(TestLabelsB, IncumCand_poly_svm_predict2, target_names = IncumCand_svm_poly_target_names2))

IncumCand_SVM_poly2_FP = IncumCand_poly_SVM_matrix2[0][1] 
IncumCand_SVM_poly2_FN = IncumCand_poly_SVM_matrix2[1][0]
IncumCand_SVM_poly2_TP = IncumCand_poly_SVM_matrix2[1][1]
IncumCand_SVM_poly2_TN = IncumCand_poly_SVM_matrix2[0][0]

# Overall accuracy
IncumCand_SVM_poly2_ACC = (IncumCand_SVM_poly2_TP + IncumCand_SVM_poly2_TN)/(IncumCand_SVM_poly2_TP + IncumCand_SVM_poly2_FP + IncumCand_SVM_poly2_FN + IncumCand_SVM_poly2_TN)
print(IncumCand_SVM_poly2_ACC)

IncumCandAccuracyDict.update({'IncumCand_SVM_poly2_ACC': IncumCand_SVM_poly2_ACC})
print(IncumCandAccuracyDict)

#############################################
################  Poly #3 ###################
#############################################

IncumCand_poly_SVM_Model3=sklearn.svm.SVC(C=.001, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(IncumCand_poly_SVM_Model3)
IncumCand_poly_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
IncumCand_poly_svm_predict3 = IncumCand_poly_SVM_Model3.predict(TestDFB_nolabels)
print("\npoly SVM 3 prediction:\n")
print(IncumCand_poly_svm_predict3)
print("\nActual:")
print(TestLabelsB)

IncumCand_poly_SVM_matrix3 = confusion_matrix(TestLabelsB, IncumCand_poly_svm_predict3)
print("\nThe poly 3 confusion matrix is:")
print(IncumCand_poly_SVM_matrix3)
print("\n\n")

IncumCand_svm_poly_target_names3 = ['0','1']
print(classification_report(TestLabelsB, IncumCand_poly_svm_predict3, target_names = IncumCand_svm_poly_target_names3))

IncumCand_SVM_poly3_FP = IncumCand_poly_SVM_matrix3[0][1] 
IncumCand_SVM_poly3_FN = IncumCand_poly_SVM_matrix3[1][0]
IncumCand_SVM_poly3_TP = IncumCand_poly_SVM_matrix3[1][1]
IncumCand_SVM_poly3_TN = IncumCand_poly_SVM_matrix3[0][0]

# Overall accuracy
IncumCand_SVM_poly3_ACC = (IncumCand_SVM_poly3_TP + IncumCand_SVM_poly3_TN)/(IncumCand_SVM_poly3_TP + IncumCand_SVM_poly3_FP + IncumCand_SVM_poly3_FN + IncumCand_SVM_poly3_TN)
print(IncumCand_SVM_poly3_ACC)

IncumCandAccuracyDict.update({'IncumCand_SVM_poly3_ACC': IncumCand_SVM_poly3_ACC})
print(IncumCandAccuracyDict)

IncumCandVisDF = pd.DataFrame(IncumCandAccuracyDict.items(), index = IncumCandAccuracyDict.keys(), columns=['Model','Accuracy'])
print(IncumCandVisDF)
SortedIncumCandVisDF = IncumCandVisDF.sort_values('Accuracy', ascending = [True])
print(SortedIncumCandVisDF)

print(PartyAccuracyDict)
print(WinnerAccuracyDict)
print(SentimentAccuracyDict)
print(IncumCandAccuracyDict)


SortedPartyVisDF.plot.barh(y='Accuracy')
SortedWinnerVisDF.plot.barh(y='Accuracy')
SortedSentimentVisDF.plot.barh(y='Accuracy')
SortedIncumCandVisDF.plot.barh(y='Accuracy')


#print(IndexList)
#print(TextList)
#print(PartyList)
#print(WinnerList)
#print(SentimentList)
#print(IncumCandList)
print(IncumPartyList)
print(UnemploymentList)
print(GDPList)
print(InflationList)
print(SatisfactionList)
print(RealIncomeGrowthList)
print(PresApprovalList)
print(IncomeTaxList)
print(DjiaVolumeList)
print(CpiList)
print(CciIndexList)

#################################################################################################
#################################################################################################
### everything from this point down is a matter of find and replace for each of the variables ###
#################################################################################################
#################################################################################################

### starting to build a model for IncumParty
IncumPartyVectDF = VectDF.copy(deep=True)
IncumPartyVectDF.insert(loc=0, column='LABEL', value=IncumPartyList)
print(IncumPartyVectDF)

bool_IncumPartyVectDF = bool_VectDF.copy(deep=True)
bool_IncumPartyVectDF.insert(loc=0, column='LABEL', value=IncumPartyList)
print(bool_IncumPartyVectDF)

tf_IncumPartyVectDF = tf_VectDF.copy(deep=True)
tf_IncumPartyVectDF.insert(loc=0, column='LABEL', value=IncumPartyList)
print(tf_IncumPartyVectDF)


##--------------------------------------------------------
####################################################################
###################### model Data Build ############################
####################################################################
##--------------------------------------------------------
from sklearn.model_selection import train_test_split

## Copy everything below here to rebuild for IncumParty data
TrainDF, TestDF = train_test_split(IncumPartyVectDF, test_size=0.3)
print(TrainDF.head())

## Now we have a training set and a testing set. 
print("\nThe training set is:")
print(TrainDF)
print("\nThe testing set is:")
print(TestDF)

## IMPORTANT - YOU CANNOT LEAVE LABELS ON THE TEST SET
## Save labels
TestLabels=TestDF["LABEL"]
print(TestLabels)
## remove labels
TestDF_nolabels = TestDF.drop(["LABEL"], axis=1)
print(TestDF_nolabels)

TrainDF_nolabels=TrainDF.drop(["LABEL"], axis=1)
print(TrainDF_nolabels)

TrainLabels=TrainDF["LABEL"]
print(TrainLabels)

############# Do the same for the binary dataframe
## Create Train/Test for TrainDFB
TrainDFB, TestDFB = train_test_split(bool_IncumPartyVectDF, test_size=0.3)
print(TrainDFB.head())

TrainDFB_nolabels=TrainDFB.drop(["LABEL"], axis=1)
print(TrainDFB_nolabels)
TrainLabelsB=TrainDFB["LABEL"]
print(TrainLabelsB)

TestDFB_nolabels=TestDFB.drop(["LABEL"], axis=1)
print(TestDFB_nolabels)
TestLabelsB=TestDFB["LABEL"]
print(TestLabelsB)

############# Do the same for the tfidf dataframe
## Create Train/Test for TrainDF tfidf
TrainDF_tf, TestDF_tf = train_test_split(tf_IncumPartyVectDF, test_size=0.3)
print(TrainDF_tf.head())

TrainDF_tf_nolabels=TrainDF_tf.drop(["LABEL"], axis=1)
print(TrainDF_tf_nolabels)
TrainLabels_tf=TrainDF_tf["LABEL"]
print(TrainLabels_tf)

TestDF_tf_nolabels=TestDF_tf.drop(["LABEL"], axis=1)
print(TestDF_tf_nolabels)
TestLabels_tf=TestDF_tf["LABEL"]
print(TestLabels_tf)



#############################################
###########  SVM  for regular model #1 ######
#############################################
from sklearn.svm import LinearSVC
print(TrainDF_nolabels.head())
print(TrainLabels.head())

IncumParty_SVM_Model=LinearSVC(C=.01)
IncumParty_SVM_Model.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
IncumParty_svm_predict = IncumParty_SVM_Model.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(IncumParty_svm_predict)
print("Actual:")
print(TestLabels)

from sklearn.metrics import confusion_matrix
IncumParty_SVM_matrix = confusion_matrix(TestLabels, IncumParty_svm_predict)
print("\nThe confusion matrix is:")
print(IncumParty_SVM_matrix)
print("\n\n")

from sklearn.metrics import classification_report
IncumParty_svm_target_names = ['0','1']
print(classification_report(TestLabels, IncumParty_svm_predict, target_names = IncumParty_svm_target_names))

IncumParty_SVM_reg_FP = IncumParty_SVM_matrix[0][1] 
IncumParty_SVM_reg_FN = IncumParty_SVM_matrix[1][0]
IncumParty_SVM_reg_TP = IncumParty_SVM_matrix[1][1]
IncumParty_SVM_reg_TN = IncumParty_SVM_matrix[0][0]

# Overall accuracy
IncumParty_SVM_reg_ACC = (IncumParty_SVM_reg_TP + IncumParty_SVM_reg_TN)/(IncumParty_SVM_reg_TP + IncumParty_SVM_reg_FP + IncumParty_SVM_reg_FN + IncumParty_SVM_reg_TN)
print(IncumParty_SVM_reg_ACC)

IncumPartyAccuracyDict = {}
IncumPartyAccuracyDict.update({'IncumParty_SVM_reg_ACC': IncumParty_SVM_reg_ACC})
print(IncumPartyAccuracyDict)

#############################################
###########  SVM  for regular model #2 ######
#############################################

IncumParty_SVM_Model2=LinearSVC(C=1)
IncumParty_SVM_Model2.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
IncumParty_svm_predict2 = IncumParty_SVM_Model2.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(IncumParty_svm_predict2)
print("Actual:")
print(TestLabels)

IncumParty_SVM_matrix2 = confusion_matrix(TestLabels, IncumParty_svm_predict2)
print("\nThe confusion matrix is:")
print(IncumParty_SVM_matrix2)
print("\n\n")

IncumParty_svm_target_names2 = ['0','1']
print(classification_report(TestLabels, IncumParty_svm_predict2, target_names = IncumParty_svm_target_names2))

IncumParty_SVM_reg2_FP = IncumParty_SVM_matrix2[0][1] 
IncumParty_SVM_reg2_FN = IncumParty_SVM_matrix2[1][0]
IncumParty_SVM_reg2_TP = IncumParty_SVM_matrix2[1][1]
IncumParty_SVM_reg2_TN = IncumParty_SVM_matrix2[0][0]

# Overall accuracy
IncumParty_SVM_reg2_ACC = (IncumParty_SVM_reg2_TP + IncumParty_SVM_reg2_TN)/(IncumParty_SVM_reg2_TP + IncumParty_SVM_reg2_FP + IncumParty_SVM_reg2_FN + IncumParty_SVM_reg2_TN)
print(IncumParty_SVM_reg2_ACC)

IncumPartyAccuracyDict.update({'IncumParty_SVM_reg2_ACC': IncumParty_SVM_reg2_ACC})
print(IncumPartyAccuracyDict)

#############################################
###########  SVM  for regular model #3 ######
#############################################

IncumParty_SVM_Model3=LinearSVC(C=100)
IncumParty_SVM_Model3.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
IncumParty_svm_predict3 = IncumParty_SVM_Model3.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(IncumParty_svm_predict3)
print("Actual:")
print(TestLabels)

IncumParty_SVM_matrix3 = confusion_matrix(TestLabels, IncumParty_svm_predict3)
print("\nThe confusion matrix is:")
print(IncumParty_SVM_matrix3)
print("\n\n")

from sklearn.metrics import classification_report
IncumParty_svm_target_names3 = ['0','1']
print(classification_report(TestLabels, IncumParty_svm_predict3, target_names = IncumParty_svm_target_names3))

IncumParty_SVM_reg3_FP = IncumParty_SVM_matrix3[0][1] 
IncumParty_SVM_reg3_FN = IncumParty_SVM_matrix3[1][0]
IncumParty_SVM_reg3_TP = IncumParty_SVM_matrix3[1][1]
IncumParty_SVM_reg3_TN = IncumParty_SVM_matrix3[0][0]

# Overall accuracy
IncumParty_SVM_reg3_ACC = (IncumParty_SVM_reg3_TP + IncumParty_SVM_reg3_TN)/(IncumParty_SVM_reg3_TP + IncumParty_SVM_reg3_FP + IncumParty_SVM_reg3_FN + IncumParty_SVM_reg3_TN)
print(IncumParty_SVM_reg3_ACC)

IncumPartyAccuracyDict.update({'IncumParty_SVM_reg3_ACC': IncumParty_SVM_reg3_ACC})
print(IncumPartyAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #1 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

IncumParty_B_SVM_Model=LinearSVC(C=100)
IncumParty_B_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
IncumParty_b_svm_predict = IncumParty_B_SVM_Model.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(IncumParty_b_svm_predict)
print("\nActual:")
print(TestLabelsB)

IncumParty_B_SVM_matrix = confusion_matrix(TestLabelsB, IncumParty_b_svm_predict)
print("\nThe confusion matrix is:")
print(IncumParty_B_SVM_matrix)
print("\n\n")

IncumParty_svm_B_target_names = ['0','1']
print(classification_report(TestLabelsB, IncumParty_b_svm_predict, target_names = IncumParty_svm_B_target_names))

IncumParty_SVM_bool_FP = IncumParty_B_SVM_matrix[0][1] 
IncumParty_SVM_bool_FN = IncumParty_B_SVM_matrix[1][0]
IncumParty_SVM_bool_TP = IncumParty_B_SVM_matrix[1][1]
IncumParty_SVM_bool_TN = IncumParty_B_SVM_matrix[0][0]

# Overall accuracy
IncumParty_SVM_bool_ACC = (IncumParty_SVM_bool_TP + IncumParty_SVM_bool_TN)/(IncumParty_SVM_bool_TP + IncumParty_SVM_bool_FP + IncumParty_SVM_bool_FN + IncumParty_SVM_bool_TN)
print(IncumParty_SVM_bool_ACC)

IncumPartyAccuracyDict.update({'IncumParty_SVM_bool_ACC': IncumParty_SVM_bool_ACC})
print(IncumPartyAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #2 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

IncumParty_B_SVM_Model2=LinearSVC(C=1)
IncumParty_B_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
IncumParty_b_svm_predict2 = IncumParty_B_SVM_Model2.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(IncumParty_b_svm_predict2)
print("\nActual:")
print(TestLabelsB)

IncumParty_B_SVM_matrix2 = confusion_matrix(TestLabelsB, IncumParty_b_svm_predict2)
print("\nThe confusion matrix is:")
print(IncumParty_B_SVM_matrix2)
print("\n\n")

IncumParty_svm_B_target_names2 = ['0','1']
print(classification_report(TestLabelsB, IncumParty_b_svm_predict2, target_names = IncumParty_svm_B_target_names2))

IncumParty_SVM_bool2_FP = IncumParty_B_SVM_matrix2[0][1] 
IncumParty_SVM_bool2_FN = IncumParty_B_SVM_matrix2[1][0]
IncumParty_SVM_bool2_TP = IncumParty_B_SVM_matrix2[1][1]
IncumParty_SVM_bool2_TN = IncumParty_B_SVM_matrix2[0][0]

# Overall accuracy
IncumParty_SVM_bool2_ACC = (IncumParty_SVM_bool2_TP + IncumParty_SVM_bool2_TN)/(IncumParty_SVM_bool2_TP + IncumParty_SVM_bool2_FP + IncumParty_SVM_bool2_FN + IncumParty_SVM_bool2_TN)
print(IncumParty_SVM_bool2_ACC)

IncumPartyAccuracyDict.update({'IncumParty_SVM_bool2_ACC': IncumParty_SVM_bool2_ACC})
print(IncumPartyAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #3 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

IncumParty_B_SVM_Model3=LinearSVC(C=.01)
IncumParty_B_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
IncumParty_b_svm_predict3 = IncumParty_B_SVM_Model3.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(IncumParty_b_svm_predict3)
print("\nActual:")
print(TestLabelsB)

IncumParty_B_SVM_matrix3 = confusion_matrix(TestLabelsB, IncumParty_b_svm_predict3)
print("\nThe confusion matrix is:")
print(IncumParty_B_SVM_matrix3)
print("\n\n")

IncumParty_svm_B_target_names3 = ['0','1']
print(classification_report(TestLabelsB, IncumParty_b_svm_predict3, target_names = IncumParty_svm_B_target_names3))

IncumParty_SVM_bool3_FP = IncumParty_B_SVM_matrix3[0][1] 
IncumParty_SVM_bool3_FN = IncumParty_B_SVM_matrix3[1][0]
IncumParty_SVM_bool3_TP = IncumParty_B_SVM_matrix3[1][1]
IncumParty_SVM_bool3_TN = IncumParty_B_SVM_matrix3[0][0]

# Overall accuracy
IncumParty_SVM_bool3_ACC = (IncumParty_SVM_bool3_TP + IncumParty_SVM_bool3_TN)/(IncumParty_SVM_bool3_TP + IncumParty_SVM_bool3_FP + IncumParty_SVM_bool3_FN + IncumParty_SVM_bool3_TN)
print(IncumParty_SVM_bool3_ACC)

IncumPartyAccuracyDict.update({'IncumParty_SVM_bool3_ACC': IncumParty_SVM_bool3_ACC})
print(IncumPartyAccuracyDict)

#############################################
###########  SVM  for tfidf model #1 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

IncumParty_tf_SVM_Model=LinearSVC(C=.001)
IncumParty_tf_SVM_Model.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
IncumParty_tf_svm_predict = IncumParty_tf_SVM_Model.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(IncumParty_tf_svm_predict)
print("\nActual:")
print(TestLabels_tf)

IncumParty_tf_SVM_matrix = confusion_matrix(TestLabels_tf, IncumParty_tf_svm_predict)
print("\nThe confusion matrix is:")
print(IncumParty_tf_SVM_matrix)
print("\n\n")

IncumParty_svm_tf_target_names = ['0','1']
print(classification_report(TestLabels_tf, IncumParty_tf_svm_predict, target_names = IncumParty_svm_tf_target_names))

IncumParty_SVM_tf_FP = IncumParty_tf_SVM_matrix[0][1] 
IncumParty_SVM_tf_FN = IncumParty_tf_SVM_matrix[1][0]
IncumParty_SVM_tf_TP = IncumParty_tf_SVM_matrix[1][1]
IncumParty_SVM_tf_TN = IncumParty_tf_SVM_matrix[0][0]

# Overall accuracy
IncumParty_SVM_tf_ACC = (IncumParty_SVM_tf_TP + IncumParty_SVM_tf_TN)/(IncumParty_SVM_tf_TP + IncumParty_SVM_tf_FP + IncumParty_SVM_tf_FN + IncumParty_SVM_tf_TN)
print(IncumParty_SVM_tf_ACC)

IncumPartyAccuracyDict.update({'IncumParty_SVM_tf_ACC': IncumParty_SVM_tf_ACC})
print(IncumPartyAccuracyDict)

#############################################
###########  SVM  for tfidf model #2 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

IncumParty_tf_SVM_Model2=LinearSVC(C=1)
IncumParty_tf_SVM_Model2.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
IncumParty_tf_svm_predict2 = IncumParty_tf_SVM_Model2.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(IncumParty_tf_svm_predict2)
print("\nActual:")
print(TestLabels_tf)

IncumParty_tf_SVM_matrix2 = confusion_matrix(TestLabels_tf, IncumParty_tf_svm_predict2)
print("\nThe confusion matrix is:")
print(IncumParty_tf_SVM_matrix2)
print("\n\n")

IncumParty_svm_tf_target_names2 = ['0','1']
print(classification_report(TestLabels_tf, IncumParty_tf_svm_predict2, target_names = IncumParty_svm_tf_target_names2))

IncumParty_SVM_tf2_FP = IncumParty_tf_SVM_matrix2[0][1] 
IncumParty_SVM_tf2_FN = IncumParty_tf_SVM_matrix2[1][0]
IncumParty_SVM_tf2_TP = IncumParty_tf_SVM_matrix2[1][1]
IncumParty_SVM_tf2_TN = IncumParty_tf_SVM_matrix2[0][0]

# Overall accuracy
IncumParty_SVM_tf2_ACC = (IncumParty_SVM_tf2_TP + IncumParty_SVM_tf2_TN)/(IncumParty_SVM_tf2_TP + IncumParty_SVM_tf2_FP + IncumParty_SVM_tf2_FN + IncumParty_SVM_tf2_TN)
print(IncumParty_SVM_tf2_ACC)

IncumPartyAccuracyDict.update({'IncumParty_SVM_tf2_ACC': IncumParty_SVM_tf2_ACC})
print(IncumPartyAccuracyDict)

#############################################
###########  SVM  for tfidf model #3 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

IncumParty_tf_SVM_Model3=LinearSVC(C=1)
IncumParty_tf_SVM_Model3.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
IncumParty_tf_svm_predict3 = IncumParty_tf_SVM_Model3.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(IncumParty_tf_svm_predict3)
print("\nActual:")
print(TestLabels_tf)

IncumParty_tf_SVM_matrix3 = confusion_matrix(TestLabels_tf, IncumParty_tf_svm_predict3)
print("\nThe confusion matrix is:")
print(IncumParty_tf_SVM_matrix3)
print("\n\n")

IncumParty_svm_tf_target_names3 = ['0','1']
print(classification_report(TestLabels_tf, IncumParty_tf_svm_predict3, target_names = IncumParty_svm_tf_target_names3))

IncumParty_SVM_tf3_FP = IncumParty_tf_SVM_matrix3[0][1] 
IncumParty_SVM_tf3_FN = IncumParty_tf_SVM_matrix3[1][0]
IncumParty_SVM_tf3_TP = IncumParty_tf_SVM_matrix3[1][1]
IncumParty_SVM_tf3_TN = IncumParty_tf_SVM_matrix3[0][0]

# Overall accuracy
IncumParty_SVM_tf3_ACC = (IncumParty_SVM_tf3_TP + IncumParty_SVM_tf3_TN)/(IncumParty_SVM_tf3_TP + IncumParty_SVM_tf3_FP + IncumParty_SVM_tf3_FN + IncumParty_SVM_tf3_TN)
print(IncumParty_SVM_tf3_ACC)

IncumPartyAccuracyDict.update({'IncumParty_SVM_tf3_ACC': IncumParty_SVM_tf3_ACC})
print(IncumPartyAccuracyDict)

#############################################
#######  SVM  with different kernels ########
#############################################
## running on Bernoulli model since it had best accuracy above
## i want to see if the accuracy gets improved in the different kernels

#############################################
###############  Sigmoid #1 #################
#############################################

IncumParty_sig_SVM_Model=sklearn.svm.SVC(C=10, kernel='sigmoid', 
                           verbose=True, gamma="scale")
IncumParty_sig_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
IncumParty_sig_svm_predict = IncumParty_sig_SVM_Model.predict(TestDFB_nolabels)
print("\nsig SVM prediction:\n")
print(IncumParty_sig_svm_predict)
print("\nActual:")
print(TestLabelsB)

IncumParty_sig_SVM_matrix = confusion_matrix(TestLabelsB, IncumParty_sig_svm_predict)
print("\nThe sig confusion matrix is:")
print(IncumParty_sig_SVM_matrix)
print("\n\n")

IncumParty_svm_sig_target_names = ['0','1']
print(classification_report(TestLabelsB, IncumParty_sig_svm_predict, target_names = IncumParty_svm_sig_target_names))

IncumParty_SVM_sig_FP = IncumParty_sig_SVM_matrix[0][1] 
IncumParty_SVM_sig_FN = IncumParty_sig_SVM_matrix[1][0]
IncumParty_SVM_sig_TP = IncumParty_sig_SVM_matrix[1][1]
IncumParty_SVM_sig_TN = IncumParty_sig_SVM_matrix[0][0]

# Overall accuracy
IncumParty_SVM_sig_ACC = (IncumParty_SVM_sig_TP + IncumParty_SVM_sig_TN)/(IncumParty_SVM_sig_TP + IncumParty_SVM_sig_FP + IncumParty_SVM_sig_FN + IncumParty_SVM_sig_TN)
print(IncumParty_SVM_sig_ACC)

IncumPartyAccuracyDict.update({'IncumParty_SVM_sig_ACC': IncumParty_SVM_sig_ACC})
print(IncumPartyAccuracyDict)

#############################################
###############  Sigmoid #2 #################
#############################################

IncumParty_sig_SVM_Model2=sklearn.svm.SVC(C=1000, kernel='sigmoid', 
                           verbose=True, gamma="scale")
IncumParty_sig_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
IncumParty_sig_svm_predict2 = IncumParty_sig_SVM_Model2.predict(TestDFB_nolabels)
print("\nsig SVM prediction 2:\n")
print(IncumParty_sig_svm_predict2)
print("\nActual:")
print(TestLabelsB)

IncumParty_sig_SVM_matrix2 = confusion_matrix(TestLabelsB, IncumParty_sig_svm_predict2)
print("\nThe sig 2 confusion matrix is:")
print(IncumParty_sig_SVM_matrix2)
print("\n\n")

IncumParty_svm_sig_target_names2 = ['0','1']
print(classification_report(TestLabelsB, IncumParty_sig_svm_predict2, target_names = IncumParty_svm_sig_target_names2))

IncumParty_SVM_sig2_FP = IncumParty_sig_SVM_matrix2[0][1] 
IncumParty_SVM_sig2_FN = IncumParty_sig_SVM_matrix2[1][0]
IncumParty_SVM_sig2_TP = IncumParty_sig_SVM_matrix2[1][1]
IncumParty_SVM_sig2_TN = IncumParty_sig_SVM_matrix2[0][0]

# Overall accuracy
IncumParty_SVM_sig2_ACC = (IncumParty_SVM_sig2_TP + IncumParty_SVM_sig2_TN)/(IncumParty_SVM_sig2_TP + IncumParty_SVM_sig2_FP + IncumParty_SVM_sig2_FN + IncumParty_SVM_sig2_TN)
print(IncumParty_SVM_sig2_ACC)

IncumPartyAccuracyDict.update({'IncumParty_SVM_sig2_ACC': IncumParty_SVM_sig2_ACC})
print(IncumPartyAccuracyDict)

#############################################
###############  Sigmoid #3 #################
#############################################

IncumParty_sig_SVM_Model3=sklearn.svm.SVC(C=.001, kernel='sigmoid', 
                           verbose=True, gamma="scale")
IncumParty_sig_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
IncumParty_sig_svm_predict3 = IncumParty_sig_SVM_Model3.predict(TestDFB_nolabels)
print("\nsig SVM prediction 3:\n")
print(IncumParty_sig_svm_predict3)
print("\nActual:")
print(TestLabelsB)

IncumParty_sig_SVM_matrix3 = confusion_matrix(TestLabelsB, IncumParty_sig_svm_predict3)
print("\nThe sig 3 confusion matrix is:")
print(IncumParty_sig_SVM_matrix3)
print("\n\n")

IncumParty_svm_sig_target_names3 = ['0','1']
print(classification_report(TestLabelsB, IncumParty_sig_svm_predict3, target_names = IncumParty_svm_sig_target_names3))

IncumParty_SVM_sig3_FP = IncumParty_sig_SVM_matrix3[0][1] 
IncumParty_SVM_sig3_FN = IncumParty_sig_SVM_matrix3[1][0]
IncumParty_SVM_sig3_TP = IncumParty_sig_SVM_matrix3[1][1]
IncumParty_SVM_sig3_TN = IncumParty_sig_SVM_matrix3[0][0]

# Overall accuracy
IncumParty_SVM_sig3_ACC = (IncumParty_SVM_sig3_TP + IncumParty_SVM_sig3_TN)/(IncumParty_SVM_sig3_TP + IncumParty_SVM_sig3_FP + IncumParty_SVM_sig3_FN + IncumParty_SVM_sig3_TN)
print(IncumParty_SVM_sig3_ACC)

IncumPartyAccuracyDict.update({'IncumParty_SVM_sig3_ACC': IncumParty_SVM_sig3_ACC})
print(IncumPartyAccuracyDict)

#############################################
################  Poly #1 ###################
#############################################

IncumParty_poly_SVM_Model=sklearn.svm.SVC(C=1000, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(IncumParty_poly_SVM_Model)
IncumParty_poly_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
IncumParty_poly_svm_predict = IncumParty_poly_SVM_Model.predict(TestDFB_nolabels)
print("\npoly SVM prediction:\n")
print(IncumParty_poly_svm_predict)
print("\nActual:")
print(TestLabelsB)

IncumParty_poly_SVM_matrix = confusion_matrix(TestLabelsB, IncumParty_poly_svm_predict)
print("\nThe poly confusion matrix is:")
print(IncumParty_poly_SVM_matrix)
print("\n\n")

IncumParty_svm_poly_target_names = ['0','1']
print(classification_report(TestLabelsB, IncumParty_poly_svm_predict, target_names = IncumParty_svm_poly_target_names))

IncumParty_SVM_poly_FP = IncumParty_poly_SVM_matrix[0][1] 
IncumParty_SVM_poly_FN = IncumParty_poly_SVM_matrix[1][0]
IncumParty_SVM_poly_TP = IncumParty_poly_SVM_matrix[1][1]
IncumParty_SVM_poly_TN = IncumParty_poly_SVM_matrix[0][0]

# Overall accuracy
IncumParty_SVM_poly_ACC = (IncumParty_SVM_poly_TP + IncumParty_SVM_poly_TN)/(IncumParty_SVM_poly_TP + IncumParty_SVM_poly_FP + IncumParty_SVM_poly_FN + IncumParty_SVM_poly_TN)
print(IncumParty_SVM_poly_ACC)

IncumPartyAccuracyDict.update({'IncumParty_SVM_poly_ACC': IncumParty_SVM_poly_ACC})
print(IncumPartyAccuracyDict)

#############################################
################  Poly #2 ###################
#############################################

IncumParty_poly_SVM_Model2=sklearn.svm.SVC(C=10, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(IncumParty_poly_SVM_Model2)
IncumParty_poly_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
IncumParty_poly_svm_predict2 = IncumParty_poly_SVM_Model2.predict(TestDFB_nolabels)
print("\npoly SVM 2 prediction:\n")
print(IncumParty_poly_svm_predict2)
print("\nActual:")
print(TestLabelsB)

IncumParty_poly_SVM_matrix2 = confusion_matrix(TestLabelsB, IncumParty_poly_svm_predict2)
print("\nThe poly 2 confusion matrix is:")
print(IncumParty_poly_SVM_matrix2)
print("\n\n")

IncumParty_svm_poly_target_names2 = ['0','1']
print(classification_report(TestLabelsB, IncumParty_poly_svm_predict2, target_names = IncumParty_svm_poly_target_names2))

IncumParty_SVM_poly2_FP = IncumParty_poly_SVM_matrix2[0][1] 
IncumParty_SVM_poly2_FN = IncumParty_poly_SVM_matrix2[1][0]
IncumParty_SVM_poly2_TP = IncumParty_poly_SVM_matrix2[1][1]
IncumParty_SVM_poly2_TN = IncumParty_poly_SVM_matrix2[0][0]

# Overall accuracy
IncumParty_SVM_poly2_ACC = (IncumParty_SVM_poly2_TP + IncumParty_SVM_poly2_TN)/(IncumParty_SVM_poly2_TP + IncumParty_SVM_poly2_FP + IncumParty_SVM_poly2_FN + IncumParty_SVM_poly2_TN)
print(IncumParty_SVM_poly2_ACC)

IncumPartyAccuracyDict.update({'IncumParty_SVM_poly2_ACC': IncumParty_SVM_poly2_ACC})
print(IncumPartyAccuracyDict)

#############################################
################  Poly #3 ###################
#############################################

IncumParty_poly_SVM_Model3=sklearn.svm.SVC(C=.001, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(IncumParty_poly_SVM_Model3)
IncumParty_poly_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
IncumParty_poly_svm_predict3 = IncumParty_poly_SVM_Model3.predict(TestDFB_nolabels)
print("\npoly SVM 3 prediction:\n")
print(IncumParty_poly_svm_predict3)
print("\nActual:")
print(TestLabelsB)

IncumParty_poly_SVM_matrix3 = confusion_matrix(TestLabelsB, IncumParty_poly_svm_predict3)
print("\nThe poly 3 confusion matrix is:")
print(IncumParty_poly_SVM_matrix3)
print("\n\n")

IncumParty_svm_poly_target_names3 = ['0','1']
print(classification_report(TestLabelsB, IncumParty_poly_svm_predict3, target_names = IncumParty_svm_poly_target_names3))

IncumParty_SVM_poly3_FP = IncumParty_poly_SVM_matrix3[0][1] 
IncumParty_SVM_poly3_FN = IncumParty_poly_SVM_matrix3[1][0]
IncumParty_SVM_poly3_TP = IncumParty_poly_SVM_matrix3[1][1]
IncumParty_SVM_poly3_TN = IncumParty_poly_SVM_matrix3[0][0]

# Overall accuracy
IncumParty_SVM_poly3_ACC = (IncumParty_SVM_poly3_TP + IncumParty_SVM_poly3_TN)/(IncumParty_SVM_poly3_TP + IncumParty_SVM_poly3_FP + IncumParty_SVM_poly3_FN + IncumParty_SVM_poly3_TN)
print(IncumParty_SVM_poly3_ACC)

IncumPartyAccuracyDict.update({'IncumParty_SVM_poly3_ACC': IncumParty_SVM_poly3_ACC})
print(IncumPartyAccuracyDict)

IncumPartyVisDF = pd.DataFrame(IncumPartyAccuracyDict.items(), index = IncumPartyAccuracyDict.keys(), columns=['Model','Accuracy'])
print(IncumPartyVisDF)
SortedIncumPartyVisDF = IncumPartyVisDF.sort_values('Accuracy', ascending = [True])
print(SortedIncumPartyVisDF)

print(PartyAccuracyDict)
print(WinnerAccuracyDict)
print(SentimentAccuracyDict)
print(IncumCandAccuracyDict)
print(IncumPartyAccuracyDict)

SortedPartyVisDF.plot.barh(y='Accuracy')
SortedWinnerVisDF.plot.barh(y='Accuracy')
SortedSentimentVisDF.plot.barh(y='Accuracy')
SortedIncumCandVisDF.plot.barh(y='Accuracy')
SortedIncumPartyVisDF.plot.barh(y='Accuracy')

#print(IndexList)
#print(TextList)
#print(PartyList)
#print(WinnerList)
#print(SentimentList)
#print(IncumCandList)
#print(IncumPartyList)
print(UnemploymentList)
print(GDPList)
print(InflationList)
print(SatisfactionList)
print(RealIncomeGrowthList)
print(PresApprovalList)
print(IncomeTaxList)
print(DjiaVolumeList)
print(CpiList)
print(CciIndexList)

#################################################################################################
#################################################################################################
### everything from this point down is a matter of find and replace for each of the variables ###
#################################################################################################
#################################################################################################

### starting to build a model for Unemployment
UnemploymentVectDF = VectDF.copy(deep=True)
UnemploymentVectDF.insert(loc=0, column='LABEL', value=UnemploymentList)
print(UnemploymentVectDF)

bool_UnemploymentVectDF = bool_VectDF.copy(deep=True)
bool_UnemploymentVectDF.insert(loc=0, column='LABEL', value=UnemploymentList)
print(bool_UnemploymentVectDF)

tf_UnemploymentVectDF = tf_VectDF.copy(deep=True)
tf_UnemploymentVectDF.insert(loc=0, column='LABEL', value=UnemploymentList)
print(tf_UnemploymentVectDF)


##--------------------------------------------------------
####################################################################
###################### model Data Build ############################
####################################################################
##--------------------------------------------------------
from sklearn.model_selection import train_test_split

## Copy everything below here to rebuild for Unemployment data
TrainDF, TestDF = train_test_split(UnemploymentVectDF, test_size=0.3)
print(TrainDF.head())

## Now we have a training set and a testing set. 
print("\nThe training set is:")
print(TrainDF)
print("\nThe testing set is:")
print(TestDF)

## IMPORTANT - YOU CANNOT LEAVE LABELS ON THE TEST SET
## Save labels
TestLabels=TestDF["LABEL"]
print(TestLabels)
## remove labels
TestDF_nolabels = TestDF.drop(["LABEL"], axis=1)
print(TestDF_nolabels)

TrainDF_nolabels=TrainDF.drop(["LABEL"], axis=1)
print(TrainDF_nolabels)

TrainLabels=TrainDF["LABEL"]
print(TrainLabels)

############# Do the same for the binary dataframe
## Create Train/Test for TrainDFB
TrainDFB, TestDFB = train_test_split(bool_UnemploymentVectDF, test_size=0.3)
print(TrainDFB.head())

TrainDFB_nolabels=TrainDFB.drop(["LABEL"], axis=1)
print(TrainDFB_nolabels)
TrainLabelsB=TrainDFB["LABEL"]
print(TrainLabelsB)

TestDFB_nolabels=TestDFB.drop(["LABEL"], axis=1)
print(TestDFB_nolabels)
TestLabelsB=TestDFB["LABEL"]
print(TestLabelsB)

############# Do the same for the tfidf dataframe
## Create Train/Test for TrainDF tfidf
TrainDF_tf, TestDF_tf = train_test_split(tf_UnemploymentVectDF, test_size=0.3)
print(TrainDF_tf.head())

TrainDF_tf_nolabels=TrainDF_tf.drop(["LABEL"], axis=1)
print(TrainDF_tf_nolabels)
TrainLabels_tf=TrainDF_tf["LABEL"]
print(TrainLabels_tf)

TestDF_tf_nolabels=TestDF_tf.drop(["LABEL"], axis=1)
print(TestDF_tf_nolabels)
TestLabels_tf=TestDF_tf["LABEL"]
print(TestLabels_tf)



#############################################
###########  SVM  for regular model #1 ######
#############################################
from sklearn.svm import LinearSVC
print(TrainDF_nolabels.head())
print(TrainLabels.head())

Unemployment_SVM_Model=LinearSVC(C=.01)
Unemployment_SVM_Model.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Unemployment_svm_predict = Unemployment_SVM_Model.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(Unemployment_svm_predict)
print("Actual:")
print(TestLabels)

from sklearn.metrics import confusion_matrix
Unemployment_SVM_matrix = confusion_matrix(TestLabels, Unemployment_svm_predict)
print("\nThe confusion matrix is:")
print(Unemployment_SVM_matrix)
print("\n\n")

from sklearn.metrics import classification_report
Unemployment_svm_target_names = ['0','1']
print(classification_report(TestLabels, Unemployment_svm_predict, target_names = Unemployment_svm_target_names))

Unemployment_SVM_reg_FP = Unemployment_SVM_matrix[0][1] 
Unemployment_SVM_reg_FN = Unemployment_SVM_matrix[1][0]
Unemployment_SVM_reg_TP = Unemployment_SVM_matrix[1][1]
Unemployment_SVM_reg_TN = Unemployment_SVM_matrix[0][0]

# Overall accuracy
Unemployment_SVM_reg_ACC = (Unemployment_SVM_reg_TP + Unemployment_SVM_reg_TN)/(Unemployment_SVM_reg_TP + Unemployment_SVM_reg_FP + Unemployment_SVM_reg_FN + Unemployment_SVM_reg_TN)
print(Unemployment_SVM_reg_ACC)

UnemploymentAccuracyDict = {}
UnemploymentAccuracyDict.update({'Unemployment_SVM_reg_ACC': Unemployment_SVM_reg_ACC})
print(UnemploymentAccuracyDict)

#############################################
###########  SVM  for regular model #2 ######
#############################################

Unemployment_SVM_Model2=LinearSVC(C=1)
Unemployment_SVM_Model2.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Unemployment_svm_predict2 = Unemployment_SVM_Model2.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(Unemployment_svm_predict2)
print("Actual:")
print(TestLabels)

Unemployment_SVM_matrix2 = confusion_matrix(TestLabels, Unemployment_svm_predict2)
print("\nThe confusion matrix is:")
print(Unemployment_SVM_matrix2)
print("\n\n")

Unemployment_svm_target_names2 = ['0','1']
print(classification_report(TestLabels, Unemployment_svm_predict2, target_names = Unemployment_svm_target_names2))

Unemployment_SVM_reg2_FP = Unemployment_SVM_matrix2[0][1] 
Unemployment_SVM_reg2_FN = Unemployment_SVM_matrix2[1][0]
Unemployment_SVM_reg2_TP = Unemployment_SVM_matrix2[1][1]
Unemployment_SVM_reg2_TN = Unemployment_SVM_matrix2[0][0]

# Overall accuracy
Unemployment_SVM_reg2_ACC = (Unemployment_SVM_reg2_TP + Unemployment_SVM_reg2_TN)/(Unemployment_SVM_reg2_TP + Unemployment_SVM_reg2_FP + Unemployment_SVM_reg2_FN + Unemployment_SVM_reg2_TN)
print(Unemployment_SVM_reg2_ACC)

UnemploymentAccuracyDict.update({'Unemployment_SVM_reg2_ACC': Unemployment_SVM_reg2_ACC})
print(UnemploymentAccuracyDict)

#############################################
###########  SVM  for regular model #3 ######
#############################################

Unemployment_SVM_Model3=LinearSVC(C=100)
Unemployment_SVM_Model3.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Unemployment_svm_predict3 = Unemployment_SVM_Model3.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(Unemployment_svm_predict3)
print("Actual:")
print(TestLabels)

Unemployment_SVM_matrix3 = confusion_matrix(TestLabels, Unemployment_svm_predict3)
print("\nThe confusion matrix is:")
print(Unemployment_SVM_matrix3)
print("\n\n")

from sklearn.metrics import classification_report
Unemployment_svm_target_names3 = ['0','1']
print(classification_report(TestLabels, Unemployment_svm_predict3, target_names = Unemployment_svm_target_names3))

Unemployment_SVM_reg3_FP = Unemployment_SVM_matrix3[0][1] 
Unemployment_SVM_reg3_FN = Unemployment_SVM_matrix3[1][0]
Unemployment_SVM_reg3_TP = Unemployment_SVM_matrix3[1][1]
Unemployment_SVM_reg3_TN = Unemployment_SVM_matrix3[0][0]

# Overall accuracy
Unemployment_SVM_reg3_ACC = (Unemployment_SVM_reg3_TP + Unemployment_SVM_reg3_TN)/(Unemployment_SVM_reg3_TP + Unemployment_SVM_reg3_FP + Unemployment_SVM_reg3_FN + Unemployment_SVM_reg3_TN)
print(Unemployment_SVM_reg3_ACC)

UnemploymentAccuracyDict.update({'Unemployment_SVM_reg3_ACC': Unemployment_SVM_reg3_ACC})
print(UnemploymentAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #1 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

Unemployment_B_SVM_Model=LinearSVC(C=100)
Unemployment_B_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Unemployment_b_svm_predict = Unemployment_B_SVM_Model.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(Unemployment_b_svm_predict)
print("\nActual:")
print(TestLabelsB)

Unemployment_B_SVM_matrix = confusion_matrix(TestLabelsB, Unemployment_b_svm_predict)
print("\nThe confusion matrix is:")
print(Unemployment_B_SVM_matrix)
print("\n\n")

Unemployment_svm_B_target_names = ['0','1']
print(classification_report(TestLabelsB, Unemployment_b_svm_predict, target_names = Unemployment_svm_B_target_names))

Unemployment_SVM_bool_FP = Unemployment_B_SVM_matrix[0][1] 
Unemployment_SVM_bool_FN = Unemployment_B_SVM_matrix[1][0]
Unemployment_SVM_bool_TP = Unemployment_B_SVM_matrix[1][1]
Unemployment_SVM_bool_TN = Unemployment_B_SVM_matrix[0][0]

# Overall accuracy
Unemployment_SVM_bool_ACC = (Unemployment_SVM_bool_TP + Unemployment_SVM_bool_TN)/(Unemployment_SVM_bool_TP + Unemployment_SVM_bool_FP + Unemployment_SVM_bool_FN + Unemployment_SVM_bool_TN)
print(Unemployment_SVM_bool_ACC)

UnemploymentAccuracyDict.update({'Unemployment_SVM_bool_ACC': Unemployment_SVM_bool_ACC})
print(UnemploymentAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #2 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

Unemployment_B_SVM_Model2=LinearSVC(C=1)
Unemployment_B_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Unemployment_b_svm_predict2 = Unemployment_B_SVM_Model2.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(Unemployment_b_svm_predict2)
print("\nActual:")
print(TestLabelsB)

Unemployment_B_SVM_matrix2 = confusion_matrix(TestLabelsB, Unemployment_b_svm_predict2)
print("\nThe confusion matrix is:")
print(Unemployment_B_SVM_matrix2)
print("\n\n")

Unemployment_svm_B_target_names2 = ['0','1']
print(classification_report(TestLabelsB, Unemployment_b_svm_predict2, target_names = Unemployment_svm_B_target_names2))

Unemployment_SVM_bool2_FP = Unemployment_B_SVM_matrix2[0][1] 
Unemployment_SVM_bool2_FN = Unemployment_B_SVM_matrix2[1][0]
Unemployment_SVM_bool2_TP = Unemployment_B_SVM_matrix2[1][1]
Unemployment_SVM_bool2_TN = Unemployment_B_SVM_matrix2[0][0]

# Overall accuracy
Unemployment_SVM_bool2_ACC = (Unemployment_SVM_bool2_TP + Unemployment_SVM_bool2_TN)/(Unemployment_SVM_bool2_TP + Unemployment_SVM_bool2_FP + Unemployment_SVM_bool2_FN + Unemployment_SVM_bool2_TN)
print(Unemployment_SVM_bool2_ACC)

UnemploymentAccuracyDict.update({'Unemployment_SVM_bool2_ACC': Unemployment_SVM_bool2_ACC})
print(UnemploymentAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #3 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

Unemployment_B_SVM_Model3=LinearSVC(C=.01)
Unemployment_B_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Unemployment_b_svm_predict3 = Unemployment_B_SVM_Model3.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(Unemployment_b_svm_predict3)
print("\nActual:")
print(TestLabelsB)

Unemployment_B_SVM_matrix3 = confusion_matrix(TestLabelsB, Unemployment_b_svm_predict3)
print("\nThe confusion matrix is:")
print(Unemployment_B_SVM_matrix3)
print("\n\n")

Unemployment_svm_B_target_names3 = ['0','1']
print(classification_report(TestLabelsB, Unemployment_b_svm_predict3, target_names = Unemployment_svm_B_target_names3))

Unemployment_SVM_bool3_FP = Unemployment_B_SVM_matrix3[0][1] 
Unemployment_SVM_bool3_FN = Unemployment_B_SVM_matrix3[1][0]
Unemployment_SVM_bool3_TP = Unemployment_B_SVM_matrix3[1][1]
Unemployment_SVM_bool3_TN = Unemployment_B_SVM_matrix3[0][0]

# Overall accuracy
Unemployment_SVM_bool3_ACC = (Unemployment_SVM_bool3_TP + Unemployment_SVM_bool3_TN)/(Unemployment_SVM_bool3_TP + Unemployment_SVM_bool3_FP + Unemployment_SVM_bool3_FN + Unemployment_SVM_bool3_TN)
print(Unemployment_SVM_bool3_ACC)

UnemploymentAccuracyDict.update({'Unemployment_SVM_bool3_ACC': Unemployment_SVM_bool3_ACC})
print(UnemploymentAccuracyDict)

#############################################
###########  SVM  for tfidf model #1 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

Unemployment_tf_SVM_Model=LinearSVC(C=.001)
Unemployment_tf_SVM_Model.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Unemployment_tf_svm_predict = Unemployment_tf_SVM_Model.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(Unemployment_tf_svm_predict)
print("\nActual:")
print(TestLabels_tf)

Unemployment_tf_SVM_matrix = confusion_matrix(TestLabels_tf, Unemployment_tf_svm_predict)
print("\nThe confusion matrix is:")
print(Unemployment_tf_SVM_matrix)
print("\n\n")

Unemployment_svm_tf_target_names = ['0','1']
print(classification_report(TestLabels_tf, Unemployment_tf_svm_predict, target_names = Unemployment_svm_tf_target_names))

Unemployment_SVM_tf_FP = Unemployment_tf_SVM_matrix[0][1] 
Unemployment_SVM_tf_FN = Unemployment_tf_SVM_matrix[1][0]
Unemployment_SVM_tf_TP = Unemployment_tf_SVM_matrix[1][1]
Unemployment_SVM_tf_TN = Unemployment_tf_SVM_matrix[0][0]

# Overall accuracy
Unemployment_SVM_tf_ACC = (Unemployment_SVM_tf_TP + Unemployment_SVM_tf_TN)/(Unemployment_SVM_tf_TP + Unemployment_SVM_tf_FP + Unemployment_SVM_tf_FN + Unemployment_SVM_tf_TN)
print(Unemployment_SVM_tf_ACC)

UnemploymentAccuracyDict.update({'Unemployment_SVM_tf_ACC': Unemployment_SVM_tf_ACC})
print(UnemploymentAccuracyDict)

#############################################
###########  SVM  for tfidf model #2 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

Unemployment_tf_SVM_Model2=LinearSVC(C=1)
Unemployment_tf_SVM_Model2.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Unemployment_tf_svm_predict2 = Unemployment_tf_SVM_Model2.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(Unemployment_tf_svm_predict2)
print("\nActual:")
print(TestLabels_tf)

Unemployment_tf_SVM_matrix2 = confusion_matrix(TestLabels_tf, Unemployment_tf_svm_predict2)
print("\nThe confusion matrix is:")
print(Unemployment_tf_SVM_matrix2)
print("\n\n")

Unemployment_svm_tf_target_names2 = ['0','1']
print(classification_report(TestLabels_tf, Unemployment_tf_svm_predict2, target_names = Unemployment_svm_tf_target_names2))

Unemployment_SVM_tf2_FP = Unemployment_tf_SVM_matrix2[0][1] 
Unemployment_SVM_tf2_FN = Unemployment_tf_SVM_matrix2[1][0]
Unemployment_SVM_tf2_TP = Unemployment_tf_SVM_matrix2[1][1]
Unemployment_SVM_tf2_TN = Unemployment_tf_SVM_matrix2[0][0]

# Overall accuracy
Unemployment_SVM_tf2_ACC = (Unemployment_SVM_tf2_TP + Unemployment_SVM_tf2_TN)/(Unemployment_SVM_tf2_TP + Unemployment_SVM_tf2_FP + Unemployment_SVM_tf2_FN + Unemployment_SVM_tf2_TN)
print(Unemployment_SVM_tf2_ACC)

UnemploymentAccuracyDict.update({'Unemployment_SVM_tf2_ACC': Unemployment_SVM_tf2_ACC})
print(UnemploymentAccuracyDict)

#############################################
###########  SVM  for tfidf model #3 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

Unemployment_tf_SVM_Model3=LinearSVC(C=1)
Unemployment_tf_SVM_Model3.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Unemployment_tf_svm_predict3 = Unemployment_tf_SVM_Model3.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(Unemployment_tf_svm_predict3)
print("\nActual:")
print(TestLabels_tf)

Unemployment_tf_SVM_matrix3 = confusion_matrix(TestLabels_tf, Unemployment_tf_svm_predict3)
print("\nThe confusion matrix is:")
print(Unemployment_tf_SVM_matrix3)
print("\n\n")

Unemployment_svm_tf_target_names3 = ['0','1']
print(classification_report(TestLabels_tf, Unemployment_tf_svm_predict3, target_names = Unemployment_svm_tf_target_names3))

Unemployment_SVM_tf3_FP = Unemployment_tf_SVM_matrix3[0][1] 
Unemployment_SVM_tf3_FN = Unemployment_tf_SVM_matrix3[1][0]
Unemployment_SVM_tf3_TP = Unemployment_tf_SVM_matrix3[1][1]
Unemployment_SVM_tf3_TN = Unemployment_tf_SVM_matrix3[0][0]

# Overall accuracy
Unemployment_SVM_tf3_ACC = (Unemployment_SVM_tf3_TP + Unemployment_SVM_tf3_TN)/(Unemployment_SVM_tf3_TP + Unemployment_SVM_tf3_FP + Unemployment_SVM_tf3_FN + Unemployment_SVM_tf3_TN)
print(Unemployment_SVM_tf3_ACC)

UnemploymentAccuracyDict.update({'Unemployment_SVM_tf3_ACC': Unemployment_SVM_tf3_ACC})
print(UnemploymentAccuracyDict)

#############################################
#######  SVM  with different kernels ########
#############################################
## running on Bernoulli model since it had best accuracy above
## i want to see if the accuracy gets improved in the different kernels

#############################################
###############  Sigmoid #1 #################
#############################################

Unemployment_sig_SVM_Model=sklearn.svm.SVC(C=10, kernel='sigmoid', 
                           verbose=True, gamma="scale")
Unemployment_sig_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
Unemployment_sig_svm_predict = Unemployment_sig_SVM_Model.predict(TestDFB_nolabels)
print("\nsig SVM prediction:\n")
print(Unemployment_sig_svm_predict)
print("\nActual:")
print(TestLabelsB)

Unemployment_sig_SVM_matrix = confusion_matrix(TestLabelsB, Unemployment_sig_svm_predict)
print("\nThe sig confusion matrix is:")
print(Unemployment_sig_SVM_matrix)
print("\n\n")

Unemployment_svm_sig_target_names = ['0','1']
print(classification_report(TestLabelsB, Unemployment_sig_svm_predict, target_names = Unemployment_svm_sig_target_names))

Unemployment_SVM_sig_FP = Unemployment_sig_SVM_matrix[0][1] 
Unemployment_SVM_sig_FN = Unemployment_sig_SVM_matrix[1][0]
Unemployment_SVM_sig_TP = Unemployment_sig_SVM_matrix[1][1]
Unemployment_SVM_sig_TN = Unemployment_sig_SVM_matrix[0][0]

# Overall accuracy
Unemployment_SVM_sig_ACC = (Unemployment_SVM_sig_TP + Unemployment_SVM_sig_TN)/(Unemployment_SVM_sig_TP + Unemployment_SVM_sig_FP + Unemployment_SVM_sig_FN + Unemployment_SVM_sig_TN)
print(Unemployment_SVM_sig_ACC)

UnemploymentAccuracyDict.update({'Unemployment_SVM_sig_ACC': Unemployment_SVM_sig_ACC})
print(UnemploymentAccuracyDict)

#############################################
###############  Sigmoid #2 #################
#############################################

Unemployment_sig_SVM_Model2=sklearn.svm.SVC(C=1000, kernel='sigmoid', 
                           verbose=True, gamma="scale")
Unemployment_sig_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
Unemployment_sig_svm_predict2 = Unemployment_sig_SVM_Model2.predict(TestDFB_nolabels)
print("\nsig SVM prediction 2:\n")
print(Unemployment_sig_svm_predict2)
print("\nActual:")
print(TestLabelsB)

Unemployment_sig_SVM_matrix2 = confusion_matrix(TestLabelsB, Unemployment_sig_svm_predict2)
print("\nThe sig 2 confusion matrix is:")
print(Unemployment_sig_SVM_matrix2)
print("\n\n")

Unemployment_svm_sig_target_names2 = ['0','1']
print(classification_report(TestLabelsB, Unemployment_sig_svm_predict2, target_names = Unemployment_svm_sig_target_names2))

Unemployment_SVM_sig2_FP = Unemployment_sig_SVM_matrix2[0][1] 
Unemployment_SVM_sig2_FN = Unemployment_sig_SVM_matrix2[1][0]
Unemployment_SVM_sig2_TP = Unemployment_sig_SVM_matrix2[1][1]
Unemployment_SVM_sig2_TN = Unemployment_sig_SVM_matrix2[0][0]

# Overall accuracy
Unemployment_SVM_sig2_ACC = (Unemployment_SVM_sig2_TP + Unemployment_SVM_sig2_TN)/(Unemployment_SVM_sig2_TP + Unemployment_SVM_sig2_FP + Unemployment_SVM_sig2_FN + Unemployment_SVM_sig2_TN)
print(Unemployment_SVM_sig2_ACC)

UnemploymentAccuracyDict.update({'Unemployment_SVM_sig2_ACC': Unemployment_SVM_sig2_ACC})
print(UnemploymentAccuracyDict)

#############################################
###############  Sigmoid #3 #################
#############################################

Unemployment_sig_SVM_Model3=sklearn.svm.SVC(C=.001, kernel='sigmoid', 
                           verbose=True, gamma="scale")
Unemployment_sig_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
Unemployment_sig_svm_predict3 = Unemployment_sig_SVM_Model3.predict(TestDFB_nolabels)
print("\nsig SVM prediction 3:\n")
print(Unemployment_sig_svm_predict3)
print("\nActual:")
print(TestLabelsB)

Unemployment_sig_SVM_matrix3 = confusion_matrix(TestLabelsB, Unemployment_sig_svm_predict3)
print("\nThe sig 3 confusion matrix is:")
print(Unemployment_sig_SVM_matrix3)
print("\n\n")

Unemployment_svm_sig_target_names3 = ['0','1']
print(classification_report(TestLabelsB, Unemployment_sig_svm_predict3, target_names = Unemployment_svm_sig_target_names3))

Unemployment_SVM_sig3_FP = Unemployment_sig_SVM_matrix3[0][1] 
Unemployment_SVM_sig3_FN = Unemployment_sig_SVM_matrix3[1][0]
Unemployment_SVM_sig3_TP = Unemployment_sig_SVM_matrix3[1][1]
Unemployment_SVM_sig3_TN = Unemployment_sig_SVM_matrix3[0][0]

# Overall accuracy
Unemployment_SVM_sig3_ACC = (Unemployment_SVM_sig3_TP + Unemployment_SVM_sig3_TN)/(Unemployment_SVM_sig3_TP + Unemployment_SVM_sig3_FP + Unemployment_SVM_sig3_FN + Unemployment_SVM_sig3_TN)
print(Unemployment_SVM_sig3_ACC)

UnemploymentAccuracyDict.update({'Unemployment_SVM_sig3_ACC': Unemployment_SVM_sig3_ACC})
print(UnemploymentAccuracyDict)

#############################################
################  Poly #1 ###################
#############################################

Unemployment_poly_SVM_Model=sklearn.svm.SVC(C=1000, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(Unemployment_poly_SVM_Model)
Unemployment_poly_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
Unemployment_poly_svm_predict = Unemployment_poly_SVM_Model.predict(TestDFB_nolabels)
print("\npoly SVM prediction:\n")
print(Unemployment_poly_svm_predict)
print("\nActual:")
print(TestLabelsB)

Unemployment_poly_SVM_matrix = confusion_matrix(TestLabelsB, Unemployment_poly_svm_predict)
print("\nThe poly confusion matrix is:")
print(Unemployment_poly_SVM_matrix)
print("\n\n")

Unemployment_svm_poly_target_names = ['0','1']
print(classification_report(TestLabelsB, Unemployment_poly_svm_predict, target_names = Unemployment_svm_poly_target_names))

Unemployment_SVM_poly_FP = Unemployment_poly_SVM_matrix[0][1] 
Unemployment_SVM_poly_FN = Unemployment_poly_SVM_matrix[1][0]
Unemployment_SVM_poly_TP = Unemployment_poly_SVM_matrix[1][1]
Unemployment_SVM_poly_TN = Unemployment_poly_SVM_matrix[0][0]

# Overall accuracy
Unemployment_SVM_poly_ACC = (Unemployment_SVM_poly_TP + Unemployment_SVM_poly_TN)/(Unemployment_SVM_poly_TP + Unemployment_SVM_poly_FP + Unemployment_SVM_poly_FN + Unemployment_SVM_poly_TN)
print(Unemployment_SVM_poly_ACC)

UnemploymentAccuracyDict.update({'Unemployment_SVM_poly_ACC': Unemployment_SVM_poly_ACC})
print(UnemploymentAccuracyDict)

#############################################
################  Poly #2 ###################
#############################################

Unemployment_poly_SVM_Model2=sklearn.svm.SVC(C=10, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(Unemployment_poly_SVM_Model2)
Unemployment_poly_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
Unemployment_poly_svm_predict2 = Unemployment_poly_SVM_Model2.predict(TestDFB_nolabels)
print("\npoly SVM 2 prediction:\n")
print(Unemployment_poly_svm_predict2)
print("\nActual:")
print(TestLabelsB)

Unemployment_poly_SVM_matrix2 = confusion_matrix(TestLabelsB, Unemployment_poly_svm_predict2)
print("\nThe poly 2 confusion matrix is:")
print(Unemployment_poly_SVM_matrix2)
print("\n\n")

Unemployment_svm_poly_target_names2 = ['0','1']
print(classification_report(TestLabelsB, Unemployment_poly_svm_predict2, target_names = Unemployment_svm_poly_target_names2))

Unemployment_SVM_poly2_FP = Unemployment_poly_SVM_matrix2[0][1] 
Unemployment_SVM_poly2_FN = Unemployment_poly_SVM_matrix2[1][0]
Unemployment_SVM_poly2_TP = Unemployment_poly_SVM_matrix2[1][1]
Unemployment_SVM_poly2_TN = Unemployment_poly_SVM_matrix2[0][0]

# Overall accuracy
Unemployment_SVM_poly2_ACC = (Unemployment_SVM_poly2_TP + Unemployment_SVM_poly2_TN)/(Unemployment_SVM_poly2_TP + Unemployment_SVM_poly2_FP + Unemployment_SVM_poly2_FN + Unemployment_SVM_poly2_TN)
print(Unemployment_SVM_poly2_ACC)

UnemploymentAccuracyDict.update({'Unemployment_SVM_poly2_ACC': Unemployment_SVM_poly2_ACC})
print(UnemploymentAccuracyDict)

#############################################
################  Poly #3 ###################
#############################################

Unemployment_poly_SVM_Model3=sklearn.svm.SVC(C=.001, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(Unemployment_poly_SVM_Model3)
Unemployment_poly_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
Unemployment_poly_svm_predict3 = Unemployment_poly_SVM_Model3.predict(TestDFB_nolabels)
print("\npoly SVM 3 prediction:\n")
print(Unemployment_poly_svm_predict3)
print("\nActual:")
print(TestLabelsB)

Unemployment_poly_SVM_matrix3 = confusion_matrix(TestLabelsB, Unemployment_poly_svm_predict3)
print("\nThe poly 3 confusion matrix is:")
print(Unemployment_poly_SVM_matrix3)
print("\n\n")

Unemployment_svm_poly_target_names3 = ['0','1']
print(classification_report(TestLabelsB, Unemployment_poly_svm_predict3, target_names = Unemployment_svm_poly_target_names3))

Unemployment_SVM_poly3_FP = Unemployment_poly_SVM_matrix3[0][1] 
Unemployment_SVM_poly3_FN = Unemployment_poly_SVM_matrix3[1][0]
Unemployment_SVM_poly3_TP = Unemployment_poly_SVM_matrix3[1][1]
Unemployment_SVM_poly3_TN = Unemployment_poly_SVM_matrix3[0][0]

# Overall accuracy
Unemployment_SVM_poly3_ACC = (Unemployment_SVM_poly3_TP + Unemployment_SVM_poly3_TN)/(Unemployment_SVM_poly3_TP + Unemployment_SVM_poly3_FP + Unemployment_SVM_poly3_FN + Unemployment_SVM_poly3_TN)
print(Unemployment_SVM_poly3_ACC)

UnemploymentAccuracyDict.update({'Unemployment_SVM_poly3_ACC': Unemployment_SVM_poly3_ACC})
print(UnemploymentAccuracyDict)

UnemploymentVisDF = pd.DataFrame(UnemploymentAccuracyDict.items(), index = UnemploymentAccuracyDict.keys(), columns=['Model','Accuracy'])
print(UnemploymentVisDF)
SortedUnemploymentVisDF = UnemploymentVisDF.sort_values('Accuracy', ascending = [True])
print(SortedUnemploymentVisDF)

print(PartyAccuracyDict)
print(WinnerAccuracyDict)
print(SentimentAccuracyDict)
print(IncumCandAccuracyDict)
print(IncumPartyAccuracyDict)
print(UnemploymentAccuracyDict)

SortedPartyVisDF.plot.barh(y='Accuracy')
SortedWinnerVisDF.plot.barh(y='Accuracy')
SortedSentimentVisDF.plot.barh(y='Accuracy')
SortedIncumCandVisDF.plot.barh(y='Accuracy')
SortedIncumPartyVisDF.plot.barh(y='Accuracy')
SortedUnemploymentVisDF.plot.barh(y='Accuracy')

#print(IndexList)
#print(TextList)
#print(PartyList)
#print(WinnerList)
#print(SentimentList)
#print(IncumCandList)
#print(IncumPartyList)
#print(UnemploymentList)
print(GDPList)
print(InflationList)
print(SatisfactionList)
print(RealIncomeGrowthList)
print(PresApprovalList)
print(IncomeTaxList)
print(DjiaVolumeList)
print(CpiList)
print(CciIndexList)

#################################################################################################
#################################################################################################
### everything from this point down is a matter of find and replace for each of the variables ###
#################################################################################################
#################################################################################################

### starting to build a model for GDP
GDPVectDF = VectDF.copy(deep=True)
GDPVectDF.insert(loc=0, column='LABEL', value=GDPList)
print(GDPVectDF)

bool_GDPVectDF = bool_VectDF.copy(deep=True)
bool_GDPVectDF.insert(loc=0, column='LABEL', value=GDPList)
print(bool_GDPVectDF)

tf_GDPVectDF = tf_VectDF.copy(deep=True)
tf_GDPVectDF.insert(loc=0, column='LABEL', value=GDPList)
print(tf_GDPVectDF)


##--------------------------------------------------------
####################################################################
###################### model Data Build ############################
####################################################################
##--------------------------------------------------------
from sklearn.model_selection import train_test_split

## Copy everything below here to rebuild for GDP data
TrainDF, TestDF = train_test_split(GDPVectDF, test_size=0.3)
print(TrainDF.head())

## Now we have a training set and a testing set. 
print("\nThe training set is:")
print(TrainDF)
print("\nThe testing set is:")
print(TestDF)

## IMPORTANT - YOU CANNOT LEAVE LABELS ON THE TEST SET
## Save labels
TestLabels=TestDF["LABEL"]
print(TestLabels)
## remove labels
TestDF_nolabels = TestDF.drop(["LABEL"], axis=1)
print(TestDF_nolabels)

TrainDF_nolabels=TrainDF.drop(["LABEL"], axis=1)
print(TrainDF_nolabels)

TrainLabels=TrainDF["LABEL"]
print(TrainLabels)

############# Do the same for the binary dataframe
## Create Train/Test for TrainDFB
TrainDFB, TestDFB = train_test_split(bool_GDPVectDF, test_size=0.3)
print(TrainDFB.head())

TrainDFB_nolabels=TrainDFB.drop(["LABEL"], axis=1)
print(TrainDFB_nolabels)
TrainLabelsB=TrainDFB["LABEL"]
print(TrainLabelsB)

TestDFB_nolabels=TestDFB.drop(["LABEL"], axis=1)
print(TestDFB_nolabels)
TestLabelsB=TestDFB["LABEL"]
print(TestLabelsB)

############# Do the same for the tfidf dataframe
## Create Train/Test for TrainDF tfidf
TrainDF_tf, TestDF_tf = train_test_split(tf_GDPVectDF, test_size=0.3)
print(TrainDF_tf.head())

TrainDF_tf_nolabels=TrainDF_tf.drop(["LABEL"], axis=1)
print(TrainDF_tf_nolabels)
TrainLabels_tf=TrainDF_tf["LABEL"]
print(TrainLabels_tf)

TestDF_tf_nolabels=TestDF_tf.drop(["LABEL"], axis=1)
print(TestDF_tf_nolabels)
TestLabels_tf=TestDF_tf["LABEL"]
print(TestLabels_tf)



#############################################
###########  SVM  for regular model #1 ######
#############################################
from sklearn.svm import LinearSVC
print(TrainDF_nolabels.head())
print(TrainLabels.head())

GDP_SVM_Model=LinearSVC(C=.01)
GDP_SVM_Model.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
GDP_svm_predict = GDP_SVM_Model.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(GDP_svm_predict)
print("Actual:")
print(TestLabels)

from sklearn.metrics import confusion_matrix
GDP_SVM_matrix = confusion_matrix(TestLabels, GDP_svm_predict)
print("\nThe confusion matrix is:")
print(GDP_SVM_matrix)
print("\n\n")

from sklearn.metrics import classification_report
GDP_svm_target_names = ['0','1']
print(classification_report(TestLabels, GDP_svm_predict, target_names = GDP_svm_target_names))

GDP_SVM_reg_FP = GDP_SVM_matrix[0][1] 
GDP_SVM_reg_FN = GDP_SVM_matrix[1][0]
GDP_SVM_reg_TP = GDP_SVM_matrix[1][1]
GDP_SVM_reg_TN = GDP_SVM_matrix[0][0]

# Overall accuracy
GDP_SVM_reg_ACC = (GDP_SVM_reg_TP + GDP_SVM_reg_TN)/(GDP_SVM_reg_TP + GDP_SVM_reg_FP + GDP_SVM_reg_FN + GDP_SVM_reg_TN)
print(GDP_SVM_reg_ACC)

GDPAccuracyDict = {}
GDPAccuracyDict.update({'GDP_SVM_reg_ACC': GDP_SVM_reg_ACC})
print(GDPAccuracyDict)

#############################################
###########  SVM  for regular model #2 ######
#############################################

GDP_SVM_Model2=LinearSVC(C=1)
GDP_SVM_Model2.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
GDP_svm_predict2 = GDP_SVM_Model2.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(GDP_svm_predict2)
print("Actual:")
print(TestLabels)

GDP_SVM_matrix2 = confusion_matrix(TestLabels, GDP_svm_predict2)
print("\nThe confusion matrix is:")
print(GDP_SVM_matrix2)
print("\n\n")

GDP_svm_target_names2 = ['0','1']
print(classification_report(TestLabels, GDP_svm_predict2, target_names = GDP_svm_target_names2))

GDP_SVM_reg2_FP = GDP_SVM_matrix2[0][1] 
GDP_SVM_reg2_FN = GDP_SVM_matrix2[1][0]
GDP_SVM_reg2_TP = GDP_SVM_matrix2[1][1]
GDP_SVM_reg2_TN = GDP_SVM_matrix2[0][0]

# Overall accuracy
GDP_SVM_reg2_ACC = (GDP_SVM_reg2_TP + GDP_SVM_reg2_TN)/(GDP_SVM_reg2_TP + GDP_SVM_reg2_FP + GDP_SVM_reg2_FN + GDP_SVM_reg2_TN)
print(GDP_SVM_reg2_ACC)

GDPAccuracyDict.update({'GDP_SVM_reg2_ACC': GDP_SVM_reg2_ACC})
print(GDPAccuracyDict)

#############################################
###########  SVM  for regular model #3 ######
#############################################

GDP_SVM_Model3=LinearSVC(C=100)
GDP_SVM_Model3.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
GDP_svm_predict3 = GDP_SVM_Model3.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(GDP_svm_predict3)
print("Actual:")
print(TestLabels)

GDP_SVM_matrix3 = confusion_matrix(TestLabels, GDP_svm_predict3)
print("\nThe confusion matrix is:")
print(GDP_SVM_matrix3)
print("\n\n")

from sklearn.metrics import classification_report
GDP_svm_target_names3 = ['0','1']
print(classification_report(TestLabels, GDP_svm_predict3, target_names = GDP_svm_target_names3))

GDP_SVM_reg3_FP = GDP_SVM_matrix3[0][1] 
GDP_SVM_reg3_FN = GDP_SVM_matrix3[1][0]
GDP_SVM_reg3_TP = GDP_SVM_matrix3[1][1]
GDP_SVM_reg3_TN = GDP_SVM_matrix3[0][0]

# Overall accuracy
GDP_SVM_reg3_ACC = (GDP_SVM_reg3_TP + GDP_SVM_reg3_TN)/(GDP_SVM_reg3_TP + GDP_SVM_reg3_FP + GDP_SVM_reg3_FN + GDP_SVM_reg3_TN)
print(GDP_SVM_reg3_ACC)

GDPAccuracyDict.update({'GDP_SVM_reg3_ACC': GDP_SVM_reg3_ACC})
print(GDPAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #1 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

GDP_B_SVM_Model=LinearSVC(C=100)
GDP_B_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
GDP_b_svm_predict = GDP_B_SVM_Model.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(GDP_b_svm_predict)
print("\nActual:")
print(TestLabelsB)

GDP_B_SVM_matrix = confusion_matrix(TestLabelsB, GDP_b_svm_predict)
print("\nThe confusion matrix is:")
print(GDP_B_SVM_matrix)
print("\n\n")

GDP_svm_B_target_names = ['0','1']
print(classification_report(TestLabelsB, GDP_b_svm_predict, target_names = GDP_svm_B_target_names))

GDP_SVM_bool_FP = GDP_B_SVM_matrix[0][1] 
GDP_SVM_bool_FN = GDP_B_SVM_matrix[1][0]
GDP_SVM_bool_TP = GDP_B_SVM_matrix[1][1]
GDP_SVM_bool_TN = GDP_B_SVM_matrix[0][0]

# Overall accuracy
GDP_SVM_bool_ACC = (GDP_SVM_bool_TP + GDP_SVM_bool_TN)/(GDP_SVM_bool_TP + GDP_SVM_bool_FP + GDP_SVM_bool_FN + GDP_SVM_bool_TN)
print(GDP_SVM_bool_ACC)

GDPAccuracyDict.update({'GDP_SVM_bool_ACC': GDP_SVM_bool_ACC})
print(GDPAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #2 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

GDP_B_SVM_Model2=LinearSVC(C=1)
GDP_B_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
GDP_b_svm_predict2 = GDP_B_SVM_Model2.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(GDP_b_svm_predict2)
print("\nActual:")
print(TestLabelsB)

GDP_B_SVM_matrix2 = confusion_matrix(TestLabelsB, GDP_b_svm_predict2)
print("\nThe confusion matrix is:")
print(GDP_B_SVM_matrix2)
print("\n\n")

GDP_svm_B_target_names2 = ['0','1']
print(classification_report(TestLabelsB, GDP_b_svm_predict2, target_names = GDP_svm_B_target_names2))

GDP_SVM_bool2_FP = GDP_B_SVM_matrix2[0][1] 
GDP_SVM_bool2_FN = GDP_B_SVM_matrix2[1][0]
GDP_SVM_bool2_TP = GDP_B_SVM_matrix2[1][1]
GDP_SVM_bool2_TN = GDP_B_SVM_matrix2[0][0]

# Overall accuracy
GDP_SVM_bool2_ACC = (GDP_SVM_bool2_TP + GDP_SVM_bool2_TN)/(GDP_SVM_bool2_TP + GDP_SVM_bool2_FP + GDP_SVM_bool2_FN + GDP_SVM_bool2_TN)
print(GDP_SVM_bool2_ACC)

GDPAccuracyDict.update({'GDP_SVM_bool2_ACC': GDP_SVM_bool2_ACC})
print(GDPAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #3 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

GDP_B_SVM_Model3=LinearSVC(C=.01)
GDP_B_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
GDP_b_svm_predict3 = GDP_B_SVM_Model3.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(GDP_b_svm_predict3)
print("\nActual:")
print(TestLabelsB)

GDP_B_SVM_matrix3 = confusion_matrix(TestLabelsB, GDP_b_svm_predict3)
print("\nThe confusion matrix is:")
print(GDP_B_SVM_matrix3)
print("\n\n")

GDP_svm_B_target_names3 = ['0','1']
print(classification_report(TestLabelsB, GDP_b_svm_predict3, target_names = GDP_svm_B_target_names3))

GDP_SVM_bool3_FP = GDP_B_SVM_matrix3[0][1] 
GDP_SVM_bool3_FN = GDP_B_SVM_matrix3[1][0]
GDP_SVM_bool3_TP = GDP_B_SVM_matrix3[1][1]
GDP_SVM_bool3_TN = GDP_B_SVM_matrix3[0][0]

# Overall accuracy
GDP_SVM_bool3_ACC = (GDP_SVM_bool3_TP + GDP_SVM_bool3_TN)/(GDP_SVM_bool3_TP + GDP_SVM_bool3_FP + GDP_SVM_bool3_FN + GDP_SVM_bool3_TN)
print(GDP_SVM_bool3_ACC)

GDPAccuracyDict.update({'GDP_SVM_bool3_ACC': GDP_SVM_bool3_ACC})
print(GDPAccuracyDict)

#############################################
###########  SVM  for tfidf model #1 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

GDP_tf_SVM_Model=LinearSVC(C=.001)
GDP_tf_SVM_Model.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
GDP_tf_svm_predict = GDP_tf_SVM_Model.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(GDP_tf_svm_predict)
print("\nActual:")
print(TestLabels_tf)

GDP_tf_SVM_matrix = confusion_matrix(TestLabels_tf, GDP_tf_svm_predict)
print("\nThe confusion matrix is:")
print(GDP_tf_SVM_matrix)
print("\n\n")

GDP_svm_tf_target_names = ['0','1']
print(classification_report(TestLabels_tf, GDP_tf_svm_predict, target_names = GDP_svm_tf_target_names))

GDP_SVM_tf_FP = GDP_tf_SVM_matrix[0][1] 
GDP_SVM_tf_FN = GDP_tf_SVM_matrix[1][0]
GDP_SVM_tf_TP = GDP_tf_SVM_matrix[1][1]
GDP_SVM_tf_TN = GDP_tf_SVM_matrix[0][0]

# Overall accuracy
GDP_SVM_tf_ACC = (GDP_SVM_tf_TP + GDP_SVM_tf_TN)/(GDP_SVM_tf_TP + GDP_SVM_tf_FP + GDP_SVM_tf_FN + GDP_SVM_tf_TN)
print(GDP_SVM_tf_ACC)

GDPAccuracyDict.update({'GDP_SVM_tf_ACC': GDP_SVM_tf_ACC})
print(GDPAccuracyDict)

#############################################
###########  SVM  for tfidf model #2 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

GDP_tf_SVM_Model2=LinearSVC(C=1)
GDP_tf_SVM_Model2.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
GDP_tf_svm_predict2 = GDP_tf_SVM_Model2.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(GDP_tf_svm_predict2)
print("\nActual:")
print(TestLabels_tf)

GDP_tf_SVM_matrix2 = confusion_matrix(TestLabels_tf, GDP_tf_svm_predict2)
print("\nThe confusion matrix is:")
print(GDP_tf_SVM_matrix2)
print("\n\n")

GDP_svm_tf_target_names2 = ['0','1']
print(classification_report(TestLabels_tf, GDP_tf_svm_predict2, target_names = GDP_svm_tf_target_names2))

GDP_SVM_tf2_FP = GDP_tf_SVM_matrix2[0][1] 
GDP_SVM_tf2_FN = GDP_tf_SVM_matrix2[1][0]
GDP_SVM_tf2_TP = GDP_tf_SVM_matrix2[1][1]
GDP_SVM_tf2_TN = GDP_tf_SVM_matrix2[0][0]

# Overall accuracy
GDP_SVM_tf2_ACC = (GDP_SVM_tf2_TP + GDP_SVM_tf2_TN)/(GDP_SVM_tf2_TP + GDP_SVM_tf2_FP + GDP_SVM_tf2_FN + GDP_SVM_tf2_TN)
print(GDP_SVM_tf2_ACC)

GDPAccuracyDict.update({'GDP_SVM_tf2_ACC': GDP_SVM_tf2_ACC})
print(GDPAccuracyDict)

#############################################
###########  SVM  for tfidf model #3 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

GDP_tf_SVM_Model3=LinearSVC(C=1)
GDP_tf_SVM_Model3.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
GDP_tf_svm_predict3 = GDP_tf_SVM_Model3.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(GDP_tf_svm_predict3)
print("\nActual:")
print(TestLabels_tf)

GDP_tf_SVM_matrix3 = confusion_matrix(TestLabels_tf, GDP_tf_svm_predict3)
print("\nThe confusion matrix is:")
print(GDP_tf_SVM_matrix3)
print("\n\n")

GDP_svm_tf_target_names3 = ['0','1']
print(classification_report(TestLabels_tf, GDP_tf_svm_predict3, target_names = GDP_svm_tf_target_names3))

GDP_SVM_tf3_FP = GDP_tf_SVM_matrix3[0][1] 
GDP_SVM_tf3_FN = GDP_tf_SVM_matrix3[1][0]
GDP_SVM_tf3_TP = GDP_tf_SVM_matrix3[1][1]
GDP_SVM_tf3_TN = GDP_tf_SVM_matrix3[0][0]

# Overall accuracy
GDP_SVM_tf3_ACC = (GDP_SVM_tf3_TP + GDP_SVM_tf3_TN)/(GDP_SVM_tf3_TP + GDP_SVM_tf3_FP + GDP_SVM_tf3_FN + GDP_SVM_tf3_TN)
print(GDP_SVM_tf3_ACC)

GDPAccuracyDict.update({'GDP_SVM_tf3_ACC': GDP_SVM_tf3_ACC})
print(GDPAccuracyDict)

#############################################
#######  SVM  with different kernels ########
#############################################
## running on Bernoulli model since it had best accuracy above
## i want to see if the accuracy gets improved in the different kernels

#############################################
###############  Sigmoid #1 #################
#############################################

GDP_sig_SVM_Model=sklearn.svm.SVC(C=10, kernel='sigmoid', 
                           verbose=True, gamma="scale")
GDP_sig_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
GDP_sig_svm_predict = GDP_sig_SVM_Model.predict(TestDFB_nolabels)
print("\nsig SVM prediction:\n")
print(GDP_sig_svm_predict)
print("\nActual:")
print(TestLabelsB)

GDP_sig_SVM_matrix = confusion_matrix(TestLabelsB, GDP_sig_svm_predict)
print("\nThe sig confusion matrix is:")
print(GDP_sig_SVM_matrix)
print("\n\n")

GDP_svm_sig_target_names = ['0','1']
print(classification_report(TestLabelsB, GDP_sig_svm_predict, target_names = GDP_svm_sig_target_names))

GDP_SVM_sig_FP = GDP_sig_SVM_matrix[0][1] 
GDP_SVM_sig_FN = GDP_sig_SVM_matrix[1][0]
GDP_SVM_sig_TP = GDP_sig_SVM_matrix[1][1]
GDP_SVM_sig_TN = GDP_sig_SVM_matrix[0][0]

# Overall accuracy
GDP_SVM_sig_ACC = (GDP_SVM_sig_TP + GDP_SVM_sig_TN)/(GDP_SVM_sig_TP + GDP_SVM_sig_FP + GDP_SVM_sig_FN + GDP_SVM_sig_TN)
print(GDP_SVM_sig_ACC)

GDPAccuracyDict.update({'GDP_SVM_sig_ACC': GDP_SVM_sig_ACC})
print(GDPAccuracyDict)

#############################################
###############  Sigmoid #2 #################
#############################################

GDP_sig_SVM_Model2=sklearn.svm.SVC(C=1000, kernel='sigmoid', 
                           verbose=True, gamma="scale")
GDP_sig_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
GDP_sig_svm_predict2 = GDP_sig_SVM_Model2.predict(TestDFB_nolabels)
print("\nsig SVM prediction 2:\n")
print(GDP_sig_svm_predict2)
print("\nActual:")
print(TestLabelsB)

GDP_sig_SVM_matrix2 = confusion_matrix(TestLabelsB, GDP_sig_svm_predict2)
print("\nThe sig 2 confusion matrix is:")
print(GDP_sig_SVM_matrix2)
print("\n\n")

GDP_svm_sig_target_names2 = ['0','1']
print(classification_report(TestLabelsB, GDP_sig_svm_predict2, target_names = GDP_svm_sig_target_names2))

GDP_SVM_sig2_FP = GDP_sig_SVM_matrix2[0][1] 
GDP_SVM_sig2_FN = GDP_sig_SVM_matrix2[1][0]
GDP_SVM_sig2_TP = GDP_sig_SVM_matrix2[1][1]
GDP_SVM_sig2_TN = GDP_sig_SVM_matrix2[0][0]

# Overall accuracy
GDP_SVM_sig2_ACC = (GDP_SVM_sig2_TP + GDP_SVM_sig2_TN)/(GDP_SVM_sig2_TP + GDP_SVM_sig2_FP + GDP_SVM_sig2_FN + GDP_SVM_sig2_TN)
print(GDP_SVM_sig2_ACC)

GDPAccuracyDict.update({'GDP_SVM_sig2_ACC': GDP_SVM_sig2_ACC})
print(GDPAccuracyDict)

#############################################
###############  Sigmoid #3 #################
#############################################

GDP_sig_SVM_Model3=sklearn.svm.SVC(C=.001, kernel='sigmoid', 
                           verbose=True, gamma="scale")
GDP_sig_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
GDP_sig_svm_predict3 = GDP_sig_SVM_Model3.predict(TestDFB_nolabels)
print("\nsig SVM prediction 3:\n")
print(GDP_sig_svm_predict3)
print("\nActual:")
print(TestLabelsB)

GDP_sig_SVM_matrix3 = confusion_matrix(TestLabelsB, GDP_sig_svm_predict3)
print("\nThe sig 3 confusion matrix is:")
print(GDP_sig_SVM_matrix3)
print("\n\n")

GDP_svm_sig_target_names3 = ['0','1']
print(classification_report(TestLabelsB, GDP_sig_svm_predict3, target_names = GDP_svm_sig_target_names3))

GDP_SVM_sig3_FP = GDP_sig_SVM_matrix3[0][1] 
GDP_SVM_sig3_FN = GDP_sig_SVM_matrix3[1][0]
GDP_SVM_sig3_TP = GDP_sig_SVM_matrix3[1][1]
GDP_SVM_sig3_TN = GDP_sig_SVM_matrix3[0][0]

# Overall accuracy
GDP_SVM_sig3_ACC = (GDP_SVM_sig3_TP + GDP_SVM_sig3_TN)/(GDP_SVM_sig3_TP + GDP_SVM_sig3_FP + GDP_SVM_sig3_FN + GDP_SVM_sig3_TN)
print(GDP_SVM_sig3_ACC)

GDPAccuracyDict.update({'GDP_SVM_sig3_ACC': GDP_SVM_sig3_ACC})
print(GDPAccuracyDict)

#############################################
################  Poly #1 ###################
#############################################

GDP_poly_SVM_Model=sklearn.svm.SVC(C=1000, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(GDP_poly_SVM_Model)
GDP_poly_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
GDP_poly_svm_predict = GDP_poly_SVM_Model.predict(TestDFB_nolabels)
print("\npoly SVM prediction:\n")
print(GDP_poly_svm_predict)
print("\nActual:")
print(TestLabelsB)

GDP_poly_SVM_matrix = confusion_matrix(TestLabelsB, GDP_poly_svm_predict)
print("\nThe poly confusion matrix is:")
print(GDP_poly_SVM_matrix)
print("\n\n")

GDP_svm_poly_target_names = ['0','1']
print(classification_report(TestLabelsB, GDP_poly_svm_predict, target_names = GDP_svm_poly_target_names))

GDP_SVM_poly_FP = GDP_poly_SVM_matrix[0][1] 
GDP_SVM_poly_FN = GDP_poly_SVM_matrix[1][0]
GDP_SVM_poly_TP = GDP_poly_SVM_matrix[1][1]
GDP_SVM_poly_TN = GDP_poly_SVM_matrix[0][0]

# Overall accuracy
GDP_SVM_poly_ACC = (GDP_SVM_poly_TP + GDP_SVM_poly_TN)/(GDP_SVM_poly_TP + GDP_SVM_poly_FP + GDP_SVM_poly_FN + GDP_SVM_poly_TN)
print(GDP_SVM_poly_ACC)

GDPAccuracyDict.update({'GDP_SVM_poly_ACC': GDP_SVM_poly_ACC})
print(GDPAccuracyDict)

#############################################
################  Poly #2 ###################
#############################################

GDP_poly_SVM_Model2=sklearn.svm.SVC(C=10, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(GDP_poly_SVM_Model2)
GDP_poly_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
GDP_poly_svm_predict2 = GDP_poly_SVM_Model2.predict(TestDFB_nolabels)
print("\npoly SVM 2 prediction:\n")
print(GDP_poly_svm_predict2)
print("\nActual:")
print(TestLabelsB)

GDP_poly_SVM_matrix2 = confusion_matrix(TestLabelsB, GDP_poly_svm_predict2)
print("\nThe poly 2 confusion matrix is:")
print(GDP_poly_SVM_matrix2)
print("\n\n")

GDP_svm_poly_target_names2 = ['0','1']
print(classification_report(TestLabelsB, GDP_poly_svm_predict2, target_names = GDP_svm_poly_target_names2))

GDP_SVM_poly2_FP = GDP_poly_SVM_matrix2[0][1] 
GDP_SVM_poly2_FN = GDP_poly_SVM_matrix2[1][0]
GDP_SVM_poly2_TP = GDP_poly_SVM_matrix2[1][1]
GDP_SVM_poly2_TN = GDP_poly_SVM_matrix2[0][0]

# Overall accuracy
GDP_SVM_poly2_ACC = (GDP_SVM_poly2_TP + GDP_SVM_poly2_TN)/(GDP_SVM_poly2_TP + GDP_SVM_poly2_FP + GDP_SVM_poly2_FN + GDP_SVM_poly2_TN)
print(GDP_SVM_poly2_ACC)

GDPAccuracyDict.update({'GDP_SVM_poly2_ACC': GDP_SVM_poly2_ACC})
print(GDPAccuracyDict)

#############################################
################  Poly #3 ###################
#############################################

GDP_poly_SVM_Model3=sklearn.svm.SVC(C=.001, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(GDP_poly_SVM_Model3)
GDP_poly_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
GDP_poly_svm_predict3 = GDP_poly_SVM_Model3.predict(TestDFB_nolabels)
print("\npoly SVM 3 prediction:\n")
print(GDP_poly_svm_predict3)
print("\nActual:")
print(TestLabelsB)

GDP_poly_SVM_matrix3 = confusion_matrix(TestLabelsB, GDP_poly_svm_predict3)
print("\nThe poly 3 confusion matrix is:")
print(GDP_poly_SVM_matrix3)
print("\n\n")

GDP_svm_poly_target_names3 = ['0','1']
print(classification_report(TestLabelsB, GDP_poly_svm_predict3, target_names = GDP_svm_poly_target_names3))

GDP_SVM_poly3_FP = GDP_poly_SVM_matrix3[0][1] 
GDP_SVM_poly3_FN = GDP_poly_SVM_matrix3[1][0]
GDP_SVM_poly3_TP = GDP_poly_SVM_matrix3[1][1]
GDP_SVM_poly3_TN = GDP_poly_SVM_matrix3[0][0]

# Overall accuracy
GDP_SVM_poly3_ACC = (GDP_SVM_poly3_TP + GDP_SVM_poly3_TN)/(GDP_SVM_poly3_TP + GDP_SVM_poly3_FP + GDP_SVM_poly3_FN + GDP_SVM_poly3_TN)
print(GDP_SVM_poly3_ACC)

GDPAccuracyDict.update({'GDP_SVM_poly3_ACC': GDP_SVM_poly3_ACC})
print(GDPAccuracyDict)

GDPVisDF = pd.DataFrame(GDPAccuracyDict.items(), index = GDPAccuracyDict.keys(), columns=['Model','Accuracy'])
print(GDPVisDF)
SortedGDPVisDF = GDPVisDF.sort_values('Accuracy', ascending = [True])
print(SortedGDPVisDF)

print(PartyAccuracyDict)
print(WinnerAccuracyDict)
print(SentimentAccuracyDict)
print(IncumCandAccuracyDict)
print(IncumPartyAccuracyDict)
print(UnemploymentAccuracyDict)
print(GDPAccuracyDict)

SortedPartyVisDF.plot.barh(y='Accuracy')
SortedWinnerVisDF.plot.barh(y='Accuracy')
SortedSentimentVisDF.plot.barh(y='Accuracy')
SortedIncumCandVisDF.plot.barh(y='Accuracy')
SortedIncumPartyVisDF.plot.barh(y='Accuracy')
SortedUnemploymentVisDF.plot.barh(y='Accuracy')
SortedGDPVisDF.plot.barh(y='Accuracy')


#print(IndexList)
#print(TextList)
#print(PartyList)
#print(WinnerList)
#print(SentimentList)
#print(IncumCandList)
#print(IncumPartyList)
#print(UnemploymentList)
#print(GDPList)
print(InflationList)
print(SatisfactionList)
print(RealIncomeGrowthList)
print(PresApprovalList)
print(IncomeTaxList)
print(DjiaVolumeList)
print(CpiList)
print(CciIndexList)

#################################################################################################
#################################################################################################
### everything from this point down is a matter of find and replace for each of the variables ###
#################################################################################################
#################################################################################################

### starting to build a model for Inflation
InflationVectDF = VectDF.copy(deep=True)
InflationVectDF.insert(loc=0, column='LABEL', value=InflationList)
print(InflationVectDF)

bool_InflationVectDF = bool_VectDF.copy(deep=True)
bool_InflationVectDF.insert(loc=0, column='LABEL', value=InflationList)
print(bool_InflationVectDF)

tf_InflationVectDF = tf_VectDF.copy(deep=True)
tf_InflationVectDF.insert(loc=0, column='LABEL', value=InflationList)
print(tf_InflationVectDF)


##--------------------------------------------------------
####################################################################
###################### model Data Build ############################
####################################################################
##--------------------------------------------------------
from sklearn.model_selection import train_test_split

## Copy everything below here to rebuild for Inflation data
TrainDF, TestDF = train_test_split(InflationVectDF, test_size=0.3)
print(TrainDF.head())

## Now we have a training set and a testing set. 
print("\nThe training set is:")
print(TrainDF)
print("\nThe testing set is:")
print(TestDF)

## IMPORTANT - YOU CANNOT LEAVE LABELS ON THE TEST SET
## Save labels
TestLabels=TestDF["LABEL"]
print(TestLabels)
## remove labels
TestDF_nolabels = TestDF.drop(["LABEL"], axis=1)
print(TestDF_nolabels)

TrainDF_nolabels=TrainDF.drop(["LABEL"], axis=1)
print(TrainDF_nolabels)

TrainLabels=TrainDF["LABEL"]
print(TrainLabels)

############# Do the same for the binary dataframe
## Create Train/Test for TrainDFB
TrainDFB, TestDFB = train_test_split(bool_InflationVectDF, test_size=0.3)
print(TrainDFB.head())

TrainDFB_nolabels=TrainDFB.drop(["LABEL"], axis=1)
print(TrainDFB_nolabels)
TrainLabelsB=TrainDFB["LABEL"]
print(TrainLabelsB)

TestDFB_nolabels=TestDFB.drop(["LABEL"], axis=1)
print(TestDFB_nolabels)
TestLabelsB=TestDFB["LABEL"]
print(TestLabelsB)

############# Do the same for the tfidf dataframe
## Create Train/Test for TrainDF tfidf
TrainDF_tf, TestDF_tf = train_test_split(tf_InflationVectDF, test_size=0.3)
print(TrainDF_tf.head())

TrainDF_tf_nolabels=TrainDF_tf.drop(["LABEL"], axis=1)
print(TrainDF_tf_nolabels)
TrainLabels_tf=TrainDF_tf["LABEL"]
print(TrainLabels_tf)

TestDF_tf_nolabels=TestDF_tf.drop(["LABEL"], axis=1)
print(TestDF_tf_nolabels)
TestLabels_tf=TestDF_tf["LABEL"]
print(TestLabels_tf)



#############################################
###########  SVM  for regular model #1 ######
#############################################
from sklearn.svm import LinearSVC
print(TrainDF_nolabels.head())
print(TrainLabels.head())

Inflation_SVM_Model=LinearSVC(C=.01)
Inflation_SVM_Model.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Inflation_svm_predict = Inflation_SVM_Model.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(Inflation_svm_predict)
print("Actual:")
print(TestLabels)

from sklearn.metrics import confusion_matrix
Inflation_SVM_matrix = confusion_matrix(TestLabels, Inflation_svm_predict)
print("\nThe confusion matrix is:")
print(Inflation_SVM_matrix)
print("\n\n")

from sklearn.metrics import classification_report
Inflation_svm_target_names = ['0','1']
print(classification_report(TestLabels, Inflation_svm_predict, target_names = Inflation_svm_target_names))

Inflation_SVM_reg_FP = Inflation_SVM_matrix[0][1] 
Inflation_SVM_reg_FN = Inflation_SVM_matrix[1][0]
Inflation_SVM_reg_TP = Inflation_SVM_matrix[1][1]
Inflation_SVM_reg_TN = Inflation_SVM_matrix[0][0]

# Overall accuracy
Inflation_SVM_reg_ACC = (Inflation_SVM_reg_TP + Inflation_SVM_reg_TN)/(Inflation_SVM_reg_TP + Inflation_SVM_reg_FP + Inflation_SVM_reg_FN + Inflation_SVM_reg_TN)
print(Inflation_SVM_reg_ACC)

InflationAccuracyDict = {}
InflationAccuracyDict.update({'Inflation_SVM_reg_ACC': Inflation_SVM_reg_ACC})
print(InflationAccuracyDict)

#############################################
###########  SVM  for regular model #2 ######
#############################################

Inflation_SVM_Model2=LinearSVC(C=1)
Inflation_SVM_Model2.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Inflation_svm_predict2 = Inflation_SVM_Model2.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(Inflation_svm_predict2)
print("Actual:")
print(TestLabels)

Inflation_SVM_matrix2 = confusion_matrix(TestLabels, Inflation_svm_predict2)
print("\nThe confusion matrix is:")
print(Inflation_SVM_matrix2)
print("\n\n")

Inflation_svm_target_names2 = ['0','1']
print(classification_report(TestLabels, Inflation_svm_predict2, target_names = Inflation_svm_target_names2))

Inflation_SVM_reg2_FP = Inflation_SVM_matrix2[0][1] 
Inflation_SVM_reg2_FN = Inflation_SVM_matrix2[1][0]
Inflation_SVM_reg2_TP = Inflation_SVM_matrix2[1][1]
Inflation_SVM_reg2_TN = Inflation_SVM_matrix2[0][0]

# Overall accuracy
Inflation_SVM_reg2_ACC = (Inflation_SVM_reg2_TP + Inflation_SVM_reg2_TN)/(Inflation_SVM_reg2_TP + Inflation_SVM_reg2_FP + Inflation_SVM_reg2_FN + Inflation_SVM_reg2_TN)
print(Inflation_SVM_reg2_ACC)

InflationAccuracyDict.update({'Inflation_SVM_reg2_ACC': Inflation_SVM_reg2_ACC})
print(InflationAccuracyDict)

#############################################
###########  SVM  for regular model #3 ######
#############################################

Inflation_SVM_Model3=LinearSVC(C=100)
Inflation_SVM_Model3.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Inflation_svm_predict3 = Inflation_SVM_Model3.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(Inflation_svm_predict3)
print("Actual:")
print(TestLabels)

Inflation_SVM_matrix3 = confusion_matrix(TestLabels, Inflation_svm_predict3)
print("\nThe confusion matrix is:")
print(Inflation_SVM_matrix3)
print("\n\n")

from sklearn.metrics import classification_report
Inflation_svm_target_names3 = ['0','1']
print(classification_report(TestLabels, Inflation_svm_predict3, target_names = Inflation_svm_target_names3))

Inflation_SVM_reg3_FP = Inflation_SVM_matrix3[0][1] 
Inflation_SVM_reg3_FN = Inflation_SVM_matrix3[1][0]
Inflation_SVM_reg3_TP = Inflation_SVM_matrix3[1][1]
Inflation_SVM_reg3_TN = Inflation_SVM_matrix3[0][0]

# Overall accuracy
Inflation_SVM_reg3_ACC = (Inflation_SVM_reg3_TP + Inflation_SVM_reg3_TN)/(Inflation_SVM_reg3_TP + Inflation_SVM_reg3_FP + Inflation_SVM_reg3_FN + Inflation_SVM_reg3_TN)
print(Inflation_SVM_reg3_ACC)

InflationAccuracyDict.update({'Inflation_SVM_reg3_ACC': Inflation_SVM_reg3_ACC})
print(InflationAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #1 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

Inflation_B_SVM_Model=LinearSVC(C=100)
Inflation_B_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Inflation_b_svm_predict = Inflation_B_SVM_Model.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(Inflation_b_svm_predict)
print("\nActual:")
print(TestLabelsB)

Inflation_B_SVM_matrix = confusion_matrix(TestLabelsB, Inflation_b_svm_predict)
print("\nThe confusion matrix is:")
print(Inflation_B_SVM_matrix)
print("\n\n")

Inflation_svm_B_target_names = ['0','1']
print(classification_report(TestLabelsB, Inflation_b_svm_predict, target_names = Inflation_svm_B_target_names))

Inflation_SVM_bool_FP = Inflation_B_SVM_matrix[0][1] 
Inflation_SVM_bool_FN = Inflation_B_SVM_matrix[1][0]
Inflation_SVM_bool_TP = Inflation_B_SVM_matrix[1][1]
Inflation_SVM_bool_TN = Inflation_B_SVM_matrix[0][0]

# Overall accuracy
Inflation_SVM_bool_ACC = (Inflation_SVM_bool_TP + Inflation_SVM_bool_TN)/(Inflation_SVM_bool_TP + Inflation_SVM_bool_FP + Inflation_SVM_bool_FN + Inflation_SVM_bool_TN)
print(Inflation_SVM_bool_ACC)

InflationAccuracyDict.update({'Inflation_SVM_bool_ACC': Inflation_SVM_bool_ACC})
print(InflationAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #2 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

Inflation_B_SVM_Model2=LinearSVC(C=1)
Inflation_B_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Inflation_b_svm_predict2 = Inflation_B_SVM_Model2.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(Inflation_b_svm_predict2)
print("\nActual:")
print(TestLabelsB)

Inflation_B_SVM_matrix2 = confusion_matrix(TestLabelsB, Inflation_b_svm_predict2)
print("\nThe confusion matrix is:")
print(Inflation_B_SVM_matrix2)
print("\n\n")

Inflation_svm_B_target_names2 = ['0','1']
print(classification_report(TestLabelsB, Inflation_b_svm_predict2, target_names = Inflation_svm_B_target_names2))

Inflation_SVM_bool2_FP = Inflation_B_SVM_matrix2[0][1] 
Inflation_SVM_bool2_FN = Inflation_B_SVM_matrix2[1][0]
Inflation_SVM_bool2_TP = Inflation_B_SVM_matrix2[1][1]
Inflation_SVM_bool2_TN = Inflation_B_SVM_matrix2[0][0]

# Overall accuracy
Inflation_SVM_bool2_ACC = (Inflation_SVM_bool2_TP + Inflation_SVM_bool2_TN)/(Inflation_SVM_bool2_TP + Inflation_SVM_bool2_FP + Inflation_SVM_bool2_FN + Inflation_SVM_bool2_TN)
print(Inflation_SVM_bool2_ACC)

InflationAccuracyDict.update({'Inflation_SVM_bool2_ACC': Inflation_SVM_bool2_ACC})
print(InflationAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #3 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

Inflation_B_SVM_Model3=LinearSVC(C=.01)
Inflation_B_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Inflation_b_svm_predict3 = Inflation_B_SVM_Model3.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(Inflation_b_svm_predict3)
print("\nActual:")
print(TestLabelsB)

Inflation_B_SVM_matrix3 = confusion_matrix(TestLabelsB, Inflation_b_svm_predict3)
print("\nThe confusion matrix is:")
print(Inflation_B_SVM_matrix3)
print("\n\n")

Inflation_svm_B_target_names3 = ['0','1']
print(classification_report(TestLabelsB, Inflation_b_svm_predict3, target_names = Inflation_svm_B_target_names3))

Inflation_SVM_bool3_FP = Inflation_B_SVM_matrix3[0][1] 
Inflation_SVM_bool3_FN = Inflation_B_SVM_matrix3[1][0]
Inflation_SVM_bool3_TP = Inflation_B_SVM_matrix3[1][1]
Inflation_SVM_bool3_TN = Inflation_B_SVM_matrix3[0][0]

# Overall accuracy
Inflation_SVM_bool3_ACC = (Inflation_SVM_bool3_TP + Inflation_SVM_bool3_TN)/(Inflation_SVM_bool3_TP + Inflation_SVM_bool3_FP + Inflation_SVM_bool3_FN + Inflation_SVM_bool3_TN)
print(Inflation_SVM_bool3_ACC)

InflationAccuracyDict.update({'Inflation_SVM_bool3_ACC': Inflation_SVM_bool3_ACC})
print(InflationAccuracyDict)

#############################################
###########  SVM  for tfidf model #1 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

Inflation_tf_SVM_Model=LinearSVC(C=.001)
Inflation_tf_SVM_Model.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Inflation_tf_svm_predict = Inflation_tf_SVM_Model.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(Inflation_tf_svm_predict)
print("\nActual:")
print(TestLabels_tf)

Inflation_tf_SVM_matrix = confusion_matrix(TestLabels_tf, Inflation_tf_svm_predict)
print("\nThe confusion matrix is:")
print(Inflation_tf_SVM_matrix)
print("\n\n")

Inflation_svm_tf_target_names = ['0','1']
print(classification_report(TestLabels_tf, Inflation_tf_svm_predict, target_names = Inflation_svm_tf_target_names))

Inflation_SVM_tf_FP = Inflation_tf_SVM_matrix[0][1] 
Inflation_SVM_tf_FN = Inflation_tf_SVM_matrix[1][0]
Inflation_SVM_tf_TP = Inflation_tf_SVM_matrix[1][1]
Inflation_SVM_tf_TN = Inflation_tf_SVM_matrix[0][0]

# Overall accuracy
Inflation_SVM_tf_ACC = (Inflation_SVM_tf_TP + Inflation_SVM_tf_TN)/(Inflation_SVM_tf_TP + Inflation_SVM_tf_FP + Inflation_SVM_tf_FN + Inflation_SVM_tf_TN)
print(Inflation_SVM_tf_ACC)

InflationAccuracyDict.update({'Inflation_SVM_tf_ACC': Inflation_SVM_tf_ACC})
print(InflationAccuracyDict)

#############################################
###########  SVM  for tfidf model #2 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

Inflation_tf_SVM_Model2=LinearSVC(C=1)
Inflation_tf_SVM_Model2.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Inflation_tf_svm_predict2 = Inflation_tf_SVM_Model2.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(Inflation_tf_svm_predict2)
print("\nActual:")
print(TestLabels_tf)

Inflation_tf_SVM_matrix2 = confusion_matrix(TestLabels_tf, Inflation_tf_svm_predict2)
print("\nThe confusion matrix is:")
print(Inflation_tf_SVM_matrix2)
print("\n\n")

Inflation_svm_tf_target_names2 = ['0','1']
print(classification_report(TestLabels_tf, Inflation_tf_svm_predict2, target_names = Inflation_svm_tf_target_names2))

Inflation_SVM_tf2_FP = Inflation_tf_SVM_matrix2[0][1] 
Inflation_SVM_tf2_FN = Inflation_tf_SVM_matrix2[1][0]
Inflation_SVM_tf2_TP = Inflation_tf_SVM_matrix2[1][1]
Inflation_SVM_tf2_TN = Inflation_tf_SVM_matrix2[0][0]

# Overall accuracy
Inflation_SVM_tf2_ACC = (Inflation_SVM_tf2_TP + Inflation_SVM_tf2_TN)/(Inflation_SVM_tf2_TP + Inflation_SVM_tf2_FP + Inflation_SVM_tf2_FN + Inflation_SVM_tf2_TN)
print(Inflation_SVM_tf2_ACC)

InflationAccuracyDict.update({'Inflation_SVM_tf2_ACC': Inflation_SVM_tf2_ACC})
print(InflationAccuracyDict)

#############################################
###########  SVM  for tfidf model #3 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

Inflation_tf_SVM_Model3=LinearSVC(C=1)
Inflation_tf_SVM_Model3.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Inflation_tf_svm_predict3 = Inflation_tf_SVM_Model3.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(Inflation_tf_svm_predict3)
print("\nActual:")
print(TestLabels_tf)

Inflation_tf_SVM_matrix3 = confusion_matrix(TestLabels_tf, Inflation_tf_svm_predict3)
print("\nThe confusion matrix is:")
print(Inflation_tf_SVM_matrix3)
print("\n\n")

Inflation_svm_tf_target_names3 = ['0','1']
print(classification_report(TestLabels_tf, Inflation_tf_svm_predict3, target_names = Inflation_svm_tf_target_names3))

Inflation_SVM_tf3_FP = Inflation_tf_SVM_matrix3[0][1] 
Inflation_SVM_tf3_FN = Inflation_tf_SVM_matrix3[1][0]
Inflation_SVM_tf3_TP = Inflation_tf_SVM_matrix3[1][1]
Inflation_SVM_tf3_TN = Inflation_tf_SVM_matrix3[0][0]

# Overall accuracy
Inflation_SVM_tf3_ACC = (Inflation_SVM_tf3_TP + Inflation_SVM_tf3_TN)/(Inflation_SVM_tf3_TP + Inflation_SVM_tf3_FP + Inflation_SVM_tf3_FN + Inflation_SVM_tf3_TN)
print(Inflation_SVM_tf3_ACC)

InflationAccuracyDict.update({'Inflation_SVM_tf3_ACC': Inflation_SVM_tf3_ACC})
print(InflationAccuracyDict)

#############################################
#######  SVM  with different kernels ########
#############################################
## running on Bernoulli model since it had best accuracy above
## i want to see if the accuracy gets improved in the different kernels

#############################################
###############  Sigmoid #1 #################
#############################################

Inflation_sig_SVM_Model=sklearn.svm.SVC(C=10, kernel='sigmoid', 
                           verbose=True, gamma="scale")
Inflation_sig_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
Inflation_sig_svm_predict = Inflation_sig_SVM_Model.predict(TestDFB_nolabels)
print("\nsig SVM prediction:\n")
print(Inflation_sig_svm_predict)
print("\nActual:")
print(TestLabelsB)

Inflation_sig_SVM_matrix = confusion_matrix(TestLabelsB, Inflation_sig_svm_predict)
print("\nThe sig confusion matrix is:")
print(Inflation_sig_SVM_matrix)
print("\n\n")

Inflation_svm_sig_target_names = ['0','1']
print(classification_report(TestLabelsB, Inflation_sig_svm_predict, target_names = Inflation_svm_sig_target_names))

Inflation_SVM_sig_FP = Inflation_sig_SVM_matrix[0][1] 
Inflation_SVM_sig_FN = Inflation_sig_SVM_matrix[1][0]
Inflation_SVM_sig_TP = Inflation_sig_SVM_matrix[1][1]
Inflation_SVM_sig_TN = Inflation_sig_SVM_matrix[0][0]

# Overall accuracy
Inflation_SVM_sig_ACC = (Inflation_SVM_sig_TP + Inflation_SVM_sig_TN)/(Inflation_SVM_sig_TP + Inflation_SVM_sig_FP + Inflation_SVM_sig_FN + Inflation_SVM_sig_TN)
print(Inflation_SVM_sig_ACC)

InflationAccuracyDict.update({'Inflation_SVM_sig_ACC': Inflation_SVM_sig_ACC})
print(InflationAccuracyDict)

#############################################
###############  Sigmoid #2 #################
#############################################

Inflation_sig_SVM_Model2=sklearn.svm.SVC(C=1000, kernel='sigmoid', 
                           verbose=True, gamma="scale")
Inflation_sig_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
Inflation_sig_svm_predict2 = Inflation_sig_SVM_Model2.predict(TestDFB_nolabels)
print("\nsig SVM prediction 2:\n")
print(Inflation_sig_svm_predict2)
print("\nActual:")
print(TestLabelsB)

Inflation_sig_SVM_matrix2 = confusion_matrix(TestLabelsB, Inflation_sig_svm_predict2)
print("\nThe sig 2 confusion matrix is:")
print(Inflation_sig_SVM_matrix2)
print("\n\n")

Inflation_svm_sig_target_names2 = ['0','1']
print(classification_report(TestLabelsB, Inflation_sig_svm_predict2, target_names = Inflation_svm_sig_target_names2))

Inflation_SVM_sig2_FP = Inflation_sig_SVM_matrix2[0][1] 
Inflation_SVM_sig2_FN = Inflation_sig_SVM_matrix2[1][0]
Inflation_SVM_sig2_TP = Inflation_sig_SVM_matrix2[1][1]
Inflation_SVM_sig2_TN = Inflation_sig_SVM_matrix2[0][0]

# Overall accuracy
Inflation_SVM_sig2_ACC = (Inflation_SVM_sig2_TP + Inflation_SVM_sig2_TN)/(Inflation_SVM_sig2_TP + Inflation_SVM_sig2_FP + Inflation_SVM_sig2_FN + Inflation_SVM_sig2_TN)
print(Inflation_SVM_sig2_ACC)

InflationAccuracyDict.update({'Inflation_SVM_sig2_ACC': Inflation_SVM_sig2_ACC})
print(InflationAccuracyDict)

#############################################
###############  Sigmoid #3 #################
#############################################

Inflation_sig_SVM_Model3=sklearn.svm.SVC(C=.001, kernel='sigmoid', 
                           verbose=True, gamma="scale")
Inflation_sig_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
Inflation_sig_svm_predict3 = Inflation_sig_SVM_Model3.predict(TestDFB_nolabels)
print("\nsig SVM prediction 3:\n")
print(Inflation_sig_svm_predict3)
print("\nActual:")
print(TestLabelsB)

Inflation_sig_SVM_matrix3 = confusion_matrix(TestLabelsB, Inflation_sig_svm_predict3)
print("\nThe sig 3 confusion matrix is:")
print(Inflation_sig_SVM_matrix3)
print("\n\n")

Inflation_svm_sig_target_names3 = ['0','1']
print(classification_report(TestLabelsB, Inflation_sig_svm_predict3, target_names = Inflation_svm_sig_target_names3))

Inflation_SVM_sig3_FP = Inflation_sig_SVM_matrix3[0][1] 
Inflation_SVM_sig3_FN = Inflation_sig_SVM_matrix3[1][0]
Inflation_SVM_sig3_TP = Inflation_sig_SVM_matrix3[1][1]
Inflation_SVM_sig3_TN = Inflation_sig_SVM_matrix3[0][0]

# Overall accuracy
Inflation_SVM_sig3_ACC = (Inflation_SVM_sig3_TP + Inflation_SVM_sig3_TN)/(Inflation_SVM_sig3_TP + Inflation_SVM_sig3_FP + Inflation_SVM_sig3_FN + Inflation_SVM_sig3_TN)
print(Inflation_SVM_sig3_ACC)

InflationAccuracyDict.update({'Inflation_SVM_sig3_ACC': Inflation_SVM_sig3_ACC})
print(InflationAccuracyDict)

#############################################
################  Poly #1 ###################
#############################################

Inflation_poly_SVM_Model=sklearn.svm.SVC(C=1000, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(Inflation_poly_SVM_Model)
Inflation_poly_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
Inflation_poly_svm_predict = Inflation_poly_SVM_Model.predict(TestDFB_nolabels)
print("\npoly SVM prediction:\n")
print(Inflation_poly_svm_predict)
print("\nActual:")
print(TestLabelsB)

Inflation_poly_SVM_matrix = confusion_matrix(TestLabelsB, Inflation_poly_svm_predict)
print("\nThe poly confusion matrix is:")
print(Inflation_poly_SVM_matrix)
print("\n\n")

Inflation_svm_poly_target_names = ['0','1']
print(classification_report(TestLabelsB, Inflation_poly_svm_predict, target_names = Inflation_svm_poly_target_names))

Inflation_SVM_poly_FP = Inflation_poly_SVM_matrix[0][1] 
Inflation_SVM_poly_FN = Inflation_poly_SVM_matrix[1][0]
Inflation_SVM_poly_TP = Inflation_poly_SVM_matrix[1][1]
Inflation_SVM_poly_TN = Inflation_poly_SVM_matrix[0][0]

# Overall accuracy
Inflation_SVM_poly_ACC = (Inflation_SVM_poly_TP + Inflation_SVM_poly_TN)/(Inflation_SVM_poly_TP + Inflation_SVM_poly_FP + Inflation_SVM_poly_FN + Inflation_SVM_poly_TN)
print(Inflation_SVM_poly_ACC)

InflationAccuracyDict.update({'Inflation_SVM_poly_ACC': Inflation_SVM_poly_ACC})
print(InflationAccuracyDict)

#############################################
################  Poly #2 ###################
#############################################

Inflation_poly_SVM_Model2=sklearn.svm.SVC(C=10, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(Inflation_poly_SVM_Model2)
Inflation_poly_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
Inflation_poly_svm_predict2 = Inflation_poly_SVM_Model2.predict(TestDFB_nolabels)
print("\npoly SVM 2 prediction:\n")
print(Inflation_poly_svm_predict2)
print("\nActual:")
print(TestLabelsB)

Inflation_poly_SVM_matrix2 = confusion_matrix(TestLabelsB, Inflation_poly_svm_predict2)
print("\nThe poly 2 confusion matrix is:")
print(Inflation_poly_SVM_matrix2)
print("\n\n")

Inflation_svm_poly_target_names2 = ['0','1']
print(classification_report(TestLabelsB, Inflation_poly_svm_predict2, target_names = Inflation_svm_poly_target_names2))

Inflation_SVM_poly2_FP = Inflation_poly_SVM_matrix2[0][1] 
Inflation_SVM_poly2_FN = Inflation_poly_SVM_matrix2[1][0]
Inflation_SVM_poly2_TP = Inflation_poly_SVM_matrix2[1][1]
Inflation_SVM_poly2_TN = Inflation_poly_SVM_matrix2[0][0]

# Overall accuracy
Inflation_SVM_poly2_ACC = (Inflation_SVM_poly2_TP + Inflation_SVM_poly2_TN)/(Inflation_SVM_poly2_TP + Inflation_SVM_poly2_FP + Inflation_SVM_poly2_FN + Inflation_SVM_poly2_TN)
print(Inflation_SVM_poly2_ACC)

InflationAccuracyDict.update({'Inflation_SVM_poly2_ACC': Inflation_SVM_poly2_ACC})
print(InflationAccuracyDict)

#############################################
################  Poly #3 ###################
#############################################

Inflation_poly_SVM_Model3=sklearn.svm.SVC(C=.001, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(Inflation_poly_SVM_Model3)
Inflation_poly_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
Inflation_poly_svm_predict3 = Inflation_poly_SVM_Model3.predict(TestDFB_nolabels)
print("\npoly SVM 3 prediction:\n")
print(Inflation_poly_svm_predict3)
print("\nActual:")
print(TestLabelsB)

Inflation_poly_SVM_matrix3 = confusion_matrix(TestLabelsB, Inflation_poly_svm_predict3)
print("\nThe poly 3 confusion matrix is:")
print(Inflation_poly_SVM_matrix3)
print("\n\n")

Inflation_svm_poly_target_names3 = ['0','1']
print(classification_report(TestLabelsB, Inflation_poly_svm_predict3, target_names = Inflation_svm_poly_target_names3))

Inflation_SVM_poly3_FP = Inflation_poly_SVM_matrix3[0][1] 
Inflation_SVM_poly3_FN = Inflation_poly_SVM_matrix3[1][0]
Inflation_SVM_poly3_TP = Inflation_poly_SVM_matrix3[1][1]
Inflation_SVM_poly3_TN = Inflation_poly_SVM_matrix3[0][0]

# Overall accuracy
Inflation_SVM_poly3_ACC = (Inflation_SVM_poly3_TP + Inflation_SVM_poly3_TN)/(Inflation_SVM_poly3_TP + Inflation_SVM_poly3_FP + Inflation_SVM_poly3_FN + Inflation_SVM_poly3_TN)
print(Inflation_SVM_poly3_ACC)

InflationAccuracyDict.update({'Inflation_SVM_poly3_ACC': Inflation_SVM_poly3_ACC})
print(InflationAccuracyDict)

InflationVisDF = pd.DataFrame(InflationAccuracyDict.items(), index = InflationAccuracyDict.keys(), columns=['Model','Accuracy'])
print(InflationVisDF)
SortedInflationVisDF = InflationVisDF.sort_values('Accuracy', ascending = [True])
print(SortedInflationVisDF)

print(PartyAccuracyDict)
print(WinnerAccuracyDict)
print(SentimentAccuracyDict)
print(IncumCandAccuracyDict)
print(IncumPartyAccuracyDict)
print(UnemploymentAccuracyDict)
print(GDPAccuracyDict)
print(InflationAccuracyDict)

SortedPartyVisDF.plot.barh(y='Accuracy')
SortedWinnerVisDF.plot.barh(y='Accuracy')
SortedSentimentVisDF.plot.barh(y='Accuracy')
SortedIncumCandVisDF.plot.barh(y='Accuracy')
SortedIncumPartyVisDF.plot.barh(y='Accuracy')
SortedUnemploymentVisDF.plot.barh(y='Accuracy')
SortedGDPVisDF.plot.barh(y='Accuracy')
SortedInflationVisDF.plot.barh(y='Accuracy')


#print(IndexList)
#print(TextList)
#print(PartyList)
#print(WinnerList)
#print(SentimentList)
#print(IncumCandList)
#print(IncumPartyList)
#print(UnemploymentList)
#print(GDPList)
#print(InflationList)
print(SatisfactionList)
print(RealIncomeGrowthList)
print(PresApprovalList)
print(IncomeTaxList)
print(DjiaVolumeList)
print(CpiList)
print(CciIndexList)

#################################################################################################
#################################################################################################
### everything from this point down is a matter of find and replace for each of the variables ###
#################################################################################################
#################################################################################################

# the rows weren't all populated for the variables below so an edited text list is needed

### starting to build a model for Satisfaction
df_satisfaction = rawfile.copy(deep=True)
df_satisfaction = df_satisfaction.filter(['id', 'satisfaction', 'comb_text'])
df_satisfaction = df_satisfaction[df_satisfaction['satisfaction'].notna()]
print(df_satisfaction)

SatisfactionList = []
TextList = []
IndexList = []

for row in df_satisfaction.itertuples():
    satisfactionlabel = row.satisfaction
    textlabel = row.comb_text
    textlabel = textlabel.replace('\n',' ') # remove \n
    textlabel = textlabel.replace("\'","'")
    textlabel = textlabel.replace('\t',' ') # remove \t
    #textlabel = textlabel.replace(',',' ') # remove ,
    #textlabel = textlabel.replace('.',' ') # remove .
    #textlabel = textlabel.replace('"',' ') # remove "
    #textlabel = textlabel.replace("'",' ') # remove '
    textlabel = textlabel.replace('nan',' ') # remove nan
    #textlabel = textlabel.replace('-',' ') # remove -
    #textlabel = textlabel.replace(' - ',' ') # remove -
    #textlabel = textlabel.replace('?',' ') # remove ?
    #textlabel = textlabel.replace('!',' ') # remove !
    #textlabel = textlabel.replace(':',' ') # remove :
    #textlabel = textlabel.replace(';',' ') # remove ;
    #textlabel = textlabel.replace('  ',' ') # remove double spaces
    indexlabel = row.id
    TextList.append(textlabel)
    IndexList.append(indexlabel)
    SatisfactionList.append(satisfactionlabel)

SatisfactionList = [ int(x) for x in SatisfactionList ]
print(SatisfactionList)

X_text=MyVect.fit_transform(TextList)

Bool_X_text=MyVect.fit_transform(TextList)

tf_X_text=MyVect.fit_transform(TextList)

## Hmm - that's not quite what we want...
## Let's get the feature names which ARE the words
ColumnNamesText=MyVect.get_feature_names()
print(ColumnNamesText)

## OK good - but we want a document topic model A DTM (matrix of counts)
VectDF=pd.DataFrame(X_text.toarray(), columns=ColumnNamesText, index = IndexList)
print(VectDF)

bool_VectDF=pd.DataFrame(Bool_X_text.toarray(), columns=ColumnNamesText, index = IndexList)
print(bool_VectDF)

tf_VectDF=pd.DataFrame(tf_X_text.toarray(), columns=ColumnNamesText, index = IndexList)
print(tf_VectDF)

SatisfactionVectDF = VectDF.copy(deep=True)
SatisfactionVectDF.insert(loc=0, column='LABEL', value=SatisfactionList)
print(SatisfactionVectDF)

bool_SatisfactionVectDF = bool_VectDF.copy(deep=True)
bool_SatisfactionVectDF.insert(loc=0, column='LABEL', value=SatisfactionList)
print(bool_SatisfactionVectDF)

tf_SatisfactionVectDF = tf_VectDF.copy(deep=True)
tf_SatisfactionVectDF.insert(loc=0, column='LABEL', value=SatisfactionList)
print(tf_SatisfactionVectDF)


##--------------------------------------------------------
####################################################################
###################### model Data Build ############################
####################################################################
##--------------------------------------------------------
from sklearn.model_selection import train_test_split

## Copy everything below here to rebuild for Satisfaction data
TrainDF, TestDF = train_test_split(SatisfactionVectDF, test_size=0.3)
print(TrainDF.head())

## Now we have a training set and a testing set. 
print("\nThe training set is:")
print(TrainDF)
print("\nThe testing set is:")
print(TestDF)

## IMPORTANT - YOU CANNOT LEAVE LABELS ON THE TEST SET
## Save labels
TestLabels=TestDF["LABEL"]
print(TestLabels)
## remove labels
TestDF_nolabels = TestDF.drop(["LABEL"], axis=1)
print(TestDF_nolabels)

TrainDF_nolabels=TrainDF.drop(["LABEL"], axis=1)
print(TrainDF_nolabels)

TrainLabels=TrainDF["LABEL"]
print(TrainLabels)

############# Do the same for the binary dataframe
## Create Train/Test for TrainDFB
TrainDFB, TestDFB = train_test_split(bool_SatisfactionVectDF, test_size=0.3)
print(TrainDFB.head())

TrainDFB_nolabels=TrainDFB.drop(["LABEL"], axis=1)
print(TrainDFB_nolabels)
TrainLabelsB=TrainDFB["LABEL"]
print(TrainLabelsB)

TestDFB_nolabels=TestDFB.drop(["LABEL"], axis=1)
print(TestDFB_nolabels)
TestLabelsB=TestDFB["LABEL"]
print(TestLabelsB)

############# Do the same for the tfidf dataframe
## Create Train/Test for TrainDF tfidf
TrainDF_tf, TestDF_tf = train_test_split(tf_SatisfactionVectDF, test_size=0.3)
print(TrainDF_tf.head())

TrainDF_tf_nolabels=TrainDF_tf.drop(["LABEL"], axis=1)
print(TrainDF_tf_nolabels)
TrainLabels_tf=TrainDF_tf["LABEL"]
print(TrainLabels_tf)

TestDF_tf_nolabels=TestDF_tf.drop(["LABEL"], axis=1)
print(TestDF_tf_nolabels)
TestLabels_tf=TestDF_tf["LABEL"]
print(TestLabels_tf)



#############################################
###########  SVM  for regular model #1 ######
#############################################
from sklearn.svm import LinearSVC
print(TrainDF_nolabels.head())
print(TrainLabels.head())

Satisfaction_SVM_Model=LinearSVC(C=.01)
Satisfaction_SVM_Model.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Satisfaction_svm_predict = Satisfaction_SVM_Model.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(Satisfaction_svm_predict)
print("Actual:")
print(TestLabels)

from sklearn.metrics import confusion_matrix
Satisfaction_SVM_matrix = confusion_matrix(TestLabels, Satisfaction_svm_predict)
print("\nThe confusion matrix is:")
print(Satisfaction_SVM_matrix)
print("\n\n")

from sklearn.metrics import classification_report
Satisfaction_svm_target_names = ['0','1']
print(classification_report(TestLabels, Satisfaction_svm_predict, target_names = Satisfaction_svm_target_names))

Satisfaction_SVM_reg_FP = Satisfaction_SVM_matrix[0][1] 
Satisfaction_SVM_reg_FN = Satisfaction_SVM_matrix[1][0]
Satisfaction_SVM_reg_TP = Satisfaction_SVM_matrix[1][1]
Satisfaction_SVM_reg_TN = Satisfaction_SVM_matrix[0][0]

# Overall accuracy
Satisfaction_SVM_reg_ACC = (Satisfaction_SVM_reg_TP + Satisfaction_SVM_reg_TN)/(Satisfaction_SVM_reg_TP + Satisfaction_SVM_reg_FP + Satisfaction_SVM_reg_FN + Satisfaction_SVM_reg_TN)
print(Satisfaction_SVM_reg_ACC)

SatisfactionAccuracyDict = {}
SatisfactionAccuracyDict.update({'Satisfaction_SVM_reg_ACC': Satisfaction_SVM_reg_ACC})
print(SatisfactionAccuracyDict)

#############################################
###########  SVM  for regular model #2 ######
#############################################

Satisfaction_SVM_Model2=LinearSVC(C=1)
Satisfaction_SVM_Model2.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Satisfaction_svm_predict2 = Satisfaction_SVM_Model2.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(Satisfaction_svm_predict2)
print("Actual:")
print(TestLabels)

Satisfaction_SVM_matrix2 = confusion_matrix(TestLabels, Satisfaction_svm_predict2)
print("\nThe confusion matrix is:")
print(Satisfaction_SVM_matrix2)
print("\n\n")

Satisfaction_svm_target_names2 = ['0','1']
print(classification_report(TestLabels, Satisfaction_svm_predict2, target_names = Satisfaction_svm_target_names2))

Satisfaction_SVM_reg2_FP = Satisfaction_SVM_matrix2[0][1] 
Satisfaction_SVM_reg2_FN = Satisfaction_SVM_matrix2[1][0]
Satisfaction_SVM_reg2_TP = Satisfaction_SVM_matrix2[1][1]
Satisfaction_SVM_reg2_TN = Satisfaction_SVM_matrix2[0][0]

# Overall accuracy
Satisfaction_SVM_reg2_ACC = (Satisfaction_SVM_reg2_TP + Satisfaction_SVM_reg2_TN)/(Satisfaction_SVM_reg2_TP + Satisfaction_SVM_reg2_FP + Satisfaction_SVM_reg2_FN + Satisfaction_SVM_reg2_TN)
print(Satisfaction_SVM_reg2_ACC)

SatisfactionAccuracyDict.update({'Satisfaction_SVM_reg2_ACC': Satisfaction_SVM_reg2_ACC})
print(SatisfactionAccuracyDict)

#############################################
###########  SVM  for regular model #3 ######
#############################################

Satisfaction_SVM_Model3=LinearSVC(C=100)
Satisfaction_SVM_Model3.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Satisfaction_svm_predict3 = Satisfaction_SVM_Model3.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(Satisfaction_svm_predict3)
print("Actual:")
print(TestLabels)

Satisfaction_SVM_matrix3 = confusion_matrix(TestLabels, Satisfaction_svm_predict3)
print("\nThe confusion matrix is:")
print(Satisfaction_SVM_matrix3)
print("\n\n")

from sklearn.metrics import classification_report
Satisfaction_svm_target_names3 = ['0','1']
print(classification_report(TestLabels, Satisfaction_svm_predict3, target_names = Satisfaction_svm_target_names3))

Satisfaction_SVM_reg3_FP = Satisfaction_SVM_matrix3[0][1] 
Satisfaction_SVM_reg3_FN = Satisfaction_SVM_matrix3[1][0]
Satisfaction_SVM_reg3_TP = Satisfaction_SVM_matrix3[1][1]
Satisfaction_SVM_reg3_TN = Satisfaction_SVM_matrix3[0][0]

# Overall accuracy
Satisfaction_SVM_reg3_ACC = (Satisfaction_SVM_reg3_TP + Satisfaction_SVM_reg3_TN)/(Satisfaction_SVM_reg3_TP + Satisfaction_SVM_reg3_FP + Satisfaction_SVM_reg3_FN + Satisfaction_SVM_reg3_TN)
print(Satisfaction_SVM_reg3_ACC)

SatisfactionAccuracyDict.update({'Satisfaction_SVM_reg3_ACC': Satisfaction_SVM_reg3_ACC})
print(SatisfactionAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #1 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

Satisfaction_B_SVM_Model=LinearSVC(C=100)
Satisfaction_B_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Satisfaction_b_svm_predict = Satisfaction_B_SVM_Model.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(Satisfaction_b_svm_predict)
print("\nActual:")
print(TestLabelsB)

Satisfaction_B_SVM_matrix = confusion_matrix(TestLabelsB, Satisfaction_b_svm_predict)
print("\nThe confusion matrix is:")
print(Satisfaction_B_SVM_matrix)
print("\n\n")

Satisfaction_svm_B_target_names = ['0','1']
print(classification_report(TestLabelsB, Satisfaction_b_svm_predict, target_names = Satisfaction_svm_B_target_names))

Satisfaction_SVM_bool_FP = Satisfaction_B_SVM_matrix[0][1] 
Satisfaction_SVM_bool_FN = Satisfaction_B_SVM_matrix[1][0]
Satisfaction_SVM_bool_TP = Satisfaction_B_SVM_matrix[1][1]
Satisfaction_SVM_bool_TN = Satisfaction_B_SVM_matrix[0][0]

# Overall accuracy
Satisfaction_SVM_bool_ACC = (Satisfaction_SVM_bool_TP + Satisfaction_SVM_bool_TN)/(Satisfaction_SVM_bool_TP + Satisfaction_SVM_bool_FP + Satisfaction_SVM_bool_FN + Satisfaction_SVM_bool_TN)
print(Satisfaction_SVM_bool_ACC)

SatisfactionAccuracyDict.update({'Satisfaction_SVM_bool_ACC': Satisfaction_SVM_bool_ACC})
print(SatisfactionAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #2 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

Satisfaction_B_SVM_Model2=LinearSVC(C=1)
Satisfaction_B_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Satisfaction_b_svm_predict2 = Satisfaction_B_SVM_Model2.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(Satisfaction_b_svm_predict2)
print("\nActual:")
print(TestLabelsB)

Satisfaction_B_SVM_matrix2 = confusion_matrix(TestLabelsB, Satisfaction_b_svm_predict2)
print("\nThe confusion matrix is:")
print(Satisfaction_B_SVM_matrix2)
print("\n\n")

Satisfaction_svm_B_target_names2 = ['0','1']
print(classification_report(TestLabelsB, Satisfaction_b_svm_predict2, target_names = Satisfaction_svm_B_target_names2))

Satisfaction_SVM_bool2_FP = Satisfaction_B_SVM_matrix2[0][1] 
Satisfaction_SVM_bool2_FN = Satisfaction_B_SVM_matrix2[1][0]
Satisfaction_SVM_bool2_TP = Satisfaction_B_SVM_matrix2[1][1]
Satisfaction_SVM_bool2_TN = Satisfaction_B_SVM_matrix2[0][0]

# Overall accuracy
Satisfaction_SVM_bool2_ACC = (Satisfaction_SVM_bool2_TP + Satisfaction_SVM_bool2_TN)/(Satisfaction_SVM_bool2_TP + Satisfaction_SVM_bool2_FP + Satisfaction_SVM_bool2_FN + Satisfaction_SVM_bool2_TN)
print(Satisfaction_SVM_bool2_ACC)

SatisfactionAccuracyDict.update({'Satisfaction_SVM_bool2_ACC': Satisfaction_SVM_bool2_ACC})
print(SatisfactionAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #3 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

Satisfaction_B_SVM_Model3=LinearSVC(C=.01)
Satisfaction_B_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Satisfaction_b_svm_predict3 = Satisfaction_B_SVM_Model3.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(Satisfaction_b_svm_predict3)
print("\nActual:")
print(TestLabelsB)

Satisfaction_B_SVM_matrix3 = confusion_matrix(TestLabelsB, Satisfaction_b_svm_predict3)
print("\nThe confusion matrix is:")
print(Satisfaction_B_SVM_matrix3)
print("\n\n")

Satisfaction_svm_B_target_names3 = ['0','1']
print(classification_report(TestLabelsB, Satisfaction_b_svm_predict3, target_names = Satisfaction_svm_B_target_names3))

Satisfaction_SVM_bool3_FP = Satisfaction_B_SVM_matrix3[0][1] 
Satisfaction_SVM_bool3_FN = Satisfaction_B_SVM_matrix3[1][0]
Satisfaction_SVM_bool3_TP = Satisfaction_B_SVM_matrix3[1][1]
Satisfaction_SVM_bool3_TN = Satisfaction_B_SVM_matrix3[0][0]

# Overall accuracy
Satisfaction_SVM_bool3_ACC = (Satisfaction_SVM_bool3_TP + Satisfaction_SVM_bool3_TN)/(Satisfaction_SVM_bool3_TP + Satisfaction_SVM_bool3_FP + Satisfaction_SVM_bool3_FN + Satisfaction_SVM_bool3_TN)
print(Satisfaction_SVM_bool3_ACC)

SatisfactionAccuracyDict.update({'Satisfaction_SVM_bool3_ACC': Satisfaction_SVM_bool3_ACC})
print(SatisfactionAccuracyDict)

#############################################
###########  SVM  for tfidf model #1 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

Satisfaction_tf_SVM_Model=LinearSVC(C=.001)
Satisfaction_tf_SVM_Model.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Satisfaction_tf_svm_predict = Satisfaction_tf_SVM_Model.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(Satisfaction_tf_svm_predict)
print("\nActual:")
print(TestLabels_tf)

Satisfaction_tf_SVM_matrix = confusion_matrix(TestLabels_tf, Satisfaction_tf_svm_predict)
print("\nThe confusion matrix is:")
print(Satisfaction_tf_SVM_matrix)
print("\n\n")

Satisfaction_svm_tf_target_names = ['0','1']
print(classification_report(TestLabels_tf, Satisfaction_tf_svm_predict, target_names = Satisfaction_svm_tf_target_names))

Satisfaction_SVM_tf_FP = Satisfaction_tf_SVM_matrix[0][1] 
Satisfaction_SVM_tf_FN = Satisfaction_tf_SVM_matrix[1][0]
Satisfaction_SVM_tf_TP = Satisfaction_tf_SVM_matrix[1][1]
Satisfaction_SVM_tf_TN = Satisfaction_tf_SVM_matrix[0][0]

# Overall accuracy
Satisfaction_SVM_tf_ACC = (Satisfaction_SVM_tf_TP + Satisfaction_SVM_tf_TN)/(Satisfaction_SVM_tf_TP + Satisfaction_SVM_tf_FP + Satisfaction_SVM_tf_FN + Satisfaction_SVM_tf_TN)
print(Satisfaction_SVM_tf_ACC)

SatisfactionAccuracyDict.update({'Satisfaction_SVM_tf_ACC': Satisfaction_SVM_tf_ACC})
print(SatisfactionAccuracyDict)

#############################################
###########  SVM  for tfidf model #2 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

Satisfaction_tf_SVM_Model2=LinearSVC(C=1)
Satisfaction_tf_SVM_Model2.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Satisfaction_tf_svm_predict2 = Satisfaction_tf_SVM_Model2.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(Satisfaction_tf_svm_predict2)
print("\nActual:")
print(TestLabels_tf)

Satisfaction_tf_SVM_matrix2 = confusion_matrix(TestLabels_tf, Satisfaction_tf_svm_predict2)
print("\nThe confusion matrix is:")
print(Satisfaction_tf_SVM_matrix2)
print("\n\n")

Satisfaction_svm_tf_target_names2 = ['0','1']
print(classification_report(TestLabels_tf, Satisfaction_tf_svm_predict2, target_names = Satisfaction_svm_tf_target_names2))

Satisfaction_SVM_tf2_FP = Satisfaction_tf_SVM_matrix2[0][1] 
Satisfaction_SVM_tf2_FN = Satisfaction_tf_SVM_matrix2[1][0]
Satisfaction_SVM_tf2_TP = Satisfaction_tf_SVM_matrix2[1][1]
Satisfaction_SVM_tf2_TN = Satisfaction_tf_SVM_matrix2[0][0]

# Overall accuracy
Satisfaction_SVM_tf2_ACC = (Satisfaction_SVM_tf2_TP + Satisfaction_SVM_tf2_TN)/(Satisfaction_SVM_tf2_TP + Satisfaction_SVM_tf2_FP + Satisfaction_SVM_tf2_FN + Satisfaction_SVM_tf2_TN)
print(Satisfaction_SVM_tf2_ACC)

SatisfactionAccuracyDict.update({'Satisfaction_SVM_tf2_ACC': Satisfaction_SVM_tf2_ACC})
print(SatisfactionAccuracyDict)

#############################################
###########  SVM  for tfidf model #3 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

Satisfaction_tf_SVM_Model3=LinearSVC(C=1)
Satisfaction_tf_SVM_Model3.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Satisfaction_tf_svm_predict3 = Satisfaction_tf_SVM_Model3.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(Satisfaction_tf_svm_predict3)
print("\nActual:")
print(TestLabels_tf)

Satisfaction_tf_SVM_matrix3 = confusion_matrix(TestLabels_tf, Satisfaction_tf_svm_predict3)
print("\nThe confusion matrix is:")
print(Satisfaction_tf_SVM_matrix3)
print("\n\n")

Satisfaction_svm_tf_target_names3 = ['0','1']
print(classification_report(TestLabels_tf, Satisfaction_tf_svm_predict3, target_names = Satisfaction_svm_tf_target_names3))

Satisfaction_SVM_tf3_FP = Satisfaction_tf_SVM_matrix3[0][1] 
Satisfaction_SVM_tf3_FN = Satisfaction_tf_SVM_matrix3[1][0]
Satisfaction_SVM_tf3_TP = Satisfaction_tf_SVM_matrix3[1][1]
Satisfaction_SVM_tf3_TN = Satisfaction_tf_SVM_matrix3[0][0]

# Overall accuracy
Satisfaction_SVM_tf3_ACC = (Satisfaction_SVM_tf3_TP + Satisfaction_SVM_tf3_TN)/(Satisfaction_SVM_tf3_TP + Satisfaction_SVM_tf3_FP + Satisfaction_SVM_tf3_FN + Satisfaction_SVM_tf3_TN)
print(Satisfaction_SVM_tf3_ACC)

SatisfactionAccuracyDict.update({'Satisfaction_SVM_tf3_ACC': Satisfaction_SVM_tf3_ACC})
print(SatisfactionAccuracyDict)

#############################################
#######  SVM  with different kernels ########
#############################################
## running on Bernoulli model since it had best accuracy above
## i want to see if the accuracy gets improved in the different kernels

#############################################
###############  Sigmoid #1 #################
#############################################

Satisfaction_sig_SVM_Model=sklearn.svm.SVC(C=10, kernel='sigmoid', 
                           verbose=True, gamma="scale")
Satisfaction_sig_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
Satisfaction_sig_svm_predict = Satisfaction_sig_SVM_Model.predict(TestDFB_nolabels)
print("\nsig SVM prediction:\n")
print(Satisfaction_sig_svm_predict)
print("\nActual:")
print(TestLabelsB)

Satisfaction_sig_SVM_matrix = confusion_matrix(TestLabelsB, Satisfaction_sig_svm_predict)
print("\nThe sig confusion matrix is:")
print(Satisfaction_sig_SVM_matrix)
print("\n\n")

Satisfaction_svm_sig_target_names = ['0','1']
print(classification_report(TestLabelsB, Satisfaction_sig_svm_predict, target_names = Satisfaction_svm_sig_target_names))

Satisfaction_SVM_sig_FP = Satisfaction_sig_SVM_matrix[0][1] 
Satisfaction_SVM_sig_FN = Satisfaction_sig_SVM_matrix[1][0]
Satisfaction_SVM_sig_TP = Satisfaction_sig_SVM_matrix[1][1]
Satisfaction_SVM_sig_TN = Satisfaction_sig_SVM_matrix[0][0]

# Overall accuracy
Satisfaction_SVM_sig_ACC = (Satisfaction_SVM_sig_TP + Satisfaction_SVM_sig_TN)/(Satisfaction_SVM_sig_TP + Satisfaction_SVM_sig_FP + Satisfaction_SVM_sig_FN + Satisfaction_SVM_sig_TN)
print(Satisfaction_SVM_sig_ACC)

SatisfactionAccuracyDict.update({'Satisfaction_SVM_sig_ACC': Satisfaction_SVM_sig_ACC})
print(SatisfactionAccuracyDict)

#############################################
###############  Sigmoid #2 #################
#############################################

Satisfaction_sig_SVM_Model2=sklearn.svm.SVC(C=1000, kernel='sigmoid', 
                           verbose=True, gamma="scale")
Satisfaction_sig_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
Satisfaction_sig_svm_predict2 = Satisfaction_sig_SVM_Model2.predict(TestDFB_nolabels)
print("\nsig SVM prediction 2:\n")
print(Satisfaction_sig_svm_predict2)
print("\nActual:")
print(TestLabelsB)

Satisfaction_sig_SVM_matrix2 = confusion_matrix(TestLabelsB, Satisfaction_sig_svm_predict2)
print("\nThe sig 2 confusion matrix is:")
print(Satisfaction_sig_SVM_matrix2)
print("\n\n")

Satisfaction_svm_sig_target_names2 = ['0','1']
print(classification_report(TestLabelsB, Satisfaction_sig_svm_predict2, target_names = Satisfaction_svm_sig_target_names2))

Satisfaction_SVM_sig2_FP = Satisfaction_sig_SVM_matrix2[0][1] 
Satisfaction_SVM_sig2_FN = Satisfaction_sig_SVM_matrix2[1][0]
Satisfaction_SVM_sig2_TP = Satisfaction_sig_SVM_matrix2[1][1]
Satisfaction_SVM_sig2_TN = Satisfaction_sig_SVM_matrix2[0][0]

# Overall accuracy
Satisfaction_SVM_sig2_ACC = (Satisfaction_SVM_sig2_TP + Satisfaction_SVM_sig2_TN)/(Satisfaction_SVM_sig2_TP + Satisfaction_SVM_sig2_FP + Satisfaction_SVM_sig2_FN + Satisfaction_SVM_sig2_TN)
print(Satisfaction_SVM_sig2_ACC)

SatisfactionAccuracyDict.update({'Satisfaction_SVM_sig2_ACC': Satisfaction_SVM_sig2_ACC})
print(SatisfactionAccuracyDict)

#############################################
###############  Sigmoid #3 #################
#############################################

Satisfaction_sig_SVM_Model3=sklearn.svm.SVC(C=.001, kernel='sigmoid', 
                           verbose=True, gamma="scale")
Satisfaction_sig_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
Satisfaction_sig_svm_predict3 = Satisfaction_sig_SVM_Model3.predict(TestDFB_nolabels)
print("\nsig SVM prediction 3:\n")
print(Satisfaction_sig_svm_predict3)
print("\nActual:")
print(TestLabelsB)

Satisfaction_sig_SVM_matrix3 = confusion_matrix(TestLabelsB, Satisfaction_sig_svm_predict3)
print("\nThe sig 3 confusion matrix is:")
print(Satisfaction_sig_SVM_matrix3)
print("\n\n")

Satisfaction_svm_sig_target_names3 = ['0','1']
print(classification_report(TestLabelsB, Satisfaction_sig_svm_predict3, target_names = Satisfaction_svm_sig_target_names3))

Satisfaction_SVM_sig3_FP = Satisfaction_sig_SVM_matrix3[0][1] 
Satisfaction_SVM_sig3_FN = Satisfaction_sig_SVM_matrix3[1][0]
Satisfaction_SVM_sig3_TP = Satisfaction_sig_SVM_matrix3[1][1]
Satisfaction_SVM_sig3_TN = Satisfaction_sig_SVM_matrix3[0][0]

# Overall accuracy
Satisfaction_SVM_sig3_ACC = (Satisfaction_SVM_sig3_TP + Satisfaction_SVM_sig3_TN)/(Satisfaction_SVM_sig3_TP + Satisfaction_SVM_sig3_FP + Satisfaction_SVM_sig3_FN + Satisfaction_SVM_sig3_TN)
print(Satisfaction_SVM_sig3_ACC)

SatisfactionAccuracyDict.update({'Satisfaction_SVM_sig3_ACC': Satisfaction_SVM_sig3_ACC})
print(SatisfactionAccuracyDict)

#############################################
################  Poly #1 ###################
#############################################

Satisfaction_poly_SVM_Model=sklearn.svm.SVC(C=1000, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(Satisfaction_poly_SVM_Model)
Satisfaction_poly_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
Satisfaction_poly_svm_predict = Satisfaction_poly_SVM_Model.predict(TestDFB_nolabels)
print("\npoly SVM prediction:\n")
print(Satisfaction_poly_svm_predict)
print("\nActual:")
print(TestLabelsB)

Satisfaction_poly_SVM_matrix = confusion_matrix(TestLabelsB, Satisfaction_poly_svm_predict)
print("\nThe poly confusion matrix is:")
print(Satisfaction_poly_SVM_matrix)
print("\n\n")

Satisfaction_svm_poly_target_names = ['0','1']
print(classification_report(TestLabelsB, Satisfaction_poly_svm_predict, target_names = Satisfaction_svm_poly_target_names))

Satisfaction_SVM_poly_FP = Satisfaction_poly_SVM_matrix[0][1] 
Satisfaction_SVM_poly_FN = Satisfaction_poly_SVM_matrix[1][0]
Satisfaction_SVM_poly_TP = Satisfaction_poly_SVM_matrix[1][1]
Satisfaction_SVM_poly_TN = Satisfaction_poly_SVM_matrix[0][0]

# Overall accuracy
Satisfaction_SVM_poly_ACC = (Satisfaction_SVM_poly_TP + Satisfaction_SVM_poly_TN)/(Satisfaction_SVM_poly_TP + Satisfaction_SVM_poly_FP + Satisfaction_SVM_poly_FN + Satisfaction_SVM_poly_TN)
print(Satisfaction_SVM_poly_ACC)

SatisfactionAccuracyDict.update({'Satisfaction_SVM_poly_ACC': Satisfaction_SVM_poly_ACC})
print(SatisfactionAccuracyDict)

#############################################
################  Poly #2 ###################
#############################################

Satisfaction_poly_SVM_Model2=sklearn.svm.SVC(C=10, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(Satisfaction_poly_SVM_Model2)
Satisfaction_poly_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
Satisfaction_poly_svm_predict2 = Satisfaction_poly_SVM_Model2.predict(TestDFB_nolabels)
print("\npoly SVM 2 prediction:\n")
print(Satisfaction_poly_svm_predict2)
print("\nActual:")
print(TestLabelsB)

Satisfaction_poly_SVM_matrix2 = confusion_matrix(TestLabelsB, Satisfaction_poly_svm_predict2)
print("\nThe poly 2 confusion matrix is:")
print(Satisfaction_poly_SVM_matrix2)
print("\n\n")

Satisfaction_svm_poly_target_names2 = ['0','1']
print(classification_report(TestLabelsB, Satisfaction_poly_svm_predict2, target_names = Satisfaction_svm_poly_target_names2))

Satisfaction_SVM_poly2_FP = Satisfaction_poly_SVM_matrix2[0][1] 
Satisfaction_SVM_poly2_FN = Satisfaction_poly_SVM_matrix2[1][0]
Satisfaction_SVM_poly2_TP = Satisfaction_poly_SVM_matrix2[1][1]
Satisfaction_SVM_poly2_TN = Satisfaction_poly_SVM_matrix2[0][0]

# Overall accuracy
Satisfaction_SVM_poly2_ACC = (Satisfaction_SVM_poly2_TP + Satisfaction_SVM_poly2_TN)/(Satisfaction_SVM_poly2_TP + Satisfaction_SVM_poly2_FP + Satisfaction_SVM_poly2_FN + Satisfaction_SVM_poly2_TN)
print(Satisfaction_SVM_poly2_ACC)

SatisfactionAccuracyDict.update({'Satisfaction_SVM_poly2_ACC': Satisfaction_SVM_poly2_ACC})
print(SatisfactionAccuracyDict)

#############################################
################  Poly #3 ###################
#############################################

Satisfaction_poly_SVM_Model3=sklearn.svm.SVC(C=.001, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(Satisfaction_poly_SVM_Model3)
Satisfaction_poly_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
Satisfaction_poly_svm_predict3 = Satisfaction_poly_SVM_Model3.predict(TestDFB_nolabels)
print("\npoly SVM 3 prediction:\n")
print(Satisfaction_poly_svm_predict3)
print("\nActual:")
print(TestLabelsB)

Satisfaction_poly_SVM_matrix3 = confusion_matrix(TestLabelsB, Satisfaction_poly_svm_predict3)
print("\nThe poly 3 confusion matrix is:")
print(Satisfaction_poly_SVM_matrix3)
print("\n\n")

Satisfaction_svm_poly_target_names3 = ['0','1']
print(classification_report(TestLabelsB, Satisfaction_poly_svm_predict3, target_names = Satisfaction_svm_poly_target_names3))

Satisfaction_SVM_poly3_FP = Satisfaction_poly_SVM_matrix3[0][1] 
Satisfaction_SVM_poly3_FN = Satisfaction_poly_SVM_matrix3[1][0]
Satisfaction_SVM_poly3_TP = Satisfaction_poly_SVM_matrix3[1][1]
Satisfaction_SVM_poly3_TN = Satisfaction_poly_SVM_matrix3[0][0]

# Overall accuracy
Satisfaction_SVM_poly3_ACC = (Satisfaction_SVM_poly3_TP + Satisfaction_SVM_poly3_TN)/(Satisfaction_SVM_poly3_TP + Satisfaction_SVM_poly3_FP + Satisfaction_SVM_poly3_FN + Satisfaction_SVM_poly3_TN)
print(Satisfaction_SVM_poly3_ACC)

SatisfactionAccuracyDict.update({'Satisfaction_SVM_poly3_ACC': Satisfaction_SVM_poly3_ACC})
print(SatisfactionAccuracyDict)

SatisfactionVisDF = pd.DataFrame(SatisfactionAccuracyDict.items(), index = SatisfactionAccuracyDict.keys(), columns=['Model','Accuracy'])
print(SatisfactionVisDF)
SortedSatisfactionVisDF = SatisfactionVisDF.sort_values('Accuracy', ascending = [True])
print(SortedSatisfactionVisDF)

print(PartyAccuracyDict)
print(WinnerAccuracyDict)
print(SentimentAccuracyDict)
print(IncumCandAccuracyDict)
print(IncumPartyAccuracyDict)
print(UnemploymentAccuracyDict)
print(GDPAccuracyDict)
print(InflationAccuracyDict)
print(SatisfactionAccuracyDict)

SortedPartyVisDF.plot.barh(y='Accuracy')
SortedWinnerVisDF.plot.barh(y='Accuracy')
SortedSentimentVisDF.plot.barh(y='Accuracy')
SortedIncumCandVisDF.plot.barh(y='Accuracy')
SortedIncumPartyVisDF.plot.barh(y='Accuracy')
SortedUnemploymentVisDF.plot.barh(y='Accuracy')
SortedGDPVisDF.plot.barh(y='Accuracy')
SortedInflationVisDF.plot.barh(y='Accuracy')
SortedSatisfactionVisDF.plot.barh(y='Accuracy')


#print(IndexList)
#print(TextList)
#print(PartyList)
#print(WinnerList)
#print(SentimentList)
#print(IncumCandList)
#print(IncumPartyList)
#print(UnemploymentList)
#print(GDPList)
#print(InflationList)
#print(SatisfactionList)
print(RealIncomeGrowthList)
print(PresApprovalList)
print(IncomeTaxList)
print(DjiaVolumeList)
print(CpiList)
print(CciIndexList)

#################################################################################################
#################################################################################################
### everything from this point down is a matter of find and replace for each of the variables ###
#################################################################################################
#################################################################################################

# the rows weren't all populated for the variables below so an edited text list is needed

### starting to build a model for RealIncomeGrowth
df_RealIncomeGrowth = rawfile.copy(deep=True)
df_RealIncomeGrowth = df_RealIncomeGrowth.filter(['id', 'real_income_growth', 'comb_text'])
df_RealIncomeGrowth = df_RealIncomeGrowth[df_RealIncomeGrowth['real_income_growth'].notna()]
print(df_RealIncomeGrowth)

RealIncomeGrowthList = []
TextList = []
IndexList = []

for row in df_RealIncomeGrowth.itertuples():
    RealIncomeGrowthlabel = row.real_income_growth
    textlabel = row.comb_text
    textlabel = textlabel.replace('\n',' ') # remove \n
    textlabel = textlabel.replace("'","'")
    textlabel = textlabel.replace('\t',' ') # remove \t
    #textlabel = textlabel.replace(',',' ') # remove ,
    #textlabel = textlabel.replace('.',' ') # remove .
    #textlabel = textlabel.replace('"',' ') # remove "
    #textlabel = textlabel.replace("'",' ') # remove '
    textlabel = textlabel.replace('nan',' ') # remove nan
    #textlabel = textlabel.replace('-',' ') # remove -
    #textlabel = textlabel.replace(' - ',' ') # remove -
    #textlabel = textlabel.replace('?',' ') # remove ?
    #textlabel = textlabel.replace('!',' ') # remove !
    #textlabel = textlabel.replace(':',' ') # remove :
    #textlabel = textlabel.replace(';',' ') # remove ;
    #textlabel = textlabel.replace('  ',' ') # remove double spaces
    indexlabel = row.id
    TextList.append(textlabel)
    IndexList.append(indexlabel)
    RealIncomeGrowthList.append(RealIncomeGrowthlabel)

RealIncomeGrowthList = [ int(x) for x in RealIncomeGrowthList ]
print(RealIncomeGrowthList)

X_text=MyVect.fit_transform(TextList)

Bool_X_text=MyVect.fit_transform(TextList)

tf_X_text=MyVect.fit_transform(TextList)

## Hmm - that's not quite what we want...
## Let's get the feature names which ARE the words
ColumnNamesText=MyVect.get_feature_names()
print(ColumnNamesText)

## OK good - but we want a document topic model A DTM (matrix of counts)
VectDF=pd.DataFrame(X_text.toarray(), columns=ColumnNamesText, index = IndexList)
print(VectDF)

bool_VectDF=pd.DataFrame(Bool_X_text.toarray(), columns=ColumnNamesText, index = IndexList)
print(bool_VectDF)

tf_VectDF=pd.DataFrame(tf_X_text.toarray(), columns=ColumnNamesText, index = IndexList)
print(tf_VectDF)

RealIncomeGrowthVectDF = VectDF.copy(deep=True)
RealIncomeGrowthVectDF.insert(loc=0, column='LABEL', value=RealIncomeGrowthList)
print(RealIncomeGrowthVectDF)

bool_RealIncomeGrowthVectDF = bool_VectDF.copy(deep=True)
bool_RealIncomeGrowthVectDF.insert(loc=0, column='LABEL', value=RealIncomeGrowthList)
print(bool_RealIncomeGrowthVectDF)

tf_RealIncomeGrowthVectDF = tf_VectDF.copy(deep=True)
tf_RealIncomeGrowthVectDF.insert(loc=0, column='LABEL', value=RealIncomeGrowthList)
print(tf_RealIncomeGrowthVectDF)


##--------------------------------------------------------
####################################################################
###################### model Data Build ############################
####################################################################
##--------------------------------------------------------
from sklearn.model_selection import train_test_split

## Copy everything below here to rebuild for RealIncomeGrowth data
TrainDF, TestDF = train_test_split(RealIncomeGrowthVectDF, test_size=0.3)
print(TrainDF.head())

## Now we have a training set and a testing set. 
print("\nThe training set is:")
print(TrainDF)
print("\nThe testing set is:")
print(TestDF)

## IMPORTANT - YOU CANNOT LEAVE LABELS ON THE TEST SET
## Save labels
TestLabels=TestDF["LABEL"]
print(TestLabels)
## remove labels
TestDF_nolabels = TestDF.drop(["LABEL"], axis=1)
print(TestDF_nolabels)

TrainDF_nolabels=TrainDF.drop(["LABEL"], axis=1)
print(TrainDF_nolabels)

TrainLabels=TrainDF["LABEL"]
print(TrainLabels)

############# Do the same for the binary dataframe
## Create Train/Test for TrainDFB
TrainDFB, TestDFB = train_test_split(bool_RealIncomeGrowthVectDF, test_size=0.3)
print(TrainDFB.head())

TrainDFB_nolabels=TrainDFB.drop(["LABEL"], axis=1)
print(TrainDFB_nolabels)
TrainLabelsB=TrainDFB["LABEL"]
print(TrainLabelsB)

TestDFB_nolabels=TestDFB.drop(["LABEL"], axis=1)
print(TestDFB_nolabels)
TestLabelsB=TestDFB["LABEL"]
print(TestLabelsB)

############# Do the same for the tfidf dataframe
## Create Train/Test for TrainDF tfidf
TrainDF_tf, TestDF_tf = train_test_split(tf_RealIncomeGrowthVectDF, test_size=0.3)
print(TrainDF_tf.head())

TrainDF_tf_nolabels=TrainDF_tf.drop(["LABEL"], axis=1)
print(TrainDF_tf_nolabels)
TrainLabels_tf=TrainDF_tf["LABEL"]
print(TrainLabels_tf)

TestDF_tf_nolabels=TestDF_tf.drop(["LABEL"], axis=1)
print(TestDF_tf_nolabels)
TestLabels_tf=TestDF_tf["LABEL"]
print(TestLabels_tf)



#############################################
###########  SVM  for regular model #1 ######
#############################################
from sklearn.svm import LinearSVC
print(TrainDF_nolabels.head())
print(TrainLabels.head())

RealIncomeGrowth_SVM_Model=LinearSVC(C=.01)
RealIncomeGrowth_SVM_Model.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
RealIncomeGrowth_svm_predict = RealIncomeGrowth_SVM_Model.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(RealIncomeGrowth_svm_predict)
print("Actual:")
print(TestLabels)

from sklearn.metrics import confusion_matrix
RealIncomeGrowth_SVM_matrix = confusion_matrix(TestLabels, RealIncomeGrowth_svm_predict)
print("\nThe confusion matrix is:")
print(RealIncomeGrowth_SVM_matrix)
print("\n\n")

from sklearn.metrics import classification_report
RealIncomeGrowth_svm_target_names = ['0','1']
print(classification_report(TestLabels, RealIncomeGrowth_svm_predict, target_names = RealIncomeGrowth_svm_target_names))

RealIncomeGrowth_SVM_reg_FP = RealIncomeGrowth_SVM_matrix[0][1] 
RealIncomeGrowth_SVM_reg_FN = RealIncomeGrowth_SVM_matrix[1][0]
RealIncomeGrowth_SVM_reg_TP = RealIncomeGrowth_SVM_matrix[1][1]
RealIncomeGrowth_SVM_reg_TN = RealIncomeGrowth_SVM_matrix[0][0]

# Overall accuracy
RealIncomeGrowth_SVM_reg_ACC = (RealIncomeGrowth_SVM_reg_TP + RealIncomeGrowth_SVM_reg_TN)/(RealIncomeGrowth_SVM_reg_TP + RealIncomeGrowth_SVM_reg_FP + RealIncomeGrowth_SVM_reg_FN + RealIncomeGrowth_SVM_reg_TN)
print(RealIncomeGrowth_SVM_reg_ACC)

RealIncomeGrowthAccuracyDict = {}
RealIncomeGrowthAccuracyDict.update({'RealIncomeGrowth_SVM_reg_ACC': RealIncomeGrowth_SVM_reg_ACC})
print(RealIncomeGrowthAccuracyDict)

#############################################
###########  SVM  for regular model #2 ######
#############################################

RealIncomeGrowth_SVM_Model2=LinearSVC(C=1)
RealIncomeGrowth_SVM_Model2.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
RealIncomeGrowth_svm_predict2 = RealIncomeGrowth_SVM_Model2.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(RealIncomeGrowth_svm_predict2)
print("Actual:")
print(TestLabels)

RealIncomeGrowth_SVM_matrix2 = confusion_matrix(TestLabels, RealIncomeGrowth_svm_predict2)
print("\nThe confusion matrix is:")
print(RealIncomeGrowth_SVM_matrix2)
print("\n\n")

RealIncomeGrowth_svm_target_names2 = ['0','1']
print(classification_report(TestLabels, RealIncomeGrowth_svm_predict2, target_names = RealIncomeGrowth_svm_target_names2))

RealIncomeGrowth_SVM_reg2_FP = RealIncomeGrowth_SVM_matrix2[0][1] 
RealIncomeGrowth_SVM_reg2_FN = RealIncomeGrowth_SVM_matrix2[1][0]
RealIncomeGrowth_SVM_reg2_TP = RealIncomeGrowth_SVM_matrix2[1][1]
RealIncomeGrowth_SVM_reg2_TN = RealIncomeGrowth_SVM_matrix2[0][0]

# Overall accuracy
RealIncomeGrowth_SVM_reg2_ACC = (RealIncomeGrowth_SVM_reg2_TP + RealIncomeGrowth_SVM_reg2_TN)/(RealIncomeGrowth_SVM_reg2_TP + RealIncomeGrowth_SVM_reg2_FP + RealIncomeGrowth_SVM_reg2_FN + RealIncomeGrowth_SVM_reg2_TN)
print(RealIncomeGrowth_SVM_reg2_ACC)

RealIncomeGrowthAccuracyDict.update({'RealIncomeGrowth_SVM_reg2_ACC': RealIncomeGrowth_SVM_reg2_ACC})
print(RealIncomeGrowthAccuracyDict)

#############################################
###########  SVM  for regular model #3 ######
#############################################

RealIncomeGrowth_SVM_Model3=LinearSVC(C=100)
RealIncomeGrowth_SVM_Model3.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
RealIncomeGrowth_svm_predict3 = RealIncomeGrowth_SVM_Model3.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(RealIncomeGrowth_svm_predict3)
print("Actual:")
print(TestLabels)

RealIncomeGrowth_SVM_matrix3 = confusion_matrix(TestLabels, RealIncomeGrowth_svm_predict3)
print("\nThe confusion matrix is:")
print(RealIncomeGrowth_SVM_matrix3)
print("\n\n")

from sklearn.metrics import classification_report
RealIncomeGrowth_svm_target_names3 = ['0','1']
print(classification_report(TestLabels, RealIncomeGrowth_svm_predict3, target_names = RealIncomeGrowth_svm_target_names3))

RealIncomeGrowth_SVM_reg3_FP = RealIncomeGrowth_SVM_matrix3[0][1] 
RealIncomeGrowth_SVM_reg3_FN = RealIncomeGrowth_SVM_matrix3[1][0]
RealIncomeGrowth_SVM_reg3_TP = RealIncomeGrowth_SVM_matrix3[1][1]
RealIncomeGrowth_SVM_reg3_TN = RealIncomeGrowth_SVM_matrix3[0][0]

# Overall accuracy
RealIncomeGrowth_SVM_reg3_ACC = (RealIncomeGrowth_SVM_reg3_TP + RealIncomeGrowth_SVM_reg3_TN)/(RealIncomeGrowth_SVM_reg3_TP + RealIncomeGrowth_SVM_reg3_FP + RealIncomeGrowth_SVM_reg3_FN + RealIncomeGrowth_SVM_reg3_TN)
print(RealIncomeGrowth_SVM_reg3_ACC)

RealIncomeGrowthAccuracyDict.update({'RealIncomeGrowth_SVM_reg3_ACC': RealIncomeGrowth_SVM_reg3_ACC})
print(RealIncomeGrowthAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #1 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

RealIncomeGrowth_B_SVM_Model=LinearSVC(C=100)
RealIncomeGrowth_B_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
RealIncomeGrowth_b_svm_predict = RealIncomeGrowth_B_SVM_Model.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(RealIncomeGrowth_b_svm_predict)
print("\nActual:")
print(TestLabelsB)

RealIncomeGrowth_B_SVM_matrix = confusion_matrix(TestLabelsB, RealIncomeGrowth_b_svm_predict)
print("\nThe confusion matrix is:")
print(RealIncomeGrowth_B_SVM_matrix)
print("\n\n")

RealIncomeGrowth_svm_B_target_names = ['0','1']
print(classification_report(TestLabelsB, RealIncomeGrowth_b_svm_predict, target_names = RealIncomeGrowth_svm_B_target_names))

RealIncomeGrowth_SVM_bool_FP = RealIncomeGrowth_B_SVM_matrix[0][1] 
RealIncomeGrowth_SVM_bool_FN = RealIncomeGrowth_B_SVM_matrix[1][0]
RealIncomeGrowth_SVM_bool_TP = RealIncomeGrowth_B_SVM_matrix[1][1]
RealIncomeGrowth_SVM_bool_TN = RealIncomeGrowth_B_SVM_matrix[0][0]

# Overall accuracy
RealIncomeGrowth_SVM_bool_ACC = (RealIncomeGrowth_SVM_bool_TP + RealIncomeGrowth_SVM_bool_TN)/(RealIncomeGrowth_SVM_bool_TP + RealIncomeGrowth_SVM_bool_FP + RealIncomeGrowth_SVM_bool_FN + RealIncomeGrowth_SVM_bool_TN)
print(RealIncomeGrowth_SVM_bool_ACC)

RealIncomeGrowthAccuracyDict.update({'RealIncomeGrowth_SVM_bool_ACC': RealIncomeGrowth_SVM_bool_ACC})
print(RealIncomeGrowthAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #2 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

RealIncomeGrowth_B_SVM_Model2=LinearSVC(C=1)
RealIncomeGrowth_B_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
RealIncomeGrowth_b_svm_predict2 = RealIncomeGrowth_B_SVM_Model2.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(RealIncomeGrowth_b_svm_predict2)
print("\nActual:")
print(TestLabelsB)

RealIncomeGrowth_B_SVM_matrix2 = confusion_matrix(TestLabelsB, RealIncomeGrowth_b_svm_predict2)
print("\nThe confusion matrix is:")
print(RealIncomeGrowth_B_SVM_matrix2)
print("\n\n")

RealIncomeGrowth_svm_B_target_names2 = ['0','1']
print(classification_report(TestLabelsB, RealIncomeGrowth_b_svm_predict2, target_names = RealIncomeGrowth_svm_B_target_names2))

RealIncomeGrowth_SVM_bool2_FP = RealIncomeGrowth_B_SVM_matrix2[0][1] 
RealIncomeGrowth_SVM_bool2_FN = RealIncomeGrowth_B_SVM_matrix2[1][0]
RealIncomeGrowth_SVM_bool2_TP = RealIncomeGrowth_B_SVM_matrix2[1][1]
RealIncomeGrowth_SVM_bool2_TN = RealIncomeGrowth_B_SVM_matrix2[0][0]

# Overall accuracy
RealIncomeGrowth_SVM_bool2_ACC = (RealIncomeGrowth_SVM_bool2_TP + RealIncomeGrowth_SVM_bool2_TN)/(RealIncomeGrowth_SVM_bool2_TP + RealIncomeGrowth_SVM_bool2_FP + RealIncomeGrowth_SVM_bool2_FN + RealIncomeGrowth_SVM_bool2_TN)
print(RealIncomeGrowth_SVM_bool2_ACC)

RealIncomeGrowthAccuracyDict.update({'RealIncomeGrowth_SVM_bool2_ACC': RealIncomeGrowth_SVM_bool2_ACC})
print(RealIncomeGrowthAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #3 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

RealIncomeGrowth_B_SVM_Model3=LinearSVC(C=.01)
RealIncomeGrowth_B_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
RealIncomeGrowth_b_svm_predict3 = RealIncomeGrowth_B_SVM_Model3.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(RealIncomeGrowth_b_svm_predict3)
print("\nActual:")
print(TestLabelsB)

RealIncomeGrowth_B_SVM_matrix3 = confusion_matrix(TestLabelsB, RealIncomeGrowth_b_svm_predict3)
print("\nThe confusion matrix is:")
print(RealIncomeGrowth_B_SVM_matrix3)
print("\n\n")

RealIncomeGrowth_svm_B_target_names3 = ['0','1']
print(classification_report(TestLabelsB, RealIncomeGrowth_b_svm_predict3, target_names = RealIncomeGrowth_svm_B_target_names3))

RealIncomeGrowth_SVM_bool3_FP = RealIncomeGrowth_B_SVM_matrix3[0][1] 
RealIncomeGrowth_SVM_bool3_FN = RealIncomeGrowth_B_SVM_matrix3[1][0]
RealIncomeGrowth_SVM_bool3_TP = RealIncomeGrowth_B_SVM_matrix3[1][1]
RealIncomeGrowth_SVM_bool3_TN = RealIncomeGrowth_B_SVM_matrix3[0][0]

# Overall accuracy
RealIncomeGrowth_SVM_bool3_ACC = (RealIncomeGrowth_SVM_bool3_TP + RealIncomeGrowth_SVM_bool3_TN)/(RealIncomeGrowth_SVM_bool3_TP + RealIncomeGrowth_SVM_bool3_FP + RealIncomeGrowth_SVM_bool3_FN + RealIncomeGrowth_SVM_bool3_TN)
print(RealIncomeGrowth_SVM_bool3_ACC)

RealIncomeGrowthAccuracyDict.update({'RealIncomeGrowth_SVM_bool3_ACC': RealIncomeGrowth_SVM_bool3_ACC})
print(RealIncomeGrowthAccuracyDict)

#############################################
###########  SVM  for tfidf model #1 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

RealIncomeGrowth_tf_SVM_Model=LinearSVC(C=.001)
RealIncomeGrowth_tf_SVM_Model.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
RealIncomeGrowth_tf_svm_predict = RealIncomeGrowth_tf_SVM_Model.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(RealIncomeGrowth_tf_svm_predict)
print("\nActual:")
print(TestLabels_tf)

RealIncomeGrowth_tf_SVM_matrix = confusion_matrix(TestLabels_tf, RealIncomeGrowth_tf_svm_predict)
print("\nThe confusion matrix is:")
print(RealIncomeGrowth_tf_SVM_matrix)
print("\n\n")

RealIncomeGrowth_svm_tf_target_names = ['0','1']
print(classification_report(TestLabels_tf, RealIncomeGrowth_tf_svm_predict, target_names = RealIncomeGrowth_svm_tf_target_names))

RealIncomeGrowth_SVM_tf_FP = RealIncomeGrowth_tf_SVM_matrix[0][1] 
RealIncomeGrowth_SVM_tf_FN = RealIncomeGrowth_tf_SVM_matrix[1][0]
RealIncomeGrowth_SVM_tf_TP = RealIncomeGrowth_tf_SVM_matrix[1][1]
RealIncomeGrowth_SVM_tf_TN = RealIncomeGrowth_tf_SVM_matrix[0][0]

# Overall accuracy
RealIncomeGrowth_SVM_tf_ACC = (RealIncomeGrowth_SVM_tf_TP + RealIncomeGrowth_SVM_tf_TN)/(RealIncomeGrowth_SVM_tf_TP + RealIncomeGrowth_SVM_tf_FP + RealIncomeGrowth_SVM_tf_FN + RealIncomeGrowth_SVM_tf_TN)
print(RealIncomeGrowth_SVM_tf_ACC)

RealIncomeGrowthAccuracyDict.update({'RealIncomeGrowth_SVM_tf_ACC': RealIncomeGrowth_SVM_tf_ACC})
print(RealIncomeGrowthAccuracyDict)

#############################################
###########  SVM  for tfidf model #2 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

RealIncomeGrowth_tf_SVM_Model2=LinearSVC(C=1)
RealIncomeGrowth_tf_SVM_Model2.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
RealIncomeGrowth_tf_svm_predict2 = RealIncomeGrowth_tf_SVM_Model2.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(RealIncomeGrowth_tf_svm_predict2)
print("\nActual:")
print(TestLabels_tf)

RealIncomeGrowth_tf_SVM_matrix2 = confusion_matrix(TestLabels_tf, RealIncomeGrowth_tf_svm_predict2)
print("\nThe confusion matrix is:")
print(RealIncomeGrowth_tf_SVM_matrix2)
print("\n\n")

RealIncomeGrowth_svm_tf_target_names2 = ['0','1']
print(classification_report(TestLabels_tf, RealIncomeGrowth_tf_svm_predict2, target_names = RealIncomeGrowth_svm_tf_target_names2))

RealIncomeGrowth_SVM_tf2_FP = RealIncomeGrowth_tf_SVM_matrix2[0][1] 
RealIncomeGrowth_SVM_tf2_FN = RealIncomeGrowth_tf_SVM_matrix2[1][0]
RealIncomeGrowth_SVM_tf2_TP = RealIncomeGrowth_tf_SVM_matrix2[1][1]
RealIncomeGrowth_SVM_tf2_TN = RealIncomeGrowth_tf_SVM_matrix2[0][0]

# Overall accuracy
RealIncomeGrowth_SVM_tf2_ACC = (RealIncomeGrowth_SVM_tf2_TP + RealIncomeGrowth_SVM_tf2_TN)/(RealIncomeGrowth_SVM_tf2_TP + RealIncomeGrowth_SVM_tf2_FP + RealIncomeGrowth_SVM_tf2_FN + RealIncomeGrowth_SVM_tf2_TN)
print(RealIncomeGrowth_SVM_tf2_ACC)

RealIncomeGrowthAccuracyDict.update({'RealIncomeGrowth_SVM_tf2_ACC': RealIncomeGrowth_SVM_tf2_ACC})
print(RealIncomeGrowthAccuracyDict)

#############################################
###########  SVM  for tfidf model #3 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

RealIncomeGrowth_tf_SVM_Model3=LinearSVC(C=1)
RealIncomeGrowth_tf_SVM_Model3.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
RealIncomeGrowth_tf_svm_predict3 = RealIncomeGrowth_tf_SVM_Model3.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(RealIncomeGrowth_tf_svm_predict3)
print("\nActual:")
print(TestLabels_tf)

RealIncomeGrowth_tf_SVM_matrix3 = confusion_matrix(TestLabels_tf, RealIncomeGrowth_tf_svm_predict3)
print("\nThe confusion matrix is:")
print(RealIncomeGrowth_tf_SVM_matrix3)
print("\n\n")

RealIncomeGrowth_svm_tf_target_names3 = ['0','1']
print(classification_report(TestLabels_tf, RealIncomeGrowth_tf_svm_predict3, target_names = RealIncomeGrowth_svm_tf_target_names3))

RealIncomeGrowth_SVM_tf3_FP = RealIncomeGrowth_tf_SVM_matrix3[0][1] 
RealIncomeGrowth_SVM_tf3_FN = RealIncomeGrowth_tf_SVM_matrix3[1][0]
RealIncomeGrowth_SVM_tf3_TP = RealIncomeGrowth_tf_SVM_matrix3[1][1]
RealIncomeGrowth_SVM_tf3_TN = RealIncomeGrowth_tf_SVM_matrix3[0][0]

# Overall accuracy
RealIncomeGrowth_SVM_tf3_ACC = (RealIncomeGrowth_SVM_tf3_TP + RealIncomeGrowth_SVM_tf3_TN)/(RealIncomeGrowth_SVM_tf3_TP + RealIncomeGrowth_SVM_tf3_FP + RealIncomeGrowth_SVM_tf3_FN + RealIncomeGrowth_SVM_tf3_TN)
print(RealIncomeGrowth_SVM_tf3_ACC)

RealIncomeGrowthAccuracyDict.update({'RealIncomeGrowth_SVM_tf3_ACC': RealIncomeGrowth_SVM_tf3_ACC})
print(RealIncomeGrowthAccuracyDict)

#############################################
#######  SVM  with different kernels ########
#############################################
## running on Bernoulli model since it had best accuracy above
## i want to see if the accuracy gets improved in the different kernels

#############################################
###############  Sigmoid #1 #################
#############################################

RealIncomeGrowth_sig_SVM_Model=sklearn.svm.SVC(C=10, kernel='sigmoid', 
                           verbose=True, gamma="scale")
RealIncomeGrowth_sig_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
RealIncomeGrowth_sig_svm_predict = RealIncomeGrowth_sig_SVM_Model.predict(TestDFB_nolabels)
print("\nsig SVM prediction:\n")
print(RealIncomeGrowth_sig_svm_predict)
print("\nActual:")
print(TestLabelsB)

RealIncomeGrowth_sig_SVM_matrix = confusion_matrix(TestLabelsB, RealIncomeGrowth_sig_svm_predict)
print("\nThe sig confusion matrix is:")
print(RealIncomeGrowth_sig_SVM_matrix)
print("\n\n")

RealIncomeGrowth_svm_sig_target_names = ['0','1']
print(classification_report(TestLabelsB, RealIncomeGrowth_sig_svm_predict, target_names = RealIncomeGrowth_svm_sig_target_names))

RealIncomeGrowth_SVM_sig_FP = RealIncomeGrowth_sig_SVM_matrix[0][1] 
RealIncomeGrowth_SVM_sig_FN = RealIncomeGrowth_sig_SVM_matrix[1][0]
RealIncomeGrowth_SVM_sig_TP = RealIncomeGrowth_sig_SVM_matrix[1][1]
RealIncomeGrowth_SVM_sig_TN = RealIncomeGrowth_sig_SVM_matrix[0][0]

# Overall accuracy
RealIncomeGrowth_SVM_sig_ACC = (RealIncomeGrowth_SVM_sig_TP + RealIncomeGrowth_SVM_sig_TN)/(RealIncomeGrowth_SVM_sig_TP + RealIncomeGrowth_SVM_sig_FP + RealIncomeGrowth_SVM_sig_FN + RealIncomeGrowth_SVM_sig_TN)
print(RealIncomeGrowth_SVM_sig_ACC)

RealIncomeGrowthAccuracyDict.update({'RealIncomeGrowth_SVM_sig_ACC': RealIncomeGrowth_SVM_sig_ACC})
print(RealIncomeGrowthAccuracyDict)

#############################################
###############  Sigmoid #2 #################
#############################################

RealIncomeGrowth_sig_SVM_Model2=sklearn.svm.SVC(C=1000, kernel='sigmoid', 
                           verbose=True, gamma="scale")
RealIncomeGrowth_sig_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
RealIncomeGrowth_sig_svm_predict2 = RealIncomeGrowth_sig_SVM_Model2.predict(TestDFB_nolabels)
print("\nsig SVM prediction 2:\n")
print(RealIncomeGrowth_sig_svm_predict2)
print("\nActual:")
print(TestLabelsB)

RealIncomeGrowth_sig_SVM_matrix2 = confusion_matrix(TestLabelsB, RealIncomeGrowth_sig_svm_predict2)
print("\nThe sig 2 confusion matrix is:")
print(RealIncomeGrowth_sig_SVM_matrix2)
print("\n\n")

RealIncomeGrowth_svm_sig_target_names2 = ['0','1']
print(classification_report(TestLabelsB, RealIncomeGrowth_sig_svm_predict2, target_names = RealIncomeGrowth_svm_sig_target_names2))

RealIncomeGrowth_SVM_sig2_FP = RealIncomeGrowth_sig_SVM_matrix2[0][1] 
RealIncomeGrowth_SVM_sig2_FN = RealIncomeGrowth_sig_SVM_matrix2[1][0]
RealIncomeGrowth_SVM_sig2_TP = RealIncomeGrowth_sig_SVM_matrix2[1][1]
RealIncomeGrowth_SVM_sig2_TN = RealIncomeGrowth_sig_SVM_matrix2[0][0]

# Overall accuracy
RealIncomeGrowth_SVM_sig2_ACC = (RealIncomeGrowth_SVM_sig2_TP + RealIncomeGrowth_SVM_sig2_TN)/(RealIncomeGrowth_SVM_sig2_TP + RealIncomeGrowth_SVM_sig2_FP + RealIncomeGrowth_SVM_sig2_FN + RealIncomeGrowth_SVM_sig2_TN)
print(RealIncomeGrowth_SVM_sig2_ACC)

RealIncomeGrowthAccuracyDict.update({'RealIncomeGrowth_SVM_sig2_ACC': RealIncomeGrowth_SVM_sig2_ACC})
print(RealIncomeGrowthAccuracyDict)

#############################################
###############  Sigmoid #3 #################
#############################################

RealIncomeGrowth_sig_SVM_Model3=sklearn.svm.SVC(C=.001, kernel='sigmoid', 
                           verbose=True, gamma="scale")
RealIncomeGrowth_sig_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
RealIncomeGrowth_sig_svm_predict3 = RealIncomeGrowth_sig_SVM_Model3.predict(TestDFB_nolabels)
print("\nsig SVM prediction 3:\n")
print(RealIncomeGrowth_sig_svm_predict3)
print("\nActual:")
print(TestLabelsB)

RealIncomeGrowth_sig_SVM_matrix3 = confusion_matrix(TestLabelsB, RealIncomeGrowth_sig_svm_predict3)
print("\nThe sig 3 confusion matrix is:")
print(RealIncomeGrowth_sig_SVM_matrix3)
print("\n\n")

RealIncomeGrowth_svm_sig_target_names3 = ['0','1']
print(classification_report(TestLabelsB, RealIncomeGrowth_sig_svm_predict3, target_names = RealIncomeGrowth_svm_sig_target_names3))

RealIncomeGrowth_SVM_sig3_FP = RealIncomeGrowth_sig_SVM_matrix3[0][1] 
RealIncomeGrowth_SVM_sig3_FN = RealIncomeGrowth_sig_SVM_matrix3[1][0]
RealIncomeGrowth_SVM_sig3_TP = RealIncomeGrowth_sig_SVM_matrix3[1][1]
RealIncomeGrowth_SVM_sig3_TN = RealIncomeGrowth_sig_SVM_matrix3[0][0]

# Overall accuracy
RealIncomeGrowth_SVM_sig3_ACC = (RealIncomeGrowth_SVM_sig3_TP + RealIncomeGrowth_SVM_sig3_TN)/(RealIncomeGrowth_SVM_sig3_TP + RealIncomeGrowth_SVM_sig3_FP + RealIncomeGrowth_SVM_sig3_FN + RealIncomeGrowth_SVM_sig3_TN)
print(RealIncomeGrowth_SVM_sig3_ACC)

RealIncomeGrowthAccuracyDict.update({'RealIncomeGrowth_SVM_sig3_ACC': RealIncomeGrowth_SVM_sig3_ACC})
print(RealIncomeGrowthAccuracyDict)

#############################################
################  Poly #1 ###################
#############################################

RealIncomeGrowth_poly_SVM_Model=sklearn.svm.SVC(C=1000, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(RealIncomeGrowth_poly_SVM_Model)
RealIncomeGrowth_poly_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
RealIncomeGrowth_poly_svm_predict = RealIncomeGrowth_poly_SVM_Model.predict(TestDFB_nolabels)
print("\npoly SVM prediction:\n")
print(RealIncomeGrowth_poly_svm_predict)
print("\nActual:")
print(TestLabelsB)

RealIncomeGrowth_poly_SVM_matrix = confusion_matrix(TestLabelsB, RealIncomeGrowth_poly_svm_predict)
print("\nThe poly confusion matrix is:")
print(RealIncomeGrowth_poly_SVM_matrix)
print("\n\n")

RealIncomeGrowth_svm_poly_target_names = ['0','1']
print(classification_report(TestLabelsB, RealIncomeGrowth_poly_svm_predict, target_names = RealIncomeGrowth_svm_poly_target_names))

RealIncomeGrowth_SVM_poly_FP = RealIncomeGrowth_poly_SVM_matrix[0][1] 
RealIncomeGrowth_SVM_poly_FN = RealIncomeGrowth_poly_SVM_matrix[1][0]
RealIncomeGrowth_SVM_poly_TP = RealIncomeGrowth_poly_SVM_matrix[1][1]
RealIncomeGrowth_SVM_poly_TN = RealIncomeGrowth_poly_SVM_matrix[0][0]

# Overall accuracy
RealIncomeGrowth_SVM_poly_ACC = (RealIncomeGrowth_SVM_poly_TP + RealIncomeGrowth_SVM_poly_TN)/(RealIncomeGrowth_SVM_poly_TP + RealIncomeGrowth_SVM_poly_FP + RealIncomeGrowth_SVM_poly_FN + RealIncomeGrowth_SVM_poly_TN)
print(RealIncomeGrowth_SVM_poly_ACC)

RealIncomeGrowthAccuracyDict.update({'RealIncomeGrowth_SVM_poly_ACC': RealIncomeGrowth_SVM_poly_ACC})
print(RealIncomeGrowthAccuracyDict)

#############################################
################  Poly #2 ###################
#############################################

RealIncomeGrowth_poly_SVM_Model2=sklearn.svm.SVC(C=10, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(RealIncomeGrowth_poly_SVM_Model2)
RealIncomeGrowth_poly_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
RealIncomeGrowth_poly_svm_predict2 = RealIncomeGrowth_poly_SVM_Model2.predict(TestDFB_nolabels)
print("\npoly SVM 2 prediction:\n")
print(RealIncomeGrowth_poly_svm_predict2)
print("\nActual:")
print(TestLabelsB)

RealIncomeGrowth_poly_SVM_matrix2 = confusion_matrix(TestLabelsB, RealIncomeGrowth_poly_svm_predict2)
print("\nThe poly 2 confusion matrix is:")
print(RealIncomeGrowth_poly_SVM_matrix2)
print("\n\n")

RealIncomeGrowth_svm_poly_target_names2 = ['0','1']
print(classification_report(TestLabelsB, RealIncomeGrowth_poly_svm_predict2, target_names = RealIncomeGrowth_svm_poly_target_names2))

RealIncomeGrowth_SVM_poly2_FP = RealIncomeGrowth_poly_SVM_matrix2[0][1] 
RealIncomeGrowth_SVM_poly2_FN = RealIncomeGrowth_poly_SVM_matrix2[1][0]
RealIncomeGrowth_SVM_poly2_TP = RealIncomeGrowth_poly_SVM_matrix2[1][1]
RealIncomeGrowth_SVM_poly2_TN = RealIncomeGrowth_poly_SVM_matrix2[0][0]

# Overall accuracy
RealIncomeGrowth_SVM_poly2_ACC = (RealIncomeGrowth_SVM_poly2_TP + RealIncomeGrowth_SVM_poly2_TN)/(RealIncomeGrowth_SVM_poly2_TP + RealIncomeGrowth_SVM_poly2_FP + RealIncomeGrowth_SVM_poly2_FN + RealIncomeGrowth_SVM_poly2_TN)
print(RealIncomeGrowth_SVM_poly2_ACC)

RealIncomeGrowthAccuracyDict.update({'RealIncomeGrowth_SVM_poly2_ACC': RealIncomeGrowth_SVM_poly2_ACC})
print(RealIncomeGrowthAccuracyDict)

#############################################
################  Poly #3 ###################
#############################################

RealIncomeGrowth_poly_SVM_Model3=sklearn.svm.SVC(C=.001, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(RealIncomeGrowth_poly_SVM_Model3)
RealIncomeGrowth_poly_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
RealIncomeGrowth_poly_svm_predict3 = RealIncomeGrowth_poly_SVM_Model3.predict(TestDFB_nolabels)
print("\npoly SVM 3 prediction:\n")
print(RealIncomeGrowth_poly_svm_predict3)
print("\nActual:")
print(TestLabelsB)

RealIncomeGrowth_poly_SVM_matrix3 = confusion_matrix(TestLabelsB, RealIncomeGrowth_poly_svm_predict3)
print("\nThe poly 3 confusion matrix is:")
print(RealIncomeGrowth_poly_SVM_matrix3)
print("\n\n")

RealIncomeGrowth_svm_poly_target_names3 = ['0','1']
print(classification_report(TestLabelsB, RealIncomeGrowth_poly_svm_predict3, target_names = RealIncomeGrowth_svm_poly_target_names3))

RealIncomeGrowth_SVM_poly3_FP = RealIncomeGrowth_poly_SVM_matrix3[0][1] 
RealIncomeGrowth_SVM_poly3_FN = RealIncomeGrowth_poly_SVM_matrix3[1][0]
RealIncomeGrowth_SVM_poly3_TP = RealIncomeGrowth_poly_SVM_matrix3[1][1]
RealIncomeGrowth_SVM_poly3_TN = RealIncomeGrowth_poly_SVM_matrix3[0][0]

# Overall accuracy
RealIncomeGrowth_SVM_poly3_ACC = (RealIncomeGrowth_SVM_poly3_TP + RealIncomeGrowth_SVM_poly3_TN)/(RealIncomeGrowth_SVM_poly3_TP + RealIncomeGrowth_SVM_poly3_FP + RealIncomeGrowth_SVM_poly3_FN + RealIncomeGrowth_SVM_poly3_TN)
print(RealIncomeGrowth_SVM_poly3_ACC)

RealIncomeGrowthAccuracyDict.update({'RealIncomeGrowth_SVM_poly3_ACC': RealIncomeGrowth_SVM_poly3_ACC})
print(RealIncomeGrowthAccuracyDict)

RealIncomeGrowthVisDF = pd.DataFrame(RealIncomeGrowthAccuracyDict.items(), index = RealIncomeGrowthAccuracyDict.keys(), columns=['Model','Accuracy'])
print(RealIncomeGrowthVisDF)
SortedRealIncomeGrowthVisDF = RealIncomeGrowthVisDF.sort_values('Accuracy', ascending = [True])
print(SortedRealIncomeGrowthVisDF)

print(PartyAccuracyDict)
print(WinnerAccuracyDict)
print(SentimentAccuracyDict)
print(IncumCandAccuracyDict)
print(IncumPartyAccuracyDict)
print(UnemploymentAccuracyDict)
print(GDPAccuracyDict)
print(InflationAccuracyDict)
print(SatisfactionAccuracyDict)
print(RealIncomeGrowthAccuracyDict)

print(SortedPartyVisDF)
print(SortedWinnerVisDF)
print(SortedSentimentVisDF)
print(SortedIncumCandVisDF)
print(SortedIncumPartyVisDF)
print(SortedUnemploymentVisDF)
print(SortedGDPVisDF)
print(SortedInflationVisDF)
print(SortedSatisfactionVisDF)
print(SortedRealIncomeGrowthVisDF)

SortedPartyVisDF.plot.barh(y='Accuracy')
SortedWinnerVisDF.plot.barh(y='Accuracy')
SortedSentimentVisDF.plot.barh(y='Accuracy')
SortedIncumCandVisDF.plot.barh(y='Accuracy')
SortedIncumPartyVisDF.plot.barh(y='Accuracy')
SortedUnemploymentVisDF.plot.barh(y='Accuracy')
SortedGDPVisDF.plot.barh(y='Accuracy')
SortedInflationVisDF.plot.barh(y='Accuracy')
SortedSatisfactionVisDF.plot.barh(y='Accuracy')
SortedRealIncomeGrowthVisDF.plot.barh(y='Accuracy')

#print(IndexList)
#print(TextList)
#print(PartyList)
#print(WinnerList)
#print(SentimentList)
#print(IncumCandList)
#print(IncumPartyList)
#print(UnemploymentList)
#print(GDPList)
#print(InflationList)
#print(SatisfactionList)
#print(RealIncomeGrowthList)
print(PresApprovalList)
print(IncomeTaxList)
print(DjiaVolumeList)
print(CpiList)
print(CciIndexList)

#################################################################################################
#################################################################################################
### everything from this point down is a matter of find and replace for each of the variables ###
#################################################################################################
#################################################################################################

# the rows weren't all populated for the variables below so an edited text list is needed

### starting to build a model for PresApproval
df_PresApproval = rawfile.copy(deep=True)
df_PresApproval = df_PresApproval.filter(['id', 'pres_approval', 'comb_text'])
df_PresApproval = df_PresApproval[df_PresApproval['pres_approval'].notna()]
print(df_PresApproval)

PresApprovalList = []
TextList = []
IndexList = []

for row in df_PresApproval.itertuples():
    PresApprovallabel = row.pres_approval
    textlabel = row.comb_text
    textlabel = textlabel.replace('\n',' ') # remove \n
    textlabel = textlabel.replace("'","'")
    textlabel = textlabel.replace('\t',' ') # remove \t
    #textlabel = textlabel.replace(',',' ') # remove ,
    #textlabel = textlabel.replace('.',' ') # remove .
    #textlabel = textlabel.replace('"',' ') # remove "
    #textlabel = textlabel.replace("'",' ') # remove '
    textlabel = textlabel.replace('nan',' ') # remove nan
    #textlabel = textlabel.replace('-',' ') # remove -
    #textlabel = textlabel.replace(' - ',' ') # remove -
    #textlabel = textlabel.replace('?',' ') # remove ?
    #textlabel = textlabel.replace('!',' ') # remove !
    #textlabel = textlabel.replace(':',' ') # remove :
    #textlabel = textlabel.replace(';',' ') # remove ;
    #textlabel = textlabel.replace('  ',' ') # remove double spaces
    indexlabel = row.id
    TextList.append(textlabel)
    IndexList.append(indexlabel)
    PresApprovalList.append(PresApprovallabel)

PresApprovalList = [ int(x) for x in PresApprovalList ]
print(PresApprovalList)

X_text=MyVect.fit_transform(TextList)

Bool_X_text=MyVect.fit_transform(TextList)

tf_X_text=MyVect.fit_transform(TextList)

## Hmm - that's not quite what we want...
## Let's get the feature names which ARE the words
ColumnNamesText=MyVect.get_feature_names()
print(ColumnNamesText)

## OK good - but we want a document topic model A DTM (matrix of counts)
VectDF=pd.DataFrame(X_text.toarray(), columns=ColumnNamesText, index = IndexList)
print(VectDF)

bool_VectDF=pd.DataFrame(Bool_X_text.toarray(), columns=ColumnNamesText, index = IndexList)
print(bool_VectDF)

tf_VectDF=pd.DataFrame(tf_X_text.toarray(), columns=ColumnNamesText, index = IndexList)
print(tf_VectDF)

PresApprovalVectDF = VectDF.copy(deep=True)
PresApprovalVectDF.insert(loc=0, column='LABEL', value=PresApprovalList)
print(PresApprovalVectDF)

bool_PresApprovalVectDF = bool_VectDF.copy(deep=True)
bool_PresApprovalVectDF.insert(loc=0, column='LABEL', value=PresApprovalList)
print(bool_PresApprovalVectDF)

tf_PresApprovalVectDF = tf_VectDF.copy(deep=True)
tf_PresApprovalVectDF.insert(loc=0, column='LABEL', value=PresApprovalList)
print(tf_PresApprovalVectDF)


##--------------------------------------------------------
####################################################################
###################### model Data Build ############################
####################################################################
##--------------------------------------------------------
from sklearn.model_selection import train_test_split

## Copy everything below here to rebuild for PresApproval data
TrainDF, TestDF = train_test_split(PresApprovalVectDF, test_size=0.3)
print(TrainDF.head())

## Now we have a training set and a testing set. 
print("\nThe training set is:")
print(TrainDF)
print("\nThe testing set is:")
print(TestDF)

## IMPORTANT - YOU CANNOT LEAVE LABELS ON THE TEST SET
## Save labels
TestLabels=TestDF["LABEL"]
print(TestLabels)
## remove labels
TestDF_nolabels = TestDF.drop(["LABEL"], axis=1)
print(TestDF_nolabels)

TrainDF_nolabels=TrainDF.drop(["LABEL"], axis=1)
print(TrainDF_nolabels)

TrainLabels=TrainDF["LABEL"]
print(TrainLabels)

############# Do the same for the binary dataframe
## Create Train/Test for TrainDFB
TrainDFB, TestDFB = train_test_split(bool_PresApprovalVectDF, test_size=0.3)
print(TrainDFB.head())

TrainDFB_nolabels=TrainDFB.drop(["LABEL"], axis=1)
print(TrainDFB_nolabels)
TrainLabelsB=TrainDFB["LABEL"]
print(TrainLabelsB)

TestDFB_nolabels=TestDFB.drop(["LABEL"], axis=1)
print(TestDFB_nolabels)
TestLabelsB=TestDFB["LABEL"]
print(TestLabelsB)

############# Do the same for the tfidf dataframe
## Create Train/Test for TrainDF tfidf
TrainDF_tf, TestDF_tf = train_test_split(tf_PresApprovalVectDF, test_size=0.3)
print(TrainDF_tf.head())

TrainDF_tf_nolabels=TrainDF_tf.drop(["LABEL"], axis=1)
print(TrainDF_tf_nolabels)
TrainLabels_tf=TrainDF_tf["LABEL"]
print(TrainLabels_tf)

TestDF_tf_nolabels=TestDF_tf.drop(["LABEL"], axis=1)
print(TestDF_tf_nolabels)
TestLabels_tf=TestDF_tf["LABEL"]
print(TestLabels_tf)



#############################################
###########  SVM  for regular model #1 ######
#############################################
from sklearn.svm import LinearSVC
print(TrainDF_nolabels.head())
print(TrainLabels.head())

PresApproval_SVM_Model=LinearSVC(C=.01)
PresApproval_SVM_Model.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
PresApproval_svm_predict = PresApproval_SVM_Model.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(PresApproval_svm_predict)
print("Actual:")
print(TestLabels)

from sklearn.metrics import confusion_matrix
PresApproval_SVM_matrix = confusion_matrix(TestLabels, PresApproval_svm_predict)
print("\nThe confusion matrix is:")
print(PresApproval_SVM_matrix)
print("\n\n")

from sklearn.metrics import classification_report
PresApproval_svm_target_names = ['0','1']
print(classification_report(TestLabels, PresApproval_svm_predict, target_names = PresApproval_svm_target_names))

PresApproval_SVM_reg_FP = PresApproval_SVM_matrix[0][1] 
PresApproval_SVM_reg_FN = PresApproval_SVM_matrix[1][0]
PresApproval_SVM_reg_TP = PresApproval_SVM_matrix[1][1]
PresApproval_SVM_reg_TN = PresApproval_SVM_matrix[0][0]

# Overall accuracy
PresApproval_SVM_reg_ACC = (PresApproval_SVM_reg_TP + PresApproval_SVM_reg_TN)/(PresApproval_SVM_reg_TP + PresApproval_SVM_reg_FP + PresApproval_SVM_reg_FN + PresApproval_SVM_reg_TN)
print(PresApproval_SVM_reg_ACC)

PresApprovalAccuracyDict = {}
PresApprovalAccuracyDict.update({'PresApproval_SVM_reg_ACC': PresApproval_SVM_reg_ACC})
print(PresApprovalAccuracyDict)

#############################################
###########  SVM  for regular model #2 ######
#############################################

PresApproval_SVM_Model2=LinearSVC(C=1)
PresApproval_SVM_Model2.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
PresApproval_svm_predict2 = PresApproval_SVM_Model2.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(PresApproval_svm_predict2)
print("Actual:")
print(TestLabels)

PresApproval_SVM_matrix2 = confusion_matrix(TestLabels, PresApproval_svm_predict2)
print("\nThe confusion matrix is:")
print(PresApproval_SVM_matrix2)
print("\n\n")

PresApproval_svm_target_names2 = ['0','1']
print(classification_report(TestLabels, PresApproval_svm_predict2, target_names = PresApproval_svm_target_names2))

PresApproval_SVM_reg2_FP = PresApproval_SVM_matrix2[0][1] 
PresApproval_SVM_reg2_FN = PresApproval_SVM_matrix2[1][0]
PresApproval_SVM_reg2_TP = PresApproval_SVM_matrix2[1][1]
PresApproval_SVM_reg2_TN = PresApproval_SVM_matrix2[0][0]

# Overall accuracy
PresApproval_SVM_reg2_ACC = (PresApproval_SVM_reg2_TP + PresApproval_SVM_reg2_TN)/(PresApproval_SVM_reg2_TP + PresApproval_SVM_reg2_FP + PresApproval_SVM_reg2_FN + PresApproval_SVM_reg2_TN)
print(PresApproval_SVM_reg2_ACC)

PresApprovalAccuracyDict.update({'PresApproval_SVM_reg2_ACC': PresApproval_SVM_reg2_ACC})
print(PresApprovalAccuracyDict)

#############################################
###########  SVM  for regular model #3 ######
#############################################

PresApproval_SVM_Model3=LinearSVC(C=100)
PresApproval_SVM_Model3.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
PresApproval_svm_predict3 = PresApproval_SVM_Model3.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(PresApproval_svm_predict3)
print("Actual:")
print(TestLabels)

PresApproval_SVM_matrix3 = confusion_matrix(TestLabels, PresApproval_svm_predict3)
print("\nThe confusion matrix is:")
print(PresApproval_SVM_matrix3)
print("\n\n")

from sklearn.metrics import classification_report
PresApproval_svm_target_names3 = ['0','1']
print(classification_report(TestLabels, PresApproval_svm_predict3, target_names = PresApproval_svm_target_names3))

PresApproval_SVM_reg3_FP = PresApproval_SVM_matrix3[0][1] 
PresApproval_SVM_reg3_FN = PresApproval_SVM_matrix3[1][0]
PresApproval_SVM_reg3_TP = PresApproval_SVM_matrix3[1][1]
PresApproval_SVM_reg3_TN = PresApproval_SVM_matrix3[0][0]

# Overall accuracy
PresApproval_SVM_reg3_ACC = (PresApproval_SVM_reg3_TP + PresApproval_SVM_reg3_TN)/(PresApproval_SVM_reg3_TP + PresApproval_SVM_reg3_FP + PresApproval_SVM_reg3_FN + PresApproval_SVM_reg3_TN)
print(PresApproval_SVM_reg3_ACC)

PresApprovalAccuracyDict.update({'PresApproval_SVM_reg3_ACC': PresApproval_SVM_reg3_ACC})
print(PresApprovalAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #1 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

PresApproval_B_SVM_Model=LinearSVC(C=100)
PresApproval_B_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
PresApproval_b_svm_predict = PresApproval_B_SVM_Model.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(PresApproval_b_svm_predict)
print("\nActual:")
print(TestLabelsB)

PresApproval_B_SVM_matrix = confusion_matrix(TestLabelsB, PresApproval_b_svm_predict)
print("\nThe confusion matrix is:")
print(PresApproval_B_SVM_matrix)
print("\n\n")

PresApproval_svm_B_target_names = ['0','1']
print(classification_report(TestLabelsB, PresApproval_b_svm_predict, target_names = PresApproval_svm_B_target_names))

PresApproval_SVM_bool_FP = PresApproval_B_SVM_matrix[0][1] 
PresApproval_SVM_bool_FN = PresApproval_B_SVM_matrix[1][0]
PresApproval_SVM_bool_TP = PresApproval_B_SVM_matrix[1][1]
PresApproval_SVM_bool_TN = PresApproval_B_SVM_matrix[0][0]

# Overall accuracy
PresApproval_SVM_bool_ACC = (PresApproval_SVM_bool_TP + PresApproval_SVM_bool_TN)/(PresApproval_SVM_bool_TP + PresApproval_SVM_bool_FP + PresApproval_SVM_bool_FN + PresApproval_SVM_bool_TN)
print(PresApproval_SVM_bool_ACC)

PresApprovalAccuracyDict.update({'PresApproval_SVM_bool_ACC': PresApproval_SVM_bool_ACC})
print(PresApprovalAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #2 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

PresApproval_B_SVM_Model2=LinearSVC(C=1)
PresApproval_B_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
PresApproval_b_svm_predict2 = PresApproval_B_SVM_Model2.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(PresApproval_b_svm_predict2)
print("\nActual:")
print(TestLabelsB)

PresApproval_B_SVM_matrix2 = confusion_matrix(TestLabelsB, PresApproval_b_svm_predict2)
print("\nThe confusion matrix is:")
print(PresApproval_B_SVM_matrix2)
print("\n\n")

PresApproval_svm_B_target_names2 = ['0','1']
print(classification_report(TestLabelsB, PresApproval_b_svm_predict2, target_names = PresApproval_svm_B_target_names2))

PresApproval_SVM_bool2_FP = PresApproval_B_SVM_matrix2[0][1] 
PresApproval_SVM_bool2_FN = PresApproval_B_SVM_matrix2[1][0]
PresApproval_SVM_bool2_TP = PresApproval_B_SVM_matrix2[1][1]
PresApproval_SVM_bool2_TN = PresApproval_B_SVM_matrix2[0][0]

# Overall accuracy
PresApproval_SVM_bool2_ACC = (PresApproval_SVM_bool2_TP + PresApproval_SVM_bool2_TN)/(PresApproval_SVM_bool2_TP + PresApproval_SVM_bool2_FP + PresApproval_SVM_bool2_FN + PresApproval_SVM_bool2_TN)
print(PresApproval_SVM_bool2_ACC)

PresApprovalAccuracyDict.update({'PresApproval_SVM_bool2_ACC': PresApproval_SVM_bool2_ACC})
print(PresApprovalAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #3 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

PresApproval_B_SVM_Model3=LinearSVC(C=.01)
PresApproval_B_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
PresApproval_b_svm_predict3 = PresApproval_B_SVM_Model3.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(PresApproval_b_svm_predict3)
print("\nActual:")
print(TestLabelsB)

PresApproval_B_SVM_matrix3 = confusion_matrix(TestLabelsB, PresApproval_b_svm_predict3)
print("\nThe confusion matrix is:")
print(PresApproval_B_SVM_matrix3)
print("\n\n")

PresApproval_svm_B_target_names3 = ['0','1']
print(classification_report(TestLabelsB, PresApproval_b_svm_predict3, target_names = PresApproval_svm_B_target_names3))

PresApproval_SVM_bool3_FP = PresApproval_B_SVM_matrix3[0][1] 
PresApproval_SVM_bool3_FN = PresApproval_B_SVM_matrix3[1][0]
PresApproval_SVM_bool3_TP = PresApproval_B_SVM_matrix3[1][1]
PresApproval_SVM_bool3_TN = PresApproval_B_SVM_matrix3[0][0]

# Overall accuracy
PresApproval_SVM_bool3_ACC = (PresApproval_SVM_bool3_TP + PresApproval_SVM_bool3_TN)/(PresApproval_SVM_bool3_TP + PresApproval_SVM_bool3_FP + PresApproval_SVM_bool3_FN + PresApproval_SVM_bool3_TN)
print(PresApproval_SVM_bool3_ACC)

PresApprovalAccuracyDict.update({'PresApproval_SVM_bool3_ACC': PresApproval_SVM_bool3_ACC})
print(PresApprovalAccuracyDict)

#############################################
###########  SVM  for tfidf model #1 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

PresApproval_tf_SVM_Model=LinearSVC(C=.001)
PresApproval_tf_SVM_Model.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
PresApproval_tf_svm_predict = PresApproval_tf_SVM_Model.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(PresApproval_tf_svm_predict)
print("\nActual:")
print(TestLabels_tf)

PresApproval_tf_SVM_matrix = confusion_matrix(TestLabels_tf, PresApproval_tf_svm_predict)
print("\nThe confusion matrix is:")
print(PresApproval_tf_SVM_matrix)
print("\n\n")

PresApproval_svm_tf_target_names = ['0','1']
print(classification_report(TestLabels_tf, PresApproval_tf_svm_predict, target_names = PresApproval_svm_tf_target_names))

PresApproval_SVM_tf_FP = PresApproval_tf_SVM_matrix[0][1] 
PresApproval_SVM_tf_FN = PresApproval_tf_SVM_matrix[1][0]
PresApproval_SVM_tf_TP = PresApproval_tf_SVM_matrix[1][1]
PresApproval_SVM_tf_TN = PresApproval_tf_SVM_matrix[0][0]

# Overall accuracy
PresApproval_SVM_tf_ACC = (PresApproval_SVM_tf_TP + PresApproval_SVM_tf_TN)/(PresApproval_SVM_tf_TP + PresApproval_SVM_tf_FP + PresApproval_SVM_tf_FN + PresApproval_SVM_tf_TN)
print(PresApproval_SVM_tf_ACC)

PresApprovalAccuracyDict.update({'PresApproval_SVM_tf_ACC': PresApproval_SVM_tf_ACC})
print(PresApprovalAccuracyDict)

#############################################
###########  SVM  for tfidf model #2 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

PresApproval_tf_SVM_Model2=LinearSVC(C=1)
PresApproval_tf_SVM_Model2.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
PresApproval_tf_svm_predict2 = PresApproval_tf_SVM_Model2.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(PresApproval_tf_svm_predict2)
print("\nActual:")
print(TestLabels_tf)

PresApproval_tf_SVM_matrix2 = confusion_matrix(TestLabels_tf, PresApproval_tf_svm_predict2)
print("\nThe confusion matrix is:")
print(PresApproval_tf_SVM_matrix2)
print("\n\n")

PresApproval_svm_tf_target_names2 = ['0','1']
print(classification_report(TestLabels_tf, PresApproval_tf_svm_predict2, target_names = PresApproval_svm_tf_target_names2))

PresApproval_SVM_tf2_FP = PresApproval_tf_SVM_matrix2[0][1] 
PresApproval_SVM_tf2_FN = PresApproval_tf_SVM_matrix2[1][0]
PresApproval_SVM_tf2_TP = PresApproval_tf_SVM_matrix2[1][1]
PresApproval_SVM_tf2_TN = PresApproval_tf_SVM_matrix2[0][0]

# Overall accuracy
PresApproval_SVM_tf2_ACC = (PresApproval_SVM_tf2_TP + PresApproval_SVM_tf2_TN)/(PresApproval_SVM_tf2_TP + PresApproval_SVM_tf2_FP + PresApproval_SVM_tf2_FN + PresApproval_SVM_tf2_TN)
print(PresApproval_SVM_tf2_ACC)

PresApprovalAccuracyDict.update({'PresApproval_SVM_tf2_ACC': PresApproval_SVM_tf2_ACC})
print(PresApprovalAccuracyDict)

#############################################
###########  SVM  for tfidf model #3 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

PresApproval_tf_SVM_Model3=LinearSVC(C=1)
PresApproval_tf_SVM_Model3.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
PresApproval_tf_svm_predict3 = PresApproval_tf_SVM_Model3.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(PresApproval_tf_svm_predict3)
print("\nActual:")
print(TestLabels_tf)

PresApproval_tf_SVM_matrix3 = confusion_matrix(TestLabels_tf, PresApproval_tf_svm_predict3)
print("\nThe confusion matrix is:")
print(PresApproval_tf_SVM_matrix3)
print("\n\n")

PresApproval_svm_tf_target_names3 = ['0','1']
print(classification_report(TestLabels_tf, PresApproval_tf_svm_predict3, target_names = PresApproval_svm_tf_target_names3))

PresApproval_SVM_tf3_FP = PresApproval_tf_SVM_matrix3[0][1] 
PresApproval_SVM_tf3_FN = PresApproval_tf_SVM_matrix3[1][0]
PresApproval_SVM_tf3_TP = PresApproval_tf_SVM_matrix3[1][1]
PresApproval_SVM_tf3_TN = PresApproval_tf_SVM_matrix3[0][0]

# Overall accuracy
PresApproval_SVM_tf3_ACC = (PresApproval_SVM_tf3_TP + PresApproval_SVM_tf3_TN)/(PresApproval_SVM_tf3_TP + PresApproval_SVM_tf3_FP + PresApproval_SVM_tf3_FN + PresApproval_SVM_tf3_TN)
print(PresApproval_SVM_tf3_ACC)

PresApprovalAccuracyDict.update({'PresApproval_SVM_tf3_ACC': PresApproval_SVM_tf3_ACC})
print(PresApprovalAccuracyDict)

#############################################
#######  SVM  with different kernels ########
#############################################
## running on Bernoulli model since it had best accuracy above
## i want to see if the accuracy gets improved in the different kernels

#############################################
###############  Sigmoid #1 #################
#############################################

PresApproval_sig_SVM_Model=sklearn.svm.SVC(C=10, kernel='sigmoid', 
                           verbose=True, gamma="scale")
PresApproval_sig_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
PresApproval_sig_svm_predict = PresApproval_sig_SVM_Model.predict(TestDFB_nolabels)
print("\nsig SVM prediction:\n")
print(PresApproval_sig_svm_predict)
print("\nActual:")
print(TestLabelsB)

PresApproval_sig_SVM_matrix = confusion_matrix(TestLabelsB, PresApproval_sig_svm_predict)
print("\nThe sig confusion matrix is:")
print(PresApproval_sig_SVM_matrix)
print("\n\n")

PresApproval_svm_sig_target_names = ['0','1']
print(classification_report(TestLabelsB, PresApproval_sig_svm_predict, target_names = PresApproval_svm_sig_target_names))

PresApproval_SVM_sig_FP = PresApproval_sig_SVM_matrix[0][1] 
PresApproval_SVM_sig_FN = PresApproval_sig_SVM_matrix[1][0]
PresApproval_SVM_sig_TP = PresApproval_sig_SVM_matrix[1][1]
PresApproval_SVM_sig_TN = PresApproval_sig_SVM_matrix[0][0]

# Overall accuracy
PresApproval_SVM_sig_ACC = (PresApproval_SVM_sig_TP + PresApproval_SVM_sig_TN)/(PresApproval_SVM_sig_TP + PresApproval_SVM_sig_FP + PresApproval_SVM_sig_FN + PresApproval_SVM_sig_TN)
print(PresApproval_SVM_sig_ACC)

PresApprovalAccuracyDict.update({'PresApproval_SVM_sig_ACC': PresApproval_SVM_sig_ACC})
print(PresApprovalAccuracyDict)

#############################################
###############  Sigmoid #2 #################
#############################################

PresApproval_sig_SVM_Model2=sklearn.svm.SVC(C=1000, kernel='sigmoid', 
                           verbose=True, gamma="scale")
PresApproval_sig_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
PresApproval_sig_svm_predict2 = PresApproval_sig_SVM_Model2.predict(TestDFB_nolabels)
print("\nsig SVM prediction 2:\n")
print(PresApproval_sig_svm_predict2)
print("\nActual:")
print(TestLabelsB)

PresApproval_sig_SVM_matrix2 = confusion_matrix(TestLabelsB, PresApproval_sig_svm_predict2)
print("\nThe sig 2 confusion matrix is:")
print(PresApproval_sig_SVM_matrix2)
print("\n\n")

PresApproval_svm_sig_target_names2 = ['0','1']
print(classification_report(TestLabelsB, PresApproval_sig_svm_predict2, target_names = PresApproval_svm_sig_target_names2))

PresApproval_SVM_sig2_FP = PresApproval_sig_SVM_matrix2[0][1] 
PresApproval_SVM_sig2_FN = PresApproval_sig_SVM_matrix2[1][0]
PresApproval_SVM_sig2_TP = PresApproval_sig_SVM_matrix2[1][1]
PresApproval_SVM_sig2_TN = PresApproval_sig_SVM_matrix2[0][0]

# Overall accuracy
PresApproval_SVM_sig2_ACC = (PresApproval_SVM_sig2_TP + PresApproval_SVM_sig2_TN)/(PresApproval_SVM_sig2_TP + PresApproval_SVM_sig2_FP + PresApproval_SVM_sig2_FN + PresApproval_SVM_sig2_TN)
print(PresApproval_SVM_sig2_ACC)

PresApprovalAccuracyDict.update({'PresApproval_SVM_sig2_ACC': PresApproval_SVM_sig2_ACC})
print(PresApprovalAccuracyDict)

#############################################
###############  Sigmoid #3 #################
#############################################

PresApproval_sig_SVM_Model3=sklearn.svm.SVC(C=.001, kernel='sigmoid', 
                           verbose=True, gamma="scale")
PresApproval_sig_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
PresApproval_sig_svm_predict3 = PresApproval_sig_SVM_Model3.predict(TestDFB_nolabels)
print("\nsig SVM prediction 3:\n")
print(PresApproval_sig_svm_predict3)
print("\nActual:")
print(TestLabelsB)

PresApproval_sig_SVM_matrix3 = confusion_matrix(TestLabelsB, PresApproval_sig_svm_predict3)
print("\nThe sig 3 confusion matrix is:")
print(PresApproval_sig_SVM_matrix3)
print("\n\n")

PresApproval_svm_sig_target_names3 = ['0','1']
print(classification_report(TestLabelsB, PresApproval_sig_svm_predict3, target_names = PresApproval_svm_sig_target_names3))

PresApproval_SVM_sig3_FP = PresApproval_sig_SVM_matrix3[0][1] 
PresApproval_SVM_sig3_FN = PresApproval_sig_SVM_matrix3[1][0]
PresApproval_SVM_sig3_TP = PresApproval_sig_SVM_matrix3[1][1]
PresApproval_SVM_sig3_TN = PresApproval_sig_SVM_matrix3[0][0]

# Overall accuracy
PresApproval_SVM_sig3_ACC = (PresApproval_SVM_sig3_TP + PresApproval_SVM_sig3_TN)/(PresApproval_SVM_sig3_TP + PresApproval_SVM_sig3_FP + PresApproval_SVM_sig3_FN + PresApproval_SVM_sig3_TN)
print(PresApproval_SVM_sig3_ACC)

PresApprovalAccuracyDict.update({'PresApproval_SVM_sig3_ACC': PresApproval_SVM_sig3_ACC})
print(PresApprovalAccuracyDict)

#############################################
################  Poly #1 ###################
#############################################

PresApproval_poly_SVM_Model=sklearn.svm.SVC(C=1000, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(PresApproval_poly_SVM_Model)
PresApproval_poly_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
PresApproval_poly_svm_predict = PresApproval_poly_SVM_Model.predict(TestDFB_nolabels)
print("\npoly SVM prediction:\n")
print(PresApproval_poly_svm_predict)
print("\nActual:")
print(TestLabelsB)

PresApproval_poly_SVM_matrix = confusion_matrix(TestLabelsB, PresApproval_poly_svm_predict)
print("\nThe poly confusion matrix is:")
print(PresApproval_poly_SVM_matrix)
print("\n\n")

PresApproval_svm_poly_target_names = ['0','1']
print(classification_report(TestLabelsB, PresApproval_poly_svm_predict, target_names = PresApproval_svm_poly_target_names))

PresApproval_SVM_poly_FP = PresApproval_poly_SVM_matrix[0][1] 
PresApproval_SVM_poly_FN = PresApproval_poly_SVM_matrix[1][0]
PresApproval_SVM_poly_TP = PresApproval_poly_SVM_matrix[1][1]
PresApproval_SVM_poly_TN = PresApproval_poly_SVM_matrix[0][0]

# Overall accuracy
PresApproval_SVM_poly_ACC = (PresApproval_SVM_poly_TP + PresApproval_SVM_poly_TN)/(PresApproval_SVM_poly_TP + PresApproval_SVM_poly_FP + PresApproval_SVM_poly_FN + PresApproval_SVM_poly_TN)
print(PresApproval_SVM_poly_ACC)

PresApprovalAccuracyDict.update({'PresApproval_SVM_poly_ACC': PresApproval_SVM_poly_ACC})
print(PresApprovalAccuracyDict)

#############################################
################  Poly #2 ###################
#############################################

PresApproval_poly_SVM_Model2=sklearn.svm.SVC(C=10, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(PresApproval_poly_SVM_Model2)
PresApproval_poly_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
PresApproval_poly_svm_predict2 = PresApproval_poly_SVM_Model2.predict(TestDFB_nolabels)
print("\npoly SVM 2 prediction:\n")
print(PresApproval_poly_svm_predict2)
print("\nActual:")
print(TestLabelsB)

PresApproval_poly_SVM_matrix2 = confusion_matrix(TestLabelsB, PresApproval_poly_svm_predict2)
print("\nThe poly 2 confusion matrix is:")
print(PresApproval_poly_SVM_matrix2)
print("\n\n")

PresApproval_svm_poly_target_names2 = ['0','1']
print(classification_report(TestLabelsB, PresApproval_poly_svm_predict2, target_names = PresApproval_svm_poly_target_names2))

PresApproval_SVM_poly2_FP = PresApproval_poly_SVM_matrix2[0][1] 
PresApproval_SVM_poly2_FN = PresApproval_poly_SVM_matrix2[1][0]
PresApproval_SVM_poly2_TP = PresApproval_poly_SVM_matrix2[1][1]
PresApproval_SVM_poly2_TN = PresApproval_poly_SVM_matrix2[0][0]

# Overall accuracy
PresApproval_SVM_poly2_ACC = (PresApproval_SVM_poly2_TP + PresApproval_SVM_poly2_TN)/(PresApproval_SVM_poly2_TP + PresApproval_SVM_poly2_FP + PresApproval_SVM_poly2_FN + PresApproval_SVM_poly2_TN)
print(PresApproval_SVM_poly2_ACC)

PresApprovalAccuracyDict.update({'PresApproval_SVM_poly2_ACC': PresApproval_SVM_poly2_ACC})
print(PresApprovalAccuracyDict)

#############################################
################  Poly #3 ###################
#############################################

PresApproval_poly_SVM_Model3=sklearn.svm.SVC(C=.001, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(PresApproval_poly_SVM_Model3)
PresApproval_poly_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
PresApproval_poly_svm_predict3 = PresApproval_poly_SVM_Model3.predict(TestDFB_nolabels)
print("\npoly SVM 3 prediction:\n")
print(PresApproval_poly_svm_predict3)
print("\nActual:")
print(TestLabelsB)

PresApproval_poly_SVM_matrix3 = confusion_matrix(TestLabelsB, PresApproval_poly_svm_predict3)
print("\nThe poly 3 confusion matrix is:")
print(PresApproval_poly_SVM_matrix3)
print("\n\n")

PresApproval_svm_poly_target_names3 = ['0','1']
print(classification_report(TestLabelsB, PresApproval_poly_svm_predict3, target_names = PresApproval_svm_poly_target_names3))

PresApproval_SVM_poly3_FP = PresApproval_poly_SVM_matrix3[0][1] 
PresApproval_SVM_poly3_FN = PresApproval_poly_SVM_matrix3[1][0]
PresApproval_SVM_poly3_TP = PresApproval_poly_SVM_matrix3[1][1]
PresApproval_SVM_poly3_TN = PresApproval_poly_SVM_matrix3[0][0]

# Overall accuracy
PresApproval_SVM_poly3_ACC = (PresApproval_SVM_poly3_TP + PresApproval_SVM_poly3_TN)/(PresApproval_SVM_poly3_TP + PresApproval_SVM_poly3_FP + PresApproval_SVM_poly3_FN + PresApproval_SVM_poly3_TN)
print(PresApproval_SVM_poly3_ACC)

PresApprovalAccuracyDict.update({'PresApproval_SVM_poly3_ACC': PresApproval_SVM_poly3_ACC})
print(PresApprovalAccuracyDict)

PresApprovalVisDF = pd.DataFrame(PresApprovalAccuracyDict.items(), index = PresApprovalAccuracyDict.keys(), columns=['Model','Accuracy'])
print(PresApprovalVisDF)
SortedPresApprovalVisDF = PresApprovalVisDF.sort_values('Accuracy', ascending = [True])
print(SortedPresApprovalVisDF)

print(PartyAccuracyDict)
print(WinnerAccuracyDict)
print(SentimentAccuracyDict)
print(IncumCandAccuracyDict)
print(IncumPartyAccuracyDict)
print(UnemploymentAccuracyDict)
print(GDPAccuracyDict)
print(InflationAccuracyDict)
print(SatisfactionAccuracyDict)
print(RealIncomeGrowthAccuracyDict)
print(PresApprovalAccuracyDict)

print(SortedPartyVisDF)
print(SortedWinnerVisDF)
print(SortedSentimentVisDF)
print(SortedIncumCandVisDF)
print(SortedIncumPartyVisDF)
print(SortedUnemploymentVisDF)
print(SortedGDPVisDF)
print(SortedInflationVisDF)
print(SortedSatisfactionVisDF)
print(SortedRealIncomeGrowthVisDF)
print(SortedPresApprovalVisDF)

SortedPartyVisDF.plot.barh(y='Accuracy')
SortedWinnerVisDF.plot.barh(y='Accuracy')
SortedSentimentVisDF.plot.barh(y='Accuracy')
SortedIncumCandVisDF.plot.barh(y='Accuracy')
SortedIncumPartyVisDF.plot.barh(y='Accuracy')
SortedUnemploymentVisDF.plot.barh(y='Accuracy')
SortedGDPVisDF.plot.barh(y='Accuracy')
SortedInflationVisDF.plot.barh(y='Accuracy')
SortedSatisfactionVisDF.plot.barh(y='Accuracy')
SortedRealIncomeGrowthVisDF.plot.barh(y='Accuracy')
SortedPresApprovalVisDF.plot.barh(y='Accuracy')


#print(IndexList)
#print(TextList)
#print(PartyList)
#print(WinnerList)
#print(SentimentList)
#print(IncumCandList)
#print(IncumPartyList)
#print(UnemploymentList)
#print(GDPList)
#print(InflationList)
#print(SatisfactionList)
#print(RealIncomeGrowthList)
#print(PresApprovalList)
print(IncomeTaxList)
print(DjiaVolumeList)
print(CpiList)
print(CciIndexList)

#################################################################################################
#################################################################################################
### everything from this point down is a matter of find and replace for each of the variables ###
#################################################################################################
#################################################################################################

# the rows weren't all populated for the variables below so an edited text list is needed

### starting to build a model for IncomeTax
df_IncomeTax = rawfile.copy(deep=True)
df_IncomeTax = df_IncomeTax.filter(['id', 'income_tax', 'comb_text'])
df_IncomeTax = df_IncomeTax[df_IncomeTax['income_tax'].notna()]
print(df_IncomeTax)

IncomeTaxList = []
TextList = []
IndexList = []

for row in df_IncomeTax.itertuples():
    IncomeTaxlabel = row.income_tax
    textlabel = row.comb_text
    textlabel = textlabel.replace('\n',' ') # remove \n
    textlabel = textlabel.replace("'","'")
    textlabel = textlabel.replace('\t',' ') # remove \t
    #textlabel = textlabel.replace(',',' ') # remove ,
    #textlabel = textlabel.replace('.',' ') # remove .
    #textlabel = textlabel.replace('"',' ') # remove "
    #textlabel = textlabel.replace("'",' ') # remove '
    textlabel = textlabel.replace('nan',' ') # remove nan
    #textlabel = textlabel.replace('-',' ') # remove -
    #textlabel = textlabel.replace(' - ',' ') # remove -
    #textlabel = textlabel.replace('?',' ') # remove ?
    #textlabel = textlabel.replace('!',' ') # remove !
    #textlabel = textlabel.replace(':',' ') # remove :
    #textlabel = textlabel.replace(';',' ') # remove ;
    #textlabel = textlabel.replace('  ',' ') # remove double spaces
    indexlabel = row.id
    TextList.append(textlabel)
    IndexList.append(indexlabel)
    IncomeTaxList.append(IncomeTaxlabel)

IncomeTaxList = [ int(x) for x in IncomeTaxList ]
print(IncomeTaxList)

X_text=MyVect.fit_transform(TextList)

Bool_X_text=MyVect.fit_transform(TextList)

tf_X_text=MyVect.fit_transform(TextList)

## Hmm - that's not quite what we want...
## Let's get the feature names which ARE the words
ColumnNamesText=MyVect.get_feature_names()
print(ColumnNamesText)

## OK good - but we want a document topic model A DTM (matrix of counts)
VectDF=pd.DataFrame(X_text.toarray(), columns=ColumnNamesText, index = IndexList)
print(VectDF)

bool_VectDF=pd.DataFrame(Bool_X_text.toarray(), columns=ColumnNamesText, index = IndexList)
print(bool_VectDF)

tf_VectDF=pd.DataFrame(tf_X_text.toarray(), columns=ColumnNamesText, index = IndexList)
print(tf_VectDF)

IncomeTaxVectDF = VectDF.copy(deep=True)
IncomeTaxVectDF.insert(loc=0, column='LABEL', value=IncomeTaxList)
print(IncomeTaxVectDF)

bool_IncomeTaxVectDF = bool_VectDF.copy(deep=True)
bool_IncomeTaxVectDF.insert(loc=0, column='LABEL', value=IncomeTaxList)
print(bool_IncomeTaxVectDF)

tf_IncomeTaxVectDF = tf_VectDF.copy(deep=True)
tf_IncomeTaxVectDF.insert(loc=0, column='LABEL', value=IncomeTaxList)
print(tf_IncomeTaxVectDF)


##--------------------------------------------------------
####################################################################
###################### model Data Build ############################
####################################################################
##--------------------------------------------------------
from sklearn.model_selection import train_test_split

## Copy everything below here to rebuild for IncomeTax data
TrainDF, TestDF = train_test_split(IncomeTaxVectDF, test_size=0.3)
print(TrainDF.head())

## Now we have a training set and a testing set. 
print("\nThe training set is:")
print(TrainDF)
print("\nThe testing set is:")
print(TestDF)

## IMPORTANT - YOU CANNOT LEAVE LABELS ON THE TEST SET
## Save labels
TestLabels=TestDF["LABEL"]
print(TestLabels)
## remove labels
TestDF_nolabels = TestDF.drop(["LABEL"], axis=1)
print(TestDF_nolabels)

TrainDF_nolabels=TrainDF.drop(["LABEL"], axis=1)
print(TrainDF_nolabels)

TrainLabels=TrainDF["LABEL"]
print(TrainLabels)

############# Do the same for the binary dataframe
## Create Train/Test for TrainDFB
TrainDFB, TestDFB = train_test_split(bool_IncomeTaxVectDF, test_size=0.3)
print(TrainDFB.head())

TrainDFB_nolabels=TrainDFB.drop(["LABEL"], axis=1)
print(TrainDFB_nolabels)
TrainLabelsB=TrainDFB["LABEL"]
print(TrainLabelsB)

TestDFB_nolabels=TestDFB.drop(["LABEL"], axis=1)
print(TestDFB_nolabels)
TestLabelsB=TestDFB["LABEL"]
print(TestLabelsB)

############# Do the same for the tfidf dataframe
## Create Train/Test for TrainDF tfidf
TrainDF_tf, TestDF_tf = train_test_split(tf_IncomeTaxVectDF, test_size=0.3)
print(TrainDF_tf.head())

TrainDF_tf_nolabels=TrainDF_tf.drop(["LABEL"], axis=1)
print(TrainDF_tf_nolabels)
TrainLabels_tf=TrainDF_tf["LABEL"]
print(TrainLabels_tf)

TestDF_tf_nolabels=TestDF_tf.drop(["LABEL"], axis=1)
print(TestDF_tf_nolabels)
TestLabels_tf=TestDF_tf["LABEL"]
print(TestLabels_tf)



#############################################
###########  SVM  for regular model #1 ######
#############################################
from sklearn.svm import LinearSVC
print(TrainDF_nolabels.head())
print(TrainLabels.head())

IncomeTax_SVM_Model=LinearSVC(C=.01)
IncomeTax_SVM_Model.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
IncomeTax_svm_predict = IncomeTax_SVM_Model.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(IncomeTax_svm_predict)
print("Actual:")
print(TestLabels)

from sklearn.metrics import confusion_matrix
IncomeTax_SVM_matrix = confusion_matrix(TestLabels, IncomeTax_svm_predict)
print("\nThe confusion matrix is:")
print(IncomeTax_SVM_matrix)
print("\n\n")

from sklearn.metrics import classification_report
IncomeTax_svm_target_names = ['0','1']
print(classification_report(TestLabels, IncomeTax_svm_predict, target_names = IncomeTax_svm_target_names))

IncomeTax_SVM_reg_FP = IncomeTax_SVM_matrix[0][1] 
IncomeTax_SVM_reg_FN = IncomeTax_SVM_matrix[1][0]
IncomeTax_SVM_reg_TP = IncomeTax_SVM_matrix[1][1]
IncomeTax_SVM_reg_TN = IncomeTax_SVM_matrix[0][0]

# Overall accuracy
IncomeTax_SVM_reg_ACC = (IncomeTax_SVM_reg_TP + IncomeTax_SVM_reg_TN)/(IncomeTax_SVM_reg_TP + IncomeTax_SVM_reg_FP + IncomeTax_SVM_reg_FN + IncomeTax_SVM_reg_TN)
print(IncomeTax_SVM_reg_ACC)

IncomeTaxAccuracyDict = {}
IncomeTaxAccuracyDict.update({'IncomeTax_SVM_reg_ACC': IncomeTax_SVM_reg_ACC})
print(IncomeTaxAccuracyDict)

#############################################
###########  SVM  for regular model #2 ######
#############################################

IncomeTax_SVM_Model2=LinearSVC(C=1)
IncomeTax_SVM_Model2.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
IncomeTax_svm_predict2 = IncomeTax_SVM_Model2.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(IncomeTax_svm_predict2)
print("Actual:")
print(TestLabels)

IncomeTax_SVM_matrix2 = confusion_matrix(TestLabels, IncomeTax_svm_predict2)
print("\nThe confusion matrix is:")
print(IncomeTax_SVM_matrix2)
print("\n\n")

IncomeTax_svm_target_names2 = ['0','1']
print(classification_report(TestLabels, IncomeTax_svm_predict2, target_names = IncomeTax_svm_target_names2))

IncomeTax_SVM_reg2_FP = IncomeTax_SVM_matrix2[0][1] 
IncomeTax_SVM_reg2_FN = IncomeTax_SVM_matrix2[1][0]
IncomeTax_SVM_reg2_TP = IncomeTax_SVM_matrix2[1][1]
IncomeTax_SVM_reg2_TN = IncomeTax_SVM_matrix2[0][0]

# Overall accuracy
IncomeTax_SVM_reg2_ACC = (IncomeTax_SVM_reg2_TP + IncomeTax_SVM_reg2_TN)/(IncomeTax_SVM_reg2_TP + IncomeTax_SVM_reg2_FP + IncomeTax_SVM_reg2_FN + IncomeTax_SVM_reg2_TN)
print(IncomeTax_SVM_reg2_ACC)

IncomeTaxAccuracyDict.update({'IncomeTax_SVM_reg2_ACC': IncomeTax_SVM_reg2_ACC})
print(IncomeTaxAccuracyDict)

#############################################
###########  SVM  for regular model #3 ######
#############################################

IncomeTax_SVM_Model3=LinearSVC(C=100)
IncomeTax_SVM_Model3.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
IncomeTax_svm_predict3 = IncomeTax_SVM_Model3.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(IncomeTax_svm_predict3)
print("Actual:")
print(TestLabels)

IncomeTax_SVM_matrix3 = confusion_matrix(TestLabels, IncomeTax_svm_predict3)
print("\nThe confusion matrix is:")
print(IncomeTax_SVM_matrix3)
print("\n\n")

from sklearn.metrics import classification_report
IncomeTax_svm_target_names3 = ['0','1']
print(classification_report(TestLabels, IncomeTax_svm_predict3, target_names = IncomeTax_svm_target_names3))

IncomeTax_SVM_reg3_FP = IncomeTax_SVM_matrix3[0][1] 
IncomeTax_SVM_reg3_FN = IncomeTax_SVM_matrix3[1][0]
IncomeTax_SVM_reg3_TP = IncomeTax_SVM_matrix3[1][1]
IncomeTax_SVM_reg3_TN = IncomeTax_SVM_matrix3[0][0]

# Overall accuracy
IncomeTax_SVM_reg3_ACC = (IncomeTax_SVM_reg3_TP + IncomeTax_SVM_reg3_TN)/(IncomeTax_SVM_reg3_TP + IncomeTax_SVM_reg3_FP + IncomeTax_SVM_reg3_FN + IncomeTax_SVM_reg3_TN)
print(IncomeTax_SVM_reg3_ACC)

IncomeTaxAccuracyDict.update({'IncomeTax_SVM_reg3_ACC': IncomeTax_SVM_reg3_ACC})
print(IncomeTaxAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #1 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

IncomeTax_B_SVM_Model=LinearSVC(C=100)
IncomeTax_B_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
IncomeTax_b_svm_predict = IncomeTax_B_SVM_Model.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(IncomeTax_b_svm_predict)
print("\nActual:")
print(TestLabelsB)

IncomeTax_B_SVM_matrix = confusion_matrix(TestLabelsB, IncomeTax_b_svm_predict)
print("\nThe confusion matrix is:")
print(IncomeTax_B_SVM_matrix)
print("\n\n")

IncomeTax_svm_B_target_names = ['0','1']
print(classification_report(TestLabelsB, IncomeTax_b_svm_predict, target_names = IncomeTax_svm_B_target_names))

IncomeTax_SVM_bool_FP = IncomeTax_B_SVM_matrix[0][1] 
IncomeTax_SVM_bool_FN = IncomeTax_B_SVM_matrix[1][0]
IncomeTax_SVM_bool_TP = IncomeTax_B_SVM_matrix[1][1]
IncomeTax_SVM_bool_TN = IncomeTax_B_SVM_matrix[0][0]

# Overall accuracy
IncomeTax_SVM_bool_ACC = (IncomeTax_SVM_bool_TP + IncomeTax_SVM_bool_TN)/(IncomeTax_SVM_bool_TP + IncomeTax_SVM_bool_FP + IncomeTax_SVM_bool_FN + IncomeTax_SVM_bool_TN)
print(IncomeTax_SVM_bool_ACC)

IncomeTaxAccuracyDict.update({'IncomeTax_SVM_bool_ACC': IncomeTax_SVM_bool_ACC})
print(IncomeTaxAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #2 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

IncomeTax_B_SVM_Model2=LinearSVC(C=1)
IncomeTax_B_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
IncomeTax_b_svm_predict2 = IncomeTax_B_SVM_Model2.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(IncomeTax_b_svm_predict2)
print("\nActual:")
print(TestLabelsB)

IncomeTax_B_SVM_matrix2 = confusion_matrix(TestLabelsB, IncomeTax_b_svm_predict2)
print("\nThe confusion matrix is:")
print(IncomeTax_B_SVM_matrix2)
print("\n\n")

IncomeTax_svm_B_target_names2 = ['0','1']
print(classification_report(TestLabelsB, IncomeTax_b_svm_predict2, target_names = IncomeTax_svm_B_target_names2))

IncomeTax_SVM_bool2_FP = IncomeTax_B_SVM_matrix2[0][1] 
IncomeTax_SVM_bool2_FN = IncomeTax_B_SVM_matrix2[1][0]
IncomeTax_SVM_bool2_TP = IncomeTax_B_SVM_matrix2[1][1]
IncomeTax_SVM_bool2_TN = IncomeTax_B_SVM_matrix2[0][0]

# Overall accuracy
IncomeTax_SVM_bool2_ACC = (IncomeTax_SVM_bool2_TP + IncomeTax_SVM_bool2_TN)/(IncomeTax_SVM_bool2_TP + IncomeTax_SVM_bool2_FP + IncomeTax_SVM_bool2_FN + IncomeTax_SVM_bool2_TN)
print(IncomeTax_SVM_bool2_ACC)

IncomeTaxAccuracyDict.update({'IncomeTax_SVM_bool2_ACC': IncomeTax_SVM_bool2_ACC})
print(IncomeTaxAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #3 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

IncomeTax_B_SVM_Model3=LinearSVC(C=.01)
IncomeTax_B_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
IncomeTax_b_svm_predict3 = IncomeTax_B_SVM_Model3.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(IncomeTax_b_svm_predict3)
print("\nActual:")
print(TestLabelsB)

IncomeTax_B_SVM_matrix3 = confusion_matrix(TestLabelsB, IncomeTax_b_svm_predict3)
print("\nThe confusion matrix is:")
print(IncomeTax_B_SVM_matrix3)
print("\n\n")

IncomeTax_svm_B_target_names3 = ['0','1']
print(classification_report(TestLabelsB, IncomeTax_b_svm_predict3, target_names = IncomeTax_svm_B_target_names3))

IncomeTax_SVM_bool3_FP = IncomeTax_B_SVM_matrix3[0][1] 
IncomeTax_SVM_bool3_FN = IncomeTax_B_SVM_matrix3[1][0]
IncomeTax_SVM_bool3_TP = IncomeTax_B_SVM_matrix3[1][1]
IncomeTax_SVM_bool3_TN = IncomeTax_B_SVM_matrix3[0][0]

# Overall accuracy
IncomeTax_SVM_bool3_ACC = (IncomeTax_SVM_bool3_TP + IncomeTax_SVM_bool3_TN)/(IncomeTax_SVM_bool3_TP + IncomeTax_SVM_bool3_FP + IncomeTax_SVM_bool3_FN + IncomeTax_SVM_bool3_TN)
print(IncomeTax_SVM_bool3_ACC)

IncomeTaxAccuracyDict.update({'IncomeTax_SVM_bool3_ACC': IncomeTax_SVM_bool3_ACC})
print(IncomeTaxAccuracyDict)

#############################################
###########  SVM  for tfidf model #1 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

IncomeTax_tf_SVM_Model=LinearSVC(C=.001)
IncomeTax_tf_SVM_Model.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
IncomeTax_tf_svm_predict = IncomeTax_tf_SVM_Model.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(IncomeTax_tf_svm_predict)
print("\nActual:")
print(TestLabels_tf)

IncomeTax_tf_SVM_matrix = confusion_matrix(TestLabels_tf, IncomeTax_tf_svm_predict)
print("\nThe confusion matrix is:")
print(IncomeTax_tf_SVM_matrix)
print("\n\n")

IncomeTax_svm_tf_target_names = ['0','1']
print(classification_report(TestLabels_tf, IncomeTax_tf_svm_predict, target_names = IncomeTax_svm_tf_target_names))

IncomeTax_SVM_tf_FP = IncomeTax_tf_SVM_matrix[0][1] 
IncomeTax_SVM_tf_FN = IncomeTax_tf_SVM_matrix[1][0]
IncomeTax_SVM_tf_TP = IncomeTax_tf_SVM_matrix[1][1]
IncomeTax_SVM_tf_TN = IncomeTax_tf_SVM_matrix[0][0]

# Overall accuracy
IncomeTax_SVM_tf_ACC = (IncomeTax_SVM_tf_TP + IncomeTax_SVM_tf_TN)/(IncomeTax_SVM_tf_TP + IncomeTax_SVM_tf_FP + IncomeTax_SVM_tf_FN + IncomeTax_SVM_tf_TN)
print(IncomeTax_SVM_tf_ACC)

IncomeTaxAccuracyDict.update({'IncomeTax_SVM_tf_ACC': IncomeTax_SVM_tf_ACC})
print(IncomeTaxAccuracyDict)

#############################################
###########  SVM  for tfidf model #2 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

IncomeTax_tf_SVM_Model2=LinearSVC(C=1)
IncomeTax_tf_SVM_Model2.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
IncomeTax_tf_svm_predict2 = IncomeTax_tf_SVM_Model2.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(IncomeTax_tf_svm_predict2)
print("\nActual:")
print(TestLabels_tf)

IncomeTax_tf_SVM_matrix2 = confusion_matrix(TestLabels_tf, IncomeTax_tf_svm_predict2)
print("\nThe confusion matrix is:")
print(IncomeTax_tf_SVM_matrix2)
print("\n\n")

IncomeTax_svm_tf_target_names2 = ['0','1']
print(classification_report(TestLabels_tf, IncomeTax_tf_svm_predict2, target_names = IncomeTax_svm_tf_target_names2))

IncomeTax_SVM_tf2_FP = IncomeTax_tf_SVM_matrix2[0][1] 
IncomeTax_SVM_tf2_FN = IncomeTax_tf_SVM_matrix2[1][0]
IncomeTax_SVM_tf2_TP = IncomeTax_tf_SVM_matrix2[1][1]
IncomeTax_SVM_tf2_TN = IncomeTax_tf_SVM_matrix2[0][0]

# Overall accuracy
IncomeTax_SVM_tf2_ACC = (IncomeTax_SVM_tf2_TP + IncomeTax_SVM_tf2_TN)/(IncomeTax_SVM_tf2_TP + IncomeTax_SVM_tf2_FP + IncomeTax_SVM_tf2_FN + IncomeTax_SVM_tf2_TN)
print(IncomeTax_SVM_tf2_ACC)

IncomeTaxAccuracyDict.update({'IncomeTax_SVM_tf2_ACC': IncomeTax_SVM_tf2_ACC})
print(IncomeTaxAccuracyDict)

#############################################
###########  SVM  for tfidf model #3 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

IncomeTax_tf_SVM_Model3=LinearSVC(C=1)
IncomeTax_tf_SVM_Model3.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
IncomeTax_tf_svm_predict3 = IncomeTax_tf_SVM_Model3.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(IncomeTax_tf_svm_predict3)
print("\nActual:")
print(TestLabels_tf)

IncomeTax_tf_SVM_matrix3 = confusion_matrix(TestLabels_tf, IncomeTax_tf_svm_predict3)
print("\nThe confusion matrix is:")
print(IncomeTax_tf_SVM_matrix3)
print("\n\n")

IncomeTax_svm_tf_target_names3 = ['0','1']
print(classification_report(TestLabels_tf, IncomeTax_tf_svm_predict3, target_names = IncomeTax_svm_tf_target_names3))

IncomeTax_SVM_tf3_FP = IncomeTax_tf_SVM_matrix3[0][1] 
IncomeTax_SVM_tf3_FN = IncomeTax_tf_SVM_matrix3[1][0]
IncomeTax_SVM_tf3_TP = IncomeTax_tf_SVM_matrix3[1][1]
IncomeTax_SVM_tf3_TN = IncomeTax_tf_SVM_matrix3[0][0]

# Overall accuracy
IncomeTax_SVM_tf3_ACC = (IncomeTax_SVM_tf3_TP + IncomeTax_SVM_tf3_TN)/(IncomeTax_SVM_tf3_TP + IncomeTax_SVM_tf3_FP + IncomeTax_SVM_tf3_FN + IncomeTax_SVM_tf3_TN)
print(IncomeTax_SVM_tf3_ACC)

IncomeTaxAccuracyDict.update({'IncomeTax_SVM_tf3_ACC': IncomeTax_SVM_tf3_ACC})
print(IncomeTaxAccuracyDict)

#############################################
#######  SVM  with different kernels ########
#############################################
## running on Bernoulli model since it had best accuracy above
## i want to see if the accuracy gets improved in the different kernels

#############################################
###############  Sigmoid #1 #################
#############################################

IncomeTax_sig_SVM_Model=sklearn.svm.SVC(C=10, kernel='sigmoid', 
                           verbose=True, gamma="scale")
IncomeTax_sig_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
IncomeTax_sig_svm_predict = IncomeTax_sig_SVM_Model.predict(TestDFB_nolabels)
print("\nsig SVM prediction:\n")
print(IncomeTax_sig_svm_predict)
print("\nActual:")
print(TestLabelsB)

IncomeTax_sig_SVM_matrix = confusion_matrix(TestLabelsB, IncomeTax_sig_svm_predict)
print("\nThe sig confusion matrix is:")
print(IncomeTax_sig_SVM_matrix)
print("\n\n")

IncomeTax_svm_sig_target_names = ['0','1']
print(classification_report(TestLabelsB, IncomeTax_sig_svm_predict, target_names = IncomeTax_svm_sig_target_names))

IncomeTax_SVM_sig_FP = IncomeTax_sig_SVM_matrix[0][1] 
IncomeTax_SVM_sig_FN = IncomeTax_sig_SVM_matrix[1][0]
IncomeTax_SVM_sig_TP = IncomeTax_sig_SVM_matrix[1][1]
IncomeTax_SVM_sig_TN = IncomeTax_sig_SVM_matrix[0][0]

# Overall accuracy
IncomeTax_SVM_sig_ACC = (IncomeTax_SVM_sig_TP + IncomeTax_SVM_sig_TN)/(IncomeTax_SVM_sig_TP + IncomeTax_SVM_sig_FP + IncomeTax_SVM_sig_FN + IncomeTax_SVM_sig_TN)
print(IncomeTax_SVM_sig_ACC)

IncomeTaxAccuracyDict.update({'IncomeTax_SVM_sig_ACC': IncomeTax_SVM_sig_ACC})
print(IncomeTaxAccuracyDict)

#############################################
###############  Sigmoid #2 #################
#############################################

IncomeTax_sig_SVM_Model2=sklearn.svm.SVC(C=1000, kernel='sigmoid', 
                           verbose=True, gamma="scale")
IncomeTax_sig_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
IncomeTax_sig_svm_predict2 = IncomeTax_sig_SVM_Model2.predict(TestDFB_nolabels)
print("\nsig SVM prediction 2:\n")
print(IncomeTax_sig_svm_predict2)
print("\nActual:")
print(TestLabelsB)

IncomeTax_sig_SVM_matrix2 = confusion_matrix(TestLabelsB, IncomeTax_sig_svm_predict2)
print("\nThe sig 2 confusion matrix is:")
print(IncomeTax_sig_SVM_matrix2)
print("\n\n")

IncomeTax_svm_sig_target_names2 = ['0','1']
print(classification_report(TestLabelsB, IncomeTax_sig_svm_predict2, target_names = IncomeTax_svm_sig_target_names2))

IncomeTax_SVM_sig2_FP = IncomeTax_sig_SVM_matrix2[0][1] 
IncomeTax_SVM_sig2_FN = IncomeTax_sig_SVM_matrix2[1][0]
IncomeTax_SVM_sig2_TP = IncomeTax_sig_SVM_matrix2[1][1]
IncomeTax_SVM_sig2_TN = IncomeTax_sig_SVM_matrix2[0][0]

# Overall accuracy
IncomeTax_SVM_sig2_ACC = (IncomeTax_SVM_sig2_TP + IncomeTax_SVM_sig2_TN)/(IncomeTax_SVM_sig2_TP + IncomeTax_SVM_sig2_FP + IncomeTax_SVM_sig2_FN + IncomeTax_SVM_sig2_TN)
print(IncomeTax_SVM_sig2_ACC)

IncomeTaxAccuracyDict.update({'IncomeTax_SVM_sig2_ACC': IncomeTax_SVM_sig2_ACC})
print(IncomeTaxAccuracyDict)

#############################################
###############  Sigmoid #3 #################
#############################################

IncomeTax_sig_SVM_Model3=sklearn.svm.SVC(C=.001, kernel='sigmoid', 
                           verbose=True, gamma="scale")
IncomeTax_sig_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
IncomeTax_sig_svm_predict3 = IncomeTax_sig_SVM_Model3.predict(TestDFB_nolabels)
print("\nsig SVM prediction 3:\n")
print(IncomeTax_sig_svm_predict3)
print("\nActual:")
print(TestLabelsB)

IncomeTax_sig_SVM_matrix3 = confusion_matrix(TestLabelsB, IncomeTax_sig_svm_predict3)
print("\nThe sig 3 confusion matrix is:")
print(IncomeTax_sig_SVM_matrix3)
print("\n\n")

IncomeTax_svm_sig_target_names3 = ['0','1']
print(classification_report(TestLabelsB, IncomeTax_sig_svm_predict3, target_names = IncomeTax_svm_sig_target_names3))

IncomeTax_SVM_sig3_FP = IncomeTax_sig_SVM_matrix3[0][1] 
IncomeTax_SVM_sig3_FN = IncomeTax_sig_SVM_matrix3[1][0]
IncomeTax_SVM_sig3_TP = IncomeTax_sig_SVM_matrix3[1][1]
IncomeTax_SVM_sig3_TN = IncomeTax_sig_SVM_matrix3[0][0]

# Overall accuracy
IncomeTax_SVM_sig3_ACC = (IncomeTax_SVM_sig3_TP + IncomeTax_SVM_sig3_TN)/(IncomeTax_SVM_sig3_TP + IncomeTax_SVM_sig3_FP + IncomeTax_SVM_sig3_FN + IncomeTax_SVM_sig3_TN)
print(IncomeTax_SVM_sig3_ACC)

IncomeTaxAccuracyDict.update({'IncomeTax_SVM_sig3_ACC': IncomeTax_SVM_sig3_ACC})
print(IncomeTaxAccuracyDict)

#############################################
################  Poly #1 ###################
#############################################

IncomeTax_poly_SVM_Model=sklearn.svm.SVC(C=1000, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(IncomeTax_poly_SVM_Model)
IncomeTax_poly_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
IncomeTax_poly_svm_predict = IncomeTax_poly_SVM_Model.predict(TestDFB_nolabels)
print("\npoly SVM prediction:\n")
print(IncomeTax_poly_svm_predict)
print("\nActual:")
print(TestLabelsB)

IncomeTax_poly_SVM_matrix = confusion_matrix(TestLabelsB, IncomeTax_poly_svm_predict)
print("\nThe poly confusion matrix is:")
print(IncomeTax_poly_SVM_matrix)
print("\n\n")

IncomeTax_svm_poly_target_names = ['0','1']
print(classification_report(TestLabelsB, IncomeTax_poly_svm_predict, target_names = IncomeTax_svm_poly_target_names))

IncomeTax_SVM_poly_FP = IncomeTax_poly_SVM_matrix[0][1] 
IncomeTax_SVM_poly_FN = IncomeTax_poly_SVM_matrix[1][0]
IncomeTax_SVM_poly_TP = IncomeTax_poly_SVM_matrix[1][1]
IncomeTax_SVM_poly_TN = IncomeTax_poly_SVM_matrix[0][0]

# Overall accuracy
IncomeTax_SVM_poly_ACC = (IncomeTax_SVM_poly_TP + IncomeTax_SVM_poly_TN)/(IncomeTax_SVM_poly_TP + IncomeTax_SVM_poly_FP + IncomeTax_SVM_poly_FN + IncomeTax_SVM_poly_TN)
print(IncomeTax_SVM_poly_ACC)

IncomeTaxAccuracyDict.update({'IncomeTax_SVM_poly_ACC': IncomeTax_SVM_poly_ACC})
print(IncomeTaxAccuracyDict)

#############################################
################  Poly #2 ###################
#############################################

IncomeTax_poly_SVM_Model2=sklearn.svm.SVC(C=10, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(IncomeTax_poly_SVM_Model2)
IncomeTax_poly_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
IncomeTax_poly_svm_predict2 = IncomeTax_poly_SVM_Model2.predict(TestDFB_nolabels)
print("\npoly SVM 2 prediction:\n")
print(IncomeTax_poly_svm_predict2)
print("\nActual:")
print(TestLabelsB)

IncomeTax_poly_SVM_matrix2 = confusion_matrix(TestLabelsB, IncomeTax_poly_svm_predict2)
print("\nThe poly 2 confusion matrix is:")
print(IncomeTax_poly_SVM_matrix2)
print("\n\n")

IncomeTax_svm_poly_target_names2 = ['0','1']
print(classification_report(TestLabelsB, IncomeTax_poly_svm_predict2, target_names = IncomeTax_svm_poly_target_names2))

IncomeTax_SVM_poly2_FP = IncomeTax_poly_SVM_matrix2[0][1] 
IncomeTax_SVM_poly2_FN = IncomeTax_poly_SVM_matrix2[1][0]
IncomeTax_SVM_poly2_TP = IncomeTax_poly_SVM_matrix2[1][1]
IncomeTax_SVM_poly2_TN = IncomeTax_poly_SVM_matrix2[0][0]

# Overall accuracy
IncomeTax_SVM_poly2_ACC = (IncomeTax_SVM_poly2_TP + IncomeTax_SVM_poly2_TN)/(IncomeTax_SVM_poly2_TP + IncomeTax_SVM_poly2_FP + IncomeTax_SVM_poly2_FN + IncomeTax_SVM_poly2_TN)
print(IncomeTax_SVM_poly2_ACC)

IncomeTaxAccuracyDict.update({'IncomeTax_SVM_poly2_ACC': IncomeTax_SVM_poly2_ACC})
print(IncomeTaxAccuracyDict)

#############################################
################  Poly #3 ###################
#############################################

IncomeTax_poly_SVM_Model3=sklearn.svm.SVC(C=.001, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(IncomeTax_poly_SVM_Model3)
IncomeTax_poly_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
IncomeTax_poly_svm_predict3 = IncomeTax_poly_SVM_Model3.predict(TestDFB_nolabels)
print("\npoly SVM 3 prediction:\n")
print(IncomeTax_poly_svm_predict3)
print("\nActual:")
print(TestLabelsB)

IncomeTax_poly_SVM_matrix3 = confusion_matrix(TestLabelsB, IncomeTax_poly_svm_predict3)
print("\nThe poly 3 confusion matrix is:")
print(IncomeTax_poly_SVM_matrix3)
print("\n\n")

IncomeTax_svm_poly_target_names3 = ['0','1']
print(classification_report(TestLabelsB, IncomeTax_poly_svm_predict3, target_names = IncomeTax_svm_poly_target_names3))

IncomeTax_SVM_poly3_FP = IncomeTax_poly_SVM_matrix3[0][1] 
IncomeTax_SVM_poly3_FN = IncomeTax_poly_SVM_matrix3[1][0]
IncomeTax_SVM_poly3_TP = IncomeTax_poly_SVM_matrix3[1][1]
IncomeTax_SVM_poly3_TN = IncomeTax_poly_SVM_matrix3[0][0]

# Overall accuracy
IncomeTax_SVM_poly3_ACC = (IncomeTax_SVM_poly3_TP + IncomeTax_SVM_poly3_TN)/(IncomeTax_SVM_poly3_TP + IncomeTax_SVM_poly3_FP + IncomeTax_SVM_poly3_FN + IncomeTax_SVM_poly3_TN)
print(IncomeTax_SVM_poly3_ACC)

IncomeTaxAccuracyDict.update({'IncomeTax_SVM_poly3_ACC': IncomeTax_SVM_poly3_ACC})
print(IncomeTaxAccuracyDict)

IncomeTaxVisDF = pd.DataFrame(IncomeTaxAccuracyDict.items(), index = IncomeTaxAccuracyDict.keys(), columns=['Model','Accuracy'])
print(IncomeTaxVisDF)
SortedIncomeTaxVisDF = IncomeTaxVisDF.sort_values('Accuracy', ascending = [True])
print(SortedIncomeTaxVisDF)

print(PartyAccuracyDict)
print(WinnerAccuracyDict)
print(SentimentAccuracyDict)
print(IncumCandAccuracyDict)
print(IncumPartyAccuracyDict)
print(UnemploymentAccuracyDict)
print(GDPAccuracyDict)
print(InflationAccuracyDict)
print(SatisfactionAccuracyDict)
print(RealIncomeGrowthAccuracyDict)
print(PresApprovalAccuracyDict)
print(IncomeTaxAccuracyDict)

print(SortedPartyVisDF)
print(SortedWinnerVisDF)
print(SortedSentimentVisDF)
print(SortedIncumCandVisDF)
print(SortedIncumPartyVisDF)
print(SortedUnemploymentVisDF)
print(SortedGDPVisDF)
print(SortedInflationVisDF)
print(SortedSatisfactionVisDF)
print(SortedRealIncomeGrowthVisDF)
print(SortedPresApprovalVisDF)
print(SortedIncomeTaxVisDF)

SortedPartyVisDF.plot.barh(y='Accuracy')
SortedWinnerVisDF.plot.barh(y='Accuracy')
SortedSentimentVisDF.plot.barh(y='Accuracy')
SortedIncumCandVisDF.plot.barh(y='Accuracy')
SortedIncumPartyVisDF.plot.barh(y='Accuracy')
SortedUnemploymentVisDF.plot.barh(y='Accuracy')
SortedGDPVisDF.plot.barh(y='Accuracy')
SortedInflationVisDF.plot.barh(y='Accuracy')
SortedSatisfactionVisDF.plot.barh(y='Accuracy')
SortedRealIncomeGrowthVisDF.plot.barh(y='Accuracy')
SortedPresApprovalVisDF.plot.barh(y='Accuracy')
SortedIncomeTaxVisDF.plot.barh(y='Accuracy')


#print(IndexList)
#print(TextList)
#print(PartyList)
#print(WinnerList)
#print(SentimentList)
#print(IncumCandList)
#print(IncumPartyList)
#print(UnemploymentList)
#print(GDPList)
#print(InflationList)
#print(SatisfactionList)
#print(RealIncomeGrowthList)
#print(PresApprovalList)
#print(IncomeTaxList)
print(DjiaVolumeList)
print(CpiList)
print(CciIndexList)

#################################################################################################
#################################################################################################
### everything from this point down is a matter of find and replace for each of the variables ###
#################################################################################################
#################################################################################################

# the rows weren't all populated for the variables below so an edited text list is needed

### starting to build a model for DjiaVolume
df_DjiaVolume = rawfile.copy(deep=True)
df_DjiaVolume = df_DjiaVolume.filter(['id', 'djia_volume', 'comb_text'])
df_DjiaVolume = df_DjiaVolume[df_DjiaVolume['djia_volume'].notna()]
print(df_DjiaVolume)

DjiaVolumeList = []
TextList = []
IndexList = []

for row in df_DjiaVolume.itertuples():
    DjiaVolumelabel = row.djia_volume
    textlabel = row.comb_text
    textlabel = textlabel.replace('\n',' ') # remove \n
    textlabel = textlabel.replace("'","'")
    textlabel = textlabel.replace('\t',' ') # remove \t
    #textlabel = textlabel.replace(',',' ') # remove ,
    #textlabel = textlabel.replace('.',' ') # remove .
    #textlabel = textlabel.replace('"',' ') # remove "
    #textlabel = textlabel.replace("'",' ') # remove '
    textlabel = textlabel.replace('nan',' ') # remove nan
    #textlabel = textlabel.replace('-',' ') # remove -
    #textlabel = textlabel.replace(' - ',' ') # remove -
    #textlabel = textlabel.replace('?',' ') # remove ?
    #textlabel = textlabel.replace('!',' ') # remove !
    #textlabel = textlabel.replace(':',' ') # remove :
    #textlabel = textlabel.replace(';',' ') # remove ;
    #textlabel = textlabel.replace('  ',' ') # remove double spaces
    indexlabel = row.id
    TextList.append(textlabel)
    IndexList.append(indexlabel)
    DjiaVolumeList.append(DjiaVolumelabel)

DjiaVolumeList = [ int(x) for x in DjiaVolumeList ]
print(DjiaVolumeList)

X_text=MyVect.fit_transform(TextList)

Bool_X_text=MyVect.fit_transform(TextList)

tf_X_text=MyVect.fit_transform(TextList)

## Hmm - that's not quite what we want...
## Let's get the feature names which ARE the words
ColumnNamesText=MyVect.get_feature_names()
print(ColumnNamesText)

## OK good - but we want a document topic model A DTM (matrix of counts)
VectDF=pd.DataFrame(X_text.toarray(), columns=ColumnNamesText, index = IndexList)
print(VectDF)

bool_VectDF=pd.DataFrame(Bool_X_text.toarray(), columns=ColumnNamesText, index = IndexList)
print(bool_VectDF)

tf_VectDF=pd.DataFrame(tf_X_text.toarray(), columns=ColumnNamesText, index = IndexList)
print(tf_VectDF)

DjiaVolumeVectDF = VectDF.copy(deep=True)
DjiaVolumeVectDF.insert(loc=0, column='LABEL', value=DjiaVolumeList)
print(DjiaVolumeVectDF)

bool_DjiaVolumeVectDF = bool_VectDF.copy(deep=True)
bool_DjiaVolumeVectDF.insert(loc=0, column='LABEL', value=DjiaVolumeList)
print(bool_DjiaVolumeVectDF)

tf_DjiaVolumeVectDF = tf_VectDF.copy(deep=True)
tf_DjiaVolumeVectDF.insert(loc=0, column='LABEL', value=DjiaVolumeList)
print(tf_DjiaVolumeVectDF)


##--------------------------------------------------------
####################################################################
###################### model Data Build ############################
####################################################################
##--------------------------------------------------------
from sklearn.model_selection import train_test_split

## Copy everything below here to rebuild for DjiaVolume data
TrainDF, TestDF = train_test_split(DjiaVolumeVectDF, test_size=0.3)
print(TrainDF.head())

## Now we have a training set and a testing set. 
print("\nThe training set is:")
print(TrainDF)
print("\nThe testing set is:")
print(TestDF)

## IMPORTANT - YOU CANNOT LEAVE LABELS ON THE TEST SET
## Save labels
TestLabels=TestDF["LABEL"]
print(TestLabels)
## remove labels
TestDF_nolabels = TestDF.drop(["LABEL"], axis=1)
print(TestDF_nolabels)

TrainDF_nolabels=TrainDF.drop(["LABEL"], axis=1)
print(TrainDF_nolabels)

TrainLabels=TrainDF["LABEL"]
print(TrainLabels)

############# Do the same for the binary dataframe
## Create Train/Test for TrainDFB
TrainDFB, TestDFB = train_test_split(bool_DjiaVolumeVectDF, test_size=0.3)
print(TrainDFB.head())

TrainDFB_nolabels=TrainDFB.drop(["LABEL"], axis=1)
print(TrainDFB_nolabels)
TrainLabelsB=TrainDFB["LABEL"]
print(TrainLabelsB)

TestDFB_nolabels=TestDFB.drop(["LABEL"], axis=1)
print(TestDFB_nolabels)
TestLabelsB=TestDFB["LABEL"]
print(TestLabelsB)

############# Do the same for the tfidf dataframe
## Create Train/Test for TrainDF tfidf
TrainDF_tf, TestDF_tf = train_test_split(tf_DjiaVolumeVectDF, test_size=0.3)
print(TrainDF_tf.head())

TrainDF_tf_nolabels=TrainDF_tf.drop(["LABEL"], axis=1)
print(TrainDF_tf_nolabels)
TrainLabels_tf=TrainDF_tf["LABEL"]
print(TrainLabels_tf)

TestDF_tf_nolabels=TestDF_tf.drop(["LABEL"], axis=1)
print(TestDF_tf_nolabels)
TestLabels_tf=TestDF_tf["LABEL"]
print(TestLabels_tf)



#############################################
###########  SVM  for regular model #1 ######
#############################################
from sklearn.svm import LinearSVC
print(TrainDF_nolabels.head())
print(TrainLabels.head())

DjiaVolume_SVM_Model=LinearSVC(C=.01)
DjiaVolume_SVM_Model.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
DjiaVolume_svm_predict = DjiaVolume_SVM_Model.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(DjiaVolume_svm_predict)
print("Actual:")
print(TestLabels)

from sklearn.metrics import confusion_matrix
DjiaVolume_SVM_matrix = confusion_matrix(TestLabels, DjiaVolume_svm_predict)
print("\nThe confusion matrix is:")
print(DjiaVolume_SVM_matrix)
print("\n\n")

from sklearn.metrics import classification_report
DjiaVolume_svm_target_names = ['0','1']
print(classification_report(TestLabels, DjiaVolume_svm_predict, target_names = DjiaVolume_svm_target_names))

DjiaVolume_SVM_reg_FP = DjiaVolume_SVM_matrix[0][1] 
DjiaVolume_SVM_reg_FN = DjiaVolume_SVM_matrix[1][0]
DjiaVolume_SVM_reg_TP = DjiaVolume_SVM_matrix[1][1]
DjiaVolume_SVM_reg_TN = DjiaVolume_SVM_matrix[0][0]

# Overall accuracy
DjiaVolume_SVM_reg_ACC = (DjiaVolume_SVM_reg_TP + DjiaVolume_SVM_reg_TN)/(DjiaVolume_SVM_reg_TP + DjiaVolume_SVM_reg_FP + DjiaVolume_SVM_reg_FN + DjiaVolume_SVM_reg_TN)
print(DjiaVolume_SVM_reg_ACC)

DjiaVolumeAccuracyDict = {}
DjiaVolumeAccuracyDict.update({'DjiaVolume_SVM_reg_ACC': DjiaVolume_SVM_reg_ACC})
print(DjiaVolumeAccuracyDict)

#############################################
###########  SVM  for regular model #2 ######
#############################################

DjiaVolume_SVM_Model2=LinearSVC(C=1)
DjiaVolume_SVM_Model2.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
DjiaVolume_svm_predict2 = DjiaVolume_SVM_Model2.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(DjiaVolume_svm_predict2)
print("Actual:")
print(TestLabels)

DjiaVolume_SVM_matrix2 = confusion_matrix(TestLabels, DjiaVolume_svm_predict2)
print("\nThe confusion matrix is:")
print(DjiaVolume_SVM_matrix2)
print("\n\n")

DjiaVolume_svm_target_names2 = ['0','1']
print(classification_report(TestLabels, DjiaVolume_svm_predict2, target_names = DjiaVolume_svm_target_names2))

DjiaVolume_SVM_reg2_FP = DjiaVolume_SVM_matrix2[0][1] 
DjiaVolume_SVM_reg2_FN = DjiaVolume_SVM_matrix2[1][0]
DjiaVolume_SVM_reg2_TP = DjiaVolume_SVM_matrix2[1][1]
DjiaVolume_SVM_reg2_TN = DjiaVolume_SVM_matrix2[0][0]

# Overall accuracy
DjiaVolume_SVM_reg2_ACC = (DjiaVolume_SVM_reg2_TP + DjiaVolume_SVM_reg2_TN)/(DjiaVolume_SVM_reg2_TP + DjiaVolume_SVM_reg2_FP + DjiaVolume_SVM_reg2_FN + DjiaVolume_SVM_reg2_TN)
print(DjiaVolume_SVM_reg2_ACC)

DjiaVolumeAccuracyDict.update({'DjiaVolume_SVM_reg2_ACC': DjiaVolume_SVM_reg2_ACC})
print(DjiaVolumeAccuracyDict)

#############################################
###########  SVM  for regular model #3 ######
#############################################

DjiaVolume_SVM_Model3=LinearSVC(C=100)
DjiaVolume_SVM_Model3.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
DjiaVolume_svm_predict3 = DjiaVolume_SVM_Model3.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(DjiaVolume_svm_predict3)
print("Actual:")
print(TestLabels)

DjiaVolume_SVM_matrix3 = confusion_matrix(TestLabels, DjiaVolume_svm_predict3)
print("\nThe confusion matrix is:")
print(DjiaVolume_SVM_matrix3)
print("\n\n")

from sklearn.metrics import classification_report
DjiaVolume_svm_target_names3 = ['0','1']
print(classification_report(TestLabels, DjiaVolume_svm_predict3, target_names = DjiaVolume_svm_target_names3))

DjiaVolume_SVM_reg3_FP = DjiaVolume_SVM_matrix3[0][1] 
DjiaVolume_SVM_reg3_FN = DjiaVolume_SVM_matrix3[1][0]
DjiaVolume_SVM_reg3_TP = DjiaVolume_SVM_matrix3[1][1]
DjiaVolume_SVM_reg3_TN = DjiaVolume_SVM_matrix3[0][0]

# Overall accuracy
DjiaVolume_SVM_reg3_ACC = (DjiaVolume_SVM_reg3_TP + DjiaVolume_SVM_reg3_TN)/(DjiaVolume_SVM_reg3_TP + DjiaVolume_SVM_reg3_FP + DjiaVolume_SVM_reg3_FN + DjiaVolume_SVM_reg3_TN)
print(DjiaVolume_SVM_reg3_ACC)

DjiaVolumeAccuracyDict.update({'DjiaVolume_SVM_reg3_ACC': DjiaVolume_SVM_reg3_ACC})
print(DjiaVolumeAccuracyDict)

#############################################
###########  SVM  for Boolean model #1 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

DjiaVolume_B_SVM_Model=LinearSVC(C=100)
DjiaVolume_B_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
DjiaVolume_b_svm_predict = DjiaVolume_B_SVM_Model.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(DjiaVolume_b_svm_predict)
print("\nActual:")
print(TestLabelsB)

DjiaVolume_B_SVM_matrix = confusion_matrix(TestLabelsB, DjiaVolume_b_svm_predict)
print("\nThe confusion matrix is:")
print(DjiaVolume_B_SVM_matrix)
print("\n\n")

DjiaVolume_svm_B_target_names = ['0','1']
print(classification_report(TestLabelsB, DjiaVolume_b_svm_predict, target_names = DjiaVolume_svm_B_target_names))

DjiaVolume_SVM_bool_FP = DjiaVolume_B_SVM_matrix[0][1] 
DjiaVolume_SVM_bool_FN = DjiaVolume_B_SVM_matrix[1][0]
DjiaVolume_SVM_bool_TP = DjiaVolume_B_SVM_matrix[1][1]
DjiaVolume_SVM_bool_TN = DjiaVolume_B_SVM_matrix[0][0]

# Overall accuracy
DjiaVolume_SVM_bool_ACC = (DjiaVolume_SVM_bool_TP + DjiaVolume_SVM_bool_TN)/(DjiaVolume_SVM_bool_TP + DjiaVolume_SVM_bool_FP + DjiaVolume_SVM_bool_FN + DjiaVolume_SVM_bool_TN)
print(DjiaVolume_SVM_bool_ACC)

DjiaVolumeAccuracyDict.update({'DjiaVolume_SVM_bool_ACC': DjiaVolume_SVM_bool_ACC})
print(DjiaVolumeAccuracyDict)

#############################################
###########  SVM  for Boolean model #2 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

DjiaVolume_B_SVM_Model2=LinearSVC(C=1)
DjiaVolume_B_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
DjiaVolume_b_svm_predict2 = DjiaVolume_B_SVM_Model2.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(DjiaVolume_b_svm_predict2)
print("\nActual:")
print(TestLabelsB)

DjiaVolume_B_SVM_matrix2 = confusion_matrix(TestLabelsB, DjiaVolume_b_svm_predict2)
print("\nThe confusion matrix is:")
print(DjiaVolume_B_SVM_matrix2)
print("\n\n")

DjiaVolume_svm_B_target_names2 = ['0','1']
print(classification_report(TestLabelsB, DjiaVolume_b_svm_predict2, target_names = DjiaVolume_svm_B_target_names2))

DjiaVolume_SVM_bool2_FP = DjiaVolume_B_SVM_matrix2[0][1] 
DjiaVolume_SVM_bool2_FN = DjiaVolume_B_SVM_matrix2[1][0]
DjiaVolume_SVM_bool2_TP = DjiaVolume_B_SVM_matrix2[1][1]
DjiaVolume_SVM_bool2_TN = DjiaVolume_B_SVM_matrix2[0][0]

# Overall accuracy
DjiaVolume_SVM_bool2_ACC = (DjiaVolume_SVM_bool2_TP + DjiaVolume_SVM_bool2_TN)/(DjiaVolume_SVM_bool2_TP + DjiaVolume_SVM_bool2_FP + DjiaVolume_SVM_bool2_FN + DjiaVolume_SVM_bool2_TN)
print(DjiaVolume_SVM_bool2_ACC)

DjiaVolumeAccuracyDict.update({'DjiaVolume_SVM_bool2_ACC': DjiaVolume_SVM_bool2_ACC})
print(DjiaVolumeAccuracyDict)

#############################################
###########  SVM  for Boolean model #3 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

DjiaVolume_B_SVM_Model3=LinearSVC(C=.01)
DjiaVolume_B_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
DjiaVolume_b_svm_predict3 = DjiaVolume_B_SVM_Model3.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(DjiaVolume_b_svm_predict3)
print("\nActual:")
print(TestLabelsB)

DjiaVolume_B_SVM_matrix3 = confusion_matrix(TestLabelsB, DjiaVolume_b_svm_predict3)
print("\nThe confusion matrix is:")
print(DjiaVolume_B_SVM_matrix3)
print("\n\n")

DjiaVolume_svm_B_target_names3 = ['0','1']
print(classification_report(TestLabelsB, DjiaVolume_b_svm_predict3, target_names = DjiaVolume_svm_B_target_names3))

DjiaVolume_SVM_bool3_FP = DjiaVolume_B_SVM_matrix3[0][1] 
DjiaVolume_SVM_bool3_FN = DjiaVolume_B_SVM_matrix3[1][0]
DjiaVolume_SVM_bool3_TP = DjiaVolume_B_SVM_matrix3[1][1]
DjiaVolume_SVM_bool3_TN = DjiaVolume_B_SVM_matrix3[0][0]

# Overall accuracy
DjiaVolume_SVM_bool3_ACC = (DjiaVolume_SVM_bool3_TP + DjiaVolume_SVM_bool3_TN)/(DjiaVolume_SVM_bool3_TP + DjiaVolume_SVM_bool3_FP + DjiaVolume_SVM_bool3_FN + DjiaVolume_SVM_bool3_TN)
print(DjiaVolume_SVM_bool3_ACC)

DjiaVolumeAccuracyDict.update({'DjiaVolume_SVM_bool3_ACC': DjiaVolume_SVM_bool3_ACC})
print(DjiaVolumeAccuracyDict)

#############################################
###########  SVM  for tfidf model #1 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

DjiaVolume_tf_SVM_Model=LinearSVC(C=.001)
DjiaVolume_tf_SVM_Model.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
DjiaVolume_tf_svm_predict = DjiaVolume_tf_SVM_Model.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(DjiaVolume_tf_svm_predict)
print("\nActual:")
print(TestLabels_tf)

DjiaVolume_tf_SVM_matrix = confusion_matrix(TestLabels_tf, DjiaVolume_tf_svm_predict)
print("\nThe confusion matrix is:")
print(DjiaVolume_tf_SVM_matrix)
print("\n\n")

DjiaVolume_svm_tf_target_names = ['0','1']
print(classification_report(TestLabels_tf, DjiaVolume_tf_svm_predict, target_names = DjiaVolume_svm_tf_target_names))

DjiaVolume_SVM_tf_FP = DjiaVolume_tf_SVM_matrix[0][1] 
DjiaVolume_SVM_tf_FN = DjiaVolume_tf_SVM_matrix[1][0]
DjiaVolume_SVM_tf_TP = DjiaVolume_tf_SVM_matrix[1][1]
DjiaVolume_SVM_tf_TN = DjiaVolume_tf_SVM_matrix[0][0]

# Overall accuracy
DjiaVolume_SVM_tf_ACC = (DjiaVolume_SVM_tf_TP + DjiaVolume_SVM_tf_TN)/(DjiaVolume_SVM_tf_TP + DjiaVolume_SVM_tf_FP + DjiaVolume_SVM_tf_FN + DjiaVolume_SVM_tf_TN)
print(DjiaVolume_SVM_tf_ACC)

DjiaVolumeAccuracyDict.update({'DjiaVolume_SVM_tf_ACC': DjiaVolume_SVM_tf_ACC})
print(DjiaVolumeAccuracyDict)

#############################################
###########  SVM  for tfidf model #2 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

DjiaVolume_tf_SVM_Model2=LinearSVC(C=1)
DjiaVolume_tf_SVM_Model2.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
DjiaVolume_tf_svm_predict2 = DjiaVolume_tf_SVM_Model2.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(DjiaVolume_tf_svm_predict2)
print("\nActual:")
print(TestLabels_tf)

DjiaVolume_tf_SVM_matrix2 = confusion_matrix(TestLabels_tf, DjiaVolume_tf_svm_predict2)
print("\nThe confusion matrix is:")
print(DjiaVolume_tf_SVM_matrix2)
print("\n\n")

DjiaVolume_svm_tf_target_names2 = ['0','1']
print(classification_report(TestLabels_tf, DjiaVolume_tf_svm_predict2, target_names = DjiaVolume_svm_tf_target_names2))

DjiaVolume_SVM_tf2_FP = DjiaVolume_tf_SVM_matrix2[0][1] 
DjiaVolume_SVM_tf2_FN = DjiaVolume_tf_SVM_matrix2[1][0]
DjiaVolume_SVM_tf2_TP = DjiaVolume_tf_SVM_matrix2[1][1]
DjiaVolume_SVM_tf2_TN = DjiaVolume_tf_SVM_matrix2[0][0]

# Overall accuracy
DjiaVolume_SVM_tf2_ACC = (DjiaVolume_SVM_tf2_TP + DjiaVolume_SVM_tf2_TN)/(DjiaVolume_SVM_tf2_TP + DjiaVolume_SVM_tf2_FP + DjiaVolume_SVM_tf2_FN + DjiaVolume_SVM_tf2_TN)
print(DjiaVolume_SVM_tf2_ACC)

DjiaVolumeAccuracyDict.update({'DjiaVolume_SVM_tf2_ACC': DjiaVolume_SVM_tf2_ACC})
print(DjiaVolumeAccuracyDict)

#############################################
###########  SVM  for tfidf model #3 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

DjiaVolume_tf_SVM_Model3=LinearSVC(C=1)
DjiaVolume_tf_SVM_Model3.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
DjiaVolume_tf_svm_predict3 = DjiaVolume_tf_SVM_Model3.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(DjiaVolume_tf_svm_predict3)
print("\nActual:")
print(TestLabels_tf)

DjiaVolume_tf_SVM_matrix3 = confusion_matrix(TestLabels_tf, DjiaVolume_tf_svm_predict3)
print("\nThe confusion matrix is:")
print(DjiaVolume_tf_SVM_matrix3)
print("\n\n")

DjiaVolume_svm_tf_target_names3 = ['0','1']
print(classification_report(TestLabels_tf, DjiaVolume_tf_svm_predict3, target_names = DjiaVolume_svm_tf_target_names3))

DjiaVolume_SVM_tf3_FP = DjiaVolume_tf_SVM_matrix3[0][1] 
DjiaVolume_SVM_tf3_FN = DjiaVolume_tf_SVM_matrix3[1][0]
DjiaVolume_SVM_tf3_TP = DjiaVolume_tf_SVM_matrix3[1][1]
DjiaVolume_SVM_tf3_TN = DjiaVolume_tf_SVM_matrix3[0][0]

# Overall accuracy
DjiaVolume_SVM_tf3_ACC = (DjiaVolume_SVM_tf3_TP + DjiaVolume_SVM_tf3_TN)/(DjiaVolume_SVM_tf3_TP + DjiaVolume_SVM_tf3_FP + DjiaVolume_SVM_tf3_FN + DjiaVolume_SVM_tf3_TN)
print(DjiaVolume_SVM_tf3_ACC)

DjiaVolumeAccuracyDict.update({'DjiaVolume_SVM_tf3_ACC': DjiaVolume_SVM_tf3_ACC})
print(DjiaVolumeAccuracyDict)

#############################################
#######  SVM  with different kernels ########
#############################################
## running on Bernoulli model since it had best accuracy above
## i want to see if the accuracy gets improved in the different kernels

#############################################
###############  Sigmoid #1 #################
#############################################

DjiaVolume_sig_SVM_Model=sklearn.svm.SVC(C=10, kernel='sigmoid', 
                           verbose=True, gamma="scale")
DjiaVolume_sig_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
DjiaVolume_sig_svm_predict = DjiaVolume_sig_SVM_Model.predict(TestDFB_nolabels)
print("\nsig SVM prediction:\n")
print(DjiaVolume_sig_svm_predict)
print("\nActual:")
print(TestLabelsB)

DjiaVolume_sig_SVM_matrix = confusion_matrix(TestLabelsB, DjiaVolume_sig_svm_predict)
print("\nThe sig confusion matrix is:")
print(DjiaVolume_sig_SVM_matrix)
print("\n\n")

DjiaVolume_svm_sig_target_names = ['0','1']
print(classification_report(TestLabelsB, DjiaVolume_sig_svm_predict, target_names = DjiaVolume_svm_sig_target_names))

DjiaVolume_SVM_sig_FP = DjiaVolume_sig_SVM_matrix[0][1] 
DjiaVolume_SVM_sig_FN = DjiaVolume_sig_SVM_matrix[1][0]
DjiaVolume_SVM_sig_TP = DjiaVolume_sig_SVM_matrix[1][1]
DjiaVolume_SVM_sig_TN = DjiaVolume_sig_SVM_matrix[0][0]

# Overall accuracy
DjiaVolume_SVM_sig_ACC = (DjiaVolume_SVM_sig_TP + DjiaVolume_SVM_sig_TN)/(DjiaVolume_SVM_sig_TP + DjiaVolume_SVM_sig_FP + DjiaVolume_SVM_sig_FN + DjiaVolume_SVM_sig_TN)
print(DjiaVolume_SVM_sig_ACC)

DjiaVolumeAccuracyDict.update({'DjiaVolume_SVM_sig_ACC': DjiaVolume_SVM_sig_ACC})
print(DjiaVolumeAccuracyDict)

#############################################
###############  Sigmoid #2 #################
#############################################

DjiaVolume_sig_SVM_Model2=sklearn.svm.SVC(C=1000, kernel='sigmoid', 
                           verbose=True, gamma="scale")
DjiaVolume_sig_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
DjiaVolume_sig_svm_predict2 = DjiaVolume_sig_SVM_Model2.predict(TestDFB_nolabels)
print("\nsig SVM prediction 2:\n")
print(DjiaVolume_sig_svm_predict2)
print("\nActual:")
print(TestLabelsB)

DjiaVolume_sig_SVM_matrix2 = confusion_matrix(TestLabelsB, DjiaVolume_sig_svm_predict2)
print("\nThe sig 2 confusion matrix is:")
print(DjiaVolume_sig_SVM_matrix2)
print("\n\n")

DjiaVolume_svm_sig_target_names2 = ['0','1']
print(classification_report(TestLabelsB, DjiaVolume_sig_svm_predict2, target_names = DjiaVolume_svm_sig_target_names2))

DjiaVolume_SVM_sig2_FP = DjiaVolume_sig_SVM_matrix2[0][1] 
DjiaVolume_SVM_sig2_FN = DjiaVolume_sig_SVM_matrix2[1][0]
DjiaVolume_SVM_sig2_TP = DjiaVolume_sig_SVM_matrix2[1][1]
DjiaVolume_SVM_sig2_TN = DjiaVolume_sig_SVM_matrix2[0][0]

# Overall accuracy
DjiaVolume_SVM_sig2_ACC = (DjiaVolume_SVM_sig2_TP + DjiaVolume_SVM_sig2_TN)/(DjiaVolume_SVM_sig2_TP + DjiaVolume_SVM_sig2_FP + DjiaVolume_SVM_sig2_FN + DjiaVolume_SVM_sig2_TN)
print(DjiaVolume_SVM_sig2_ACC)

DjiaVolumeAccuracyDict.update({'DjiaVolume_SVM_sig2_ACC': DjiaVolume_SVM_sig2_ACC})
print(DjiaVolumeAccuracyDict)

#############################################
###############  Sigmoid #3 #################
#############################################

DjiaVolume_sig_SVM_Model3=sklearn.svm.SVC(C=.001, kernel='sigmoid', 
                           verbose=True, gamma="scale")
DjiaVolume_sig_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
DjiaVolume_sig_svm_predict3 = DjiaVolume_sig_SVM_Model3.predict(TestDFB_nolabels)
print("\nsig SVM prediction 3:\n")
print(DjiaVolume_sig_svm_predict3)
print("\nActual:")
print(TestLabelsB)

DjiaVolume_sig_SVM_matrix3 = confusion_matrix(TestLabelsB, DjiaVolume_sig_svm_predict3)
print("\nThe sig 3 confusion matrix is:")
print(DjiaVolume_sig_SVM_matrix3)
print("\n\n")

DjiaVolume_svm_sig_target_names3 = ['0','1']
print(classification_report(TestLabelsB, DjiaVolume_sig_svm_predict3, target_names = DjiaVolume_svm_sig_target_names3))

DjiaVolume_SVM_sig3_FP = DjiaVolume_sig_SVM_matrix3[0][1] 
DjiaVolume_SVM_sig3_FN = DjiaVolume_sig_SVM_matrix3[1][0]
DjiaVolume_SVM_sig3_TP = DjiaVolume_sig_SVM_matrix3[1][1]
DjiaVolume_SVM_sig3_TN = DjiaVolume_sig_SVM_matrix3[0][0]

# Overall accuracy
DjiaVolume_SVM_sig3_ACC = (DjiaVolume_SVM_sig3_TP + DjiaVolume_SVM_sig3_TN)/(DjiaVolume_SVM_sig3_TP + DjiaVolume_SVM_sig3_FP + DjiaVolume_SVM_sig3_FN + DjiaVolume_SVM_sig3_TN)
print(DjiaVolume_SVM_sig3_ACC)

DjiaVolumeAccuracyDict.update({'DjiaVolume_SVM_sig3_ACC': DjiaVolume_SVM_sig3_ACC})
print(DjiaVolumeAccuracyDict)

#############################################
################  Poly #1 ###################
#############################################

DjiaVolume_poly_SVM_Model=sklearn.svm.SVC(C=1000, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(DjiaVolume_poly_SVM_Model)
DjiaVolume_poly_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
DjiaVolume_poly_svm_predict = DjiaVolume_poly_SVM_Model.predict(TestDFB_nolabels)
print("\npoly SVM prediction:\n")
print(DjiaVolume_poly_svm_predict)
print("\nActual:")
print(TestLabelsB)

DjiaVolume_poly_SVM_matrix = confusion_matrix(TestLabelsB, DjiaVolume_poly_svm_predict)
print("\nThe poly confusion matrix is:")
print(DjiaVolume_poly_SVM_matrix)
print("\n\n")

DjiaVolume_svm_poly_target_names = ['0','1']
print(classification_report(TestLabelsB, DjiaVolume_poly_svm_predict, target_names = DjiaVolume_svm_poly_target_names))

DjiaVolume_SVM_poly_FP = DjiaVolume_poly_SVM_matrix[0][1] 
DjiaVolume_SVM_poly_FN = DjiaVolume_poly_SVM_matrix[1][0]
DjiaVolume_SVM_poly_TP = DjiaVolume_poly_SVM_matrix[1][1]
DjiaVolume_SVM_poly_TN = DjiaVolume_poly_SVM_matrix[0][0]

# Overall accuracy
DjiaVolume_SVM_poly_ACC = (DjiaVolume_SVM_poly_TP + DjiaVolume_SVM_poly_TN)/(DjiaVolume_SVM_poly_TP + DjiaVolume_SVM_poly_FP + DjiaVolume_SVM_poly_FN + DjiaVolume_SVM_poly_TN)
print(DjiaVolume_SVM_poly_ACC)

DjiaVolumeAccuracyDict.update({'DjiaVolume_SVM_poly_ACC': DjiaVolume_SVM_poly_ACC})
print(DjiaVolumeAccuracyDict)

#############################################
################  Poly #2 ###################
#############################################

DjiaVolume_poly_SVM_Model2=sklearn.svm.SVC(C=10, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(DjiaVolume_poly_SVM_Model2)
DjiaVolume_poly_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
DjiaVolume_poly_svm_predict2 = DjiaVolume_poly_SVM_Model2.predict(TestDFB_nolabels)
print("\npoly SVM 2 prediction:\n")
print(DjiaVolume_poly_svm_predict2)
print("\nActual:")
print(TestLabelsB)

DjiaVolume_poly_SVM_matrix2 = confusion_matrix(TestLabelsB, DjiaVolume_poly_svm_predict2)
print("\nThe poly 2 confusion matrix is:")
print(DjiaVolume_poly_SVM_matrix2)
print("\n\n")

DjiaVolume_svm_poly_target_names2 = ['0','1']
print(classification_report(TestLabelsB, DjiaVolume_poly_svm_predict2, target_names = DjiaVolume_svm_poly_target_names2))

DjiaVolume_SVM_poly2_FP = DjiaVolume_poly_SVM_matrix2[0][1] 
DjiaVolume_SVM_poly2_FN = DjiaVolume_poly_SVM_matrix2[1][0]
DjiaVolume_SVM_poly2_TP = DjiaVolume_poly_SVM_matrix2[1][1]
DjiaVolume_SVM_poly2_TN = DjiaVolume_poly_SVM_matrix2[0][0]

# Overall accuracy
DjiaVolume_SVM_poly2_ACC = (DjiaVolume_SVM_poly2_TP + DjiaVolume_SVM_poly2_TN)/(DjiaVolume_SVM_poly2_TP + DjiaVolume_SVM_poly2_FP + DjiaVolume_SVM_poly2_FN + DjiaVolume_SVM_poly2_TN)
print(DjiaVolume_SVM_poly2_ACC)

DjiaVolumeAccuracyDict.update({'DjiaVolume_SVM_poly2_ACC': DjiaVolume_SVM_poly2_ACC})
print(DjiaVolumeAccuracyDict)

#############################################
################  Poly #3 ###################
#############################################

DjiaVolume_poly_SVM_Model3=sklearn.svm.SVC(C=.001, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(DjiaVolume_poly_SVM_Model3)
DjiaVolume_poly_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
DjiaVolume_poly_svm_predict3 = DjiaVolume_poly_SVM_Model3.predict(TestDFB_nolabels)
print("\npoly SVM 3 prediction:\n")
print(DjiaVolume_poly_svm_predict3)
print("\nActual:")
print(TestLabelsB)

DjiaVolume_poly_SVM_matrix3 = confusion_matrix(TestLabelsB, DjiaVolume_poly_svm_predict3)
print("\nThe poly 3 confusion matrix is:")
print(DjiaVolume_poly_SVM_matrix3)
print("\n\n")

DjiaVolume_svm_poly_target_names3 = ['0','1']
print(classification_report(TestLabelsB, DjiaVolume_poly_svm_predict3, target_names = DjiaVolume_svm_poly_target_names3))

DjiaVolume_SVM_poly3_FP = DjiaVolume_poly_SVM_matrix3[0][1] 
DjiaVolume_SVM_poly3_FN = DjiaVolume_poly_SVM_matrix3[1][0]
DjiaVolume_SVM_poly3_TP = DjiaVolume_poly_SVM_matrix3[1][1]
DjiaVolume_SVM_poly3_TN = DjiaVolume_poly_SVM_matrix3[0][0]

# Overall accuracy
DjiaVolume_SVM_poly3_ACC = (DjiaVolume_SVM_poly3_TP + DjiaVolume_SVM_poly3_TN)/(DjiaVolume_SVM_poly3_TP + DjiaVolume_SVM_poly3_FP + DjiaVolume_SVM_poly3_FN + DjiaVolume_SVM_poly3_TN)
print(DjiaVolume_SVM_poly3_ACC)

DjiaVolumeAccuracyDict.update({'DjiaVolume_SVM_poly3_ACC': DjiaVolume_SVM_poly3_ACC})
print(DjiaVolumeAccuracyDict)

DjiaVolumeVisDF = pd.DataFrame(DjiaVolumeAccuracyDict.items(), index = DjiaVolumeAccuracyDict.keys(), columns=['Model','Accuracy'])
print(DjiaVolumeVisDF)
SortedDjiaVolumeVisDF = DjiaVolumeVisDF.sort_values('Accuracy', ascending = [True])
print(SortedDjiaVolumeVisDF)

print(PartyAccuracyDict)
print(WinnerAccuracyDict)
print(SentimentAccuracyDict)
print(IncumCandAccuracyDict)
print(IncumPartyAccuracyDict)
print(UnemploymentAccuracyDict)
print(GDPAccuracyDict)
print(InflationAccuracyDict)
print(SatisfactionAccuracyDict)
print(RealIncomeGrowthAccuracyDict)
print(PresApprovalAccuracyDict)
print(IncomeTaxAccuracyDict)
print(DjiaVolumeAccuracyDict)

print(SortedPartyVisDF)
print(SortedWinnerVisDF)
print(SortedSentimentVisDF)
print(SortedIncumCandVisDF)
print(SortedIncumPartyVisDF)
print(SortedUnemploymentVisDF)
print(SortedGDPVisDF)
print(SortedInflationVisDF)
print(SortedSatisfactionVisDF)
print(SortedRealIncomeGrowthVisDF)
print(SortedPresApprovalVisDF)
print(SortedIncomeTaxVisDF)
print(SortedDjiaVolumeVisDF)

SortedPartyVisDF.plot.barh(y='Accuracy')
SortedWinnerVisDF.plot.barh(y='Accuracy')
SortedSentimentVisDF.plot.barh(y='Accuracy')
SortedIncumCandVisDF.plot.barh(y='Accuracy')
SortedIncumPartyVisDF.plot.barh(y='Accuracy')
SortedUnemploymentVisDF.plot.barh(y='Accuracy')
SortedGDPVisDF.plot.barh(y='Accuracy')
SortedInflationVisDF.plot.barh(y='Accuracy')
SortedSatisfactionVisDF.plot.barh(y='Accuracy')
SortedRealIncomeGrowthVisDF.plot.barh(y='Accuracy')
SortedPresApprovalVisDF.plot.barh(y='Accuracy')
SortedIncomeTaxVisDF.plot.barh(y='Accuracy')
SortedDjiaVolumeVisDF.plot.barh(y='Accuracy')

# running list to see which variables have been completed
#print(IndexList)
#print(TextList)
#print(PartyList)
#print(WinnerList)
#print(SentimentList)
#print(IncumCandList)
#print(IncumPartyList)
#print(UnemploymentList)
#print(GDPList)
#print(InflationList)
#print(SatisfactionList)
#print(RealIncomeGrowthList)
#print(PresApprovalList)
#print(IncomeTaxList)
#print(DjiaVolumeList)
print(CpiList)
print(CciIndexList)

#################################################################################################
#################################################################################################
### everything from this point down is a matter of find and replace for each of the variables ###
#################################################################################################
#################################################################################################

# the rows weren't all populated for the variables below so an edited text list is needed

### starting to build a model for Cpi
df_Cpi = rawfile.copy(deep=True)
df_Cpi = df_Cpi.filter(['id', 'cpi', 'comb_text'])
df_Cpi = df_Cpi[df_Cpi['cpi'].notna()]
print(df_Cpi)

CpiList = []
TextList = []
IndexList = []

for row in df_Cpi.itertuples():
    Cpilabel = row.cpi
    textlabel = row.comb_text
    textlabel = textlabel.replace('\n',' ') # remove \n
    textlabel = textlabel.replace("'","'")
    textlabel = textlabel.replace('\t',' ') # remove \t
    #textlabel = textlabel.replace(',',' ') # remove ,
    #textlabel = textlabel.replace('.',' ') # remove .
    #textlabel = textlabel.replace('"',' ') # remove "
    #textlabel = textlabel.replace("'",' ') # remove '
    textlabel = textlabel.replace('nan',' ') # remove nan
    #textlabel = textlabel.replace('-',' ') # remove -
    #textlabel = textlabel.replace(' - ',' ') # remove -
    #textlabel = textlabel.replace('?',' ') # remove ?
    #textlabel = textlabel.replace('!',' ') # remove !
    #textlabel = textlabel.replace(':',' ') # remove :
    #textlabel = textlabel.replace(';',' ') # remove ;
    #textlabel = textlabel.replace('  ',' ') # remove double spaces
    indexlabel = row.id
    TextList.append(textlabel)
    IndexList.append(indexlabel)
    CpiList.append(Cpilabel)

CpiList = [ int(x) for x in CpiList ]
print(CpiList)

X_text=MyVect.fit_transform(TextList)

Bool_X_text=MyVect.fit_transform(TextList)

tf_X_text=MyVect.fit_transform(TextList)

## Hmm - that's not quite what we want...
## Let's get the feature names which ARE the words
ColumnNamesText=MyVect.get_feature_names()
print(ColumnNamesText)

## OK good - but we want a document topic model A DTM (matrix of counts)
VectDF=pd.DataFrame(X_text.toarray(), columns=ColumnNamesText, index = IndexList)
print(VectDF)

bool_VectDF=pd.DataFrame(Bool_X_text.toarray(), columns=ColumnNamesText, index = IndexList)
print(bool_VectDF)

tf_VectDF=pd.DataFrame(tf_X_text.toarray(), columns=ColumnNamesText, index = IndexList)
print(tf_VectDF)

CpiVectDF = VectDF.copy(deep=True)
CpiVectDF.insert(loc=0, column='LABEL', value=CpiList)
print(CpiVectDF)

bool_CpiVectDF = bool_VectDF.copy(deep=True)
bool_CpiVectDF.insert(loc=0, column='LABEL', value=CpiList)
print(bool_CpiVectDF)

tf_CpiVectDF = tf_VectDF.copy(deep=True)
tf_CpiVectDF.insert(loc=0, column='LABEL', value=CpiList)
print(tf_CpiVectDF)


##--------------------------------------------------------
####################################################################
###################### model Data Build ############################
####################################################################
##--------------------------------------------------------
from sklearn.model_selection import train_test_split

## Copy everything below here to rebuild for Cpi data
TrainDF, TestDF = train_test_split(CpiVectDF, test_size=0.3)
print(TrainDF.head())

## Now we have a training set and a testing set. 
print("\nThe training set is:")
print(TrainDF)
print("\nThe testing set is:")
print(TestDF)

## IMPORTANT - YOU CANNOT LEAVE LABELS ON THE TEST SET
## Save labels
TestLabels=TestDF["LABEL"]
print(TestLabels)
## remove labels
TestDF_nolabels = TestDF.drop(["LABEL"], axis=1)
print(TestDF_nolabels)

TrainDF_nolabels=TrainDF.drop(["LABEL"], axis=1)
print(TrainDF_nolabels)

TrainLabels=TrainDF["LABEL"]
print(TrainLabels)

############# Do the same for the binary dataframe
## Create Train/Test for TrainDFB
TrainDFB, TestDFB = train_test_split(bool_CpiVectDF, test_size=0.3)
print(TrainDFB.head())

TrainDFB_nolabels=TrainDFB.drop(["LABEL"], axis=1)
print(TrainDFB_nolabels)
TrainLabelsB=TrainDFB["LABEL"]
print(TrainLabelsB)

TestDFB_nolabels=TestDFB.drop(["LABEL"], axis=1)
print(TestDFB_nolabels)
TestLabelsB=TestDFB["LABEL"]
print(TestLabelsB)

############# Do the same for the tfidf dataframe
## Create Train/Test for TrainDF tfidf
TrainDF_tf, TestDF_tf = train_test_split(tf_CpiVectDF, test_size=0.3)
print(TrainDF_tf.head())

TrainDF_tf_nolabels=TrainDF_tf.drop(["LABEL"], axis=1)
print(TrainDF_tf_nolabels)
TrainLabels_tf=TrainDF_tf["LABEL"]
print(TrainLabels_tf)

TestDF_tf_nolabels=TestDF_tf.drop(["LABEL"], axis=1)
print(TestDF_tf_nolabels)
TestLabels_tf=TestDF_tf["LABEL"]
print(TestLabels_tf)



#############################################
###########  SVM  for regular model #1 ######
#############################################
from sklearn.svm import LinearSVC
print(TrainDF_nolabels.head())
print(TrainLabels.head())

Cpi_SVM_Model=LinearSVC(C=.01)
Cpi_SVM_Model.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Cpi_svm_predict = Cpi_SVM_Model.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(Cpi_svm_predict)
print("Actual:")
print(TestLabels)

from sklearn.metrics import confusion_matrix
Cpi_SVM_matrix = confusion_matrix(TestLabels, Cpi_svm_predict)
print("\nThe confusion matrix is:")
print(Cpi_SVM_matrix)
print("\n\n")

from sklearn.metrics import classification_report
Cpi_svm_target_names = ['0','1']
print(classification_report(TestLabels, Cpi_svm_predict, target_names = Cpi_svm_target_names))

Cpi_SVM_reg_FP = Cpi_SVM_matrix[0][1] 
Cpi_SVM_reg_FN = Cpi_SVM_matrix[1][0]
Cpi_SVM_reg_TP = Cpi_SVM_matrix[1][1]
Cpi_SVM_reg_TN = Cpi_SVM_matrix[0][0]

# Overall accuracy
Cpi_SVM_reg_ACC = (Cpi_SVM_reg_TP + Cpi_SVM_reg_TN)/(Cpi_SVM_reg_TP + Cpi_SVM_reg_FP + Cpi_SVM_reg_FN + Cpi_SVM_reg_TN)
print(Cpi_SVM_reg_ACC)

CpiAccuracyDict = {}
CpiAccuracyDict.update({'Cpi_SVM_reg_ACC': Cpi_SVM_reg_ACC})
print(CpiAccuracyDict)

#############################################
###########  SVM  for regular model #2 ######
#############################################

Cpi_SVM_Model2=LinearSVC(C=1)
Cpi_SVM_Model2.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Cpi_svm_predict2 = Cpi_SVM_Model2.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(Cpi_svm_predict2)
print("Actual:")
print(TestLabels)

Cpi_SVM_matrix2 = confusion_matrix(TestLabels, Cpi_svm_predict2)
print("\nThe confusion matrix is:")
print(Cpi_SVM_matrix2)
print("\n\n")

Cpi_svm_target_names2 = ['0','1']
print(classification_report(TestLabels, Cpi_svm_predict2, target_names = Cpi_svm_target_names2))

Cpi_SVM_reg2_FP = Cpi_SVM_matrix2[0][1] 
Cpi_SVM_reg2_FN = Cpi_SVM_matrix2[1][0]
Cpi_SVM_reg2_TP = Cpi_SVM_matrix2[1][1]
Cpi_SVM_reg2_TN = Cpi_SVM_matrix2[0][0]

# Overall accuracy
Cpi_SVM_reg2_ACC = (Cpi_SVM_reg2_TP + Cpi_SVM_reg2_TN)/(Cpi_SVM_reg2_TP + Cpi_SVM_reg2_FP + Cpi_SVM_reg2_FN + Cpi_SVM_reg2_TN)
print(Cpi_SVM_reg2_ACC)

CpiAccuracyDict.update({'Cpi_SVM_reg2_ACC': Cpi_SVM_reg2_ACC})
print(CpiAccuracyDict)

#############################################
###########  SVM  for regular model #3 ######
#############################################

Cpi_SVM_Model3=LinearSVC(C=100)
Cpi_SVM_Model3.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Cpi_svm_predict3 = Cpi_SVM_Model3.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(Cpi_svm_predict3)
print("Actual:")
print(TestLabels)

Cpi_SVM_matrix3 = confusion_matrix(TestLabels, Cpi_svm_predict3)
print("\nThe confusion matrix is:")
print(Cpi_SVM_matrix3)
print("\n\n")

from sklearn.metrics import classification_report
Cpi_svm_target_names3 = ['0','1']
print(classification_report(TestLabels, Cpi_svm_predict3, target_names = Cpi_svm_target_names3))

Cpi_SVM_reg3_FP = Cpi_SVM_matrix3[0][1] 
Cpi_SVM_reg3_FN = Cpi_SVM_matrix3[1][0]
Cpi_SVM_reg3_TP = Cpi_SVM_matrix3[1][1]
Cpi_SVM_reg3_TN = Cpi_SVM_matrix3[0][0]

# Overall accuracy
Cpi_SVM_reg3_ACC = (Cpi_SVM_reg3_TP + Cpi_SVM_reg3_TN)/(Cpi_SVM_reg3_TP + Cpi_SVM_reg3_FP + Cpi_SVM_reg3_FN + Cpi_SVM_reg3_TN)
print(Cpi_SVM_reg3_ACC)

CpiAccuracyDict.update({'Cpi_SVM_reg3_ACC': Cpi_SVM_reg3_ACC})
print(CpiAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #1 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

Cpi_B_SVM_Model=LinearSVC(C=100)
Cpi_B_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Cpi_b_svm_predict = Cpi_B_SVM_Model.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(Cpi_b_svm_predict)
print("\nActual:")
print(TestLabelsB)

Cpi_B_SVM_matrix = confusion_matrix(TestLabelsB, Cpi_b_svm_predict)
print("\nThe confusion matrix is:")
print(Cpi_B_SVM_matrix)
print("\n\n")

Cpi_svm_B_target_names = ['0','1']
print(classification_report(TestLabelsB, Cpi_b_svm_predict, target_names = Cpi_svm_B_target_names))

Cpi_SVM_bool_FP = Cpi_B_SVM_matrix[0][1] 
Cpi_SVM_bool_FN = Cpi_B_SVM_matrix[1][0]
Cpi_SVM_bool_TP = Cpi_B_SVM_matrix[1][1]
Cpi_SVM_bool_TN = Cpi_B_SVM_matrix[0][0]

# Overall accuracy
Cpi_SVM_bool_ACC = (Cpi_SVM_bool_TP + Cpi_SVM_bool_TN)/(Cpi_SVM_bool_TP + Cpi_SVM_bool_FP + Cpi_SVM_bool_FN + Cpi_SVM_bool_TN)
print(Cpi_SVM_bool_ACC)

CpiAccuracyDict.update({'Cpi_SVM_bool_ACC': Cpi_SVM_bool_ACC})
print(CpiAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #2 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

Cpi_B_SVM_Model2=LinearSVC(C=1)
Cpi_B_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Cpi_b_svm_predict2 = Cpi_B_SVM_Model2.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(Cpi_b_svm_predict2)
print("\nActual:")
print(TestLabelsB)

Cpi_B_SVM_matrix2 = confusion_matrix(TestLabelsB, Cpi_b_svm_predict2)
print("\nThe confusion matrix is:")
print(Cpi_B_SVM_matrix2)
print("\n\n")

Cpi_svm_B_target_names2 = ['0','1']
print(classification_report(TestLabelsB, Cpi_b_svm_predict2, target_names = Cpi_svm_B_target_names2))

Cpi_SVM_bool2_FP = Cpi_B_SVM_matrix2[0][1] 
Cpi_SVM_bool2_FN = Cpi_B_SVM_matrix2[1][0]
Cpi_SVM_bool2_TP = Cpi_B_SVM_matrix2[1][1]
Cpi_SVM_bool2_TN = Cpi_B_SVM_matrix2[0][0]

# Overall accuracy
Cpi_SVM_bool2_ACC = (Cpi_SVM_bool2_TP + Cpi_SVM_bool2_TN)/(Cpi_SVM_bool2_TP + Cpi_SVM_bool2_FP + Cpi_SVM_bool2_FN + Cpi_SVM_bool2_TN)
print(Cpi_SVM_bool2_ACC)

CpiAccuracyDict.update({'Cpi_SVM_bool2_ACC': Cpi_SVM_bool2_ACC})
print(CpiAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #3 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

Cpi_B_SVM_Model3=LinearSVC(C=.01)
Cpi_B_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Cpi_b_svm_predict3 = Cpi_B_SVM_Model3.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(Cpi_b_svm_predict3)
print("\nActual:")
print(TestLabelsB)

Cpi_B_SVM_matrix3 = confusion_matrix(TestLabelsB, Cpi_b_svm_predict3)
print("\nThe confusion matrix is:")
print(Cpi_B_SVM_matrix3)
print("\n\n")

Cpi_svm_B_target_names3 = ['0','1']
print(classification_report(TestLabelsB, Cpi_b_svm_predict3, target_names = Cpi_svm_B_target_names3))

Cpi_SVM_bool3_FP = Cpi_B_SVM_matrix3[0][1] 
Cpi_SVM_bool3_FN = Cpi_B_SVM_matrix3[1][0]
Cpi_SVM_bool3_TP = Cpi_B_SVM_matrix3[1][1]
Cpi_SVM_bool3_TN = Cpi_B_SVM_matrix3[0][0]

# Overall accuracy
Cpi_SVM_bool3_ACC = (Cpi_SVM_bool3_TP + Cpi_SVM_bool3_TN)/(Cpi_SVM_bool3_TP + Cpi_SVM_bool3_FP + Cpi_SVM_bool3_FN + Cpi_SVM_bool3_TN)
print(Cpi_SVM_bool3_ACC)

CpiAccuracyDict.update({'Cpi_SVM_bool3_ACC': Cpi_SVM_bool3_ACC})
print(CpiAccuracyDict)

#############################################
###########  SVM  for tfidf model #1 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

Cpi_tf_SVM_Model=LinearSVC(C=.001)
Cpi_tf_SVM_Model.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Cpi_tf_svm_predict = Cpi_tf_SVM_Model.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(Cpi_tf_svm_predict)
print("\nActual:")
print(TestLabels_tf)

Cpi_tf_SVM_matrix = confusion_matrix(TestLabels_tf, Cpi_tf_svm_predict)
print("\nThe confusion matrix is:")
print(Cpi_tf_SVM_matrix)
print("\n\n")

Cpi_svm_tf_target_names = ['0','1']
print(classification_report(TestLabels_tf, Cpi_tf_svm_predict, target_names = Cpi_svm_tf_target_names))

Cpi_SVM_tf_FP = Cpi_tf_SVM_matrix[0][1] 
Cpi_SVM_tf_FN = Cpi_tf_SVM_matrix[1][0]
Cpi_SVM_tf_TP = Cpi_tf_SVM_matrix[1][1]
Cpi_SVM_tf_TN = Cpi_tf_SVM_matrix[0][0]

# Overall accuracy
Cpi_SVM_tf_ACC = (Cpi_SVM_tf_TP + Cpi_SVM_tf_TN)/(Cpi_SVM_tf_TP + Cpi_SVM_tf_FP + Cpi_SVM_tf_FN + Cpi_SVM_tf_TN)
print(Cpi_SVM_tf_ACC)

CpiAccuracyDict.update({'Cpi_SVM_tf_ACC': Cpi_SVM_tf_ACC})
print(CpiAccuracyDict)

#############################################
###########  SVM  for tfidf model #2 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

Cpi_tf_SVM_Model2=LinearSVC(C=1)
Cpi_tf_SVM_Model2.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Cpi_tf_svm_predict2 = Cpi_tf_SVM_Model2.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(Cpi_tf_svm_predict2)
print("\nActual:")
print(TestLabels_tf)

Cpi_tf_SVM_matrix2 = confusion_matrix(TestLabels_tf, Cpi_tf_svm_predict2)
print("\nThe confusion matrix is:")
print(Cpi_tf_SVM_matrix2)
print("\n\n")

Cpi_svm_tf_target_names2 = ['0','1']
print(classification_report(TestLabels_tf, Cpi_tf_svm_predict2, target_names = Cpi_svm_tf_target_names2))

Cpi_SVM_tf2_FP = Cpi_tf_SVM_matrix2[0][1] 
Cpi_SVM_tf2_FN = Cpi_tf_SVM_matrix2[1][0]
Cpi_SVM_tf2_TP = Cpi_tf_SVM_matrix2[1][1]
Cpi_SVM_tf2_TN = Cpi_tf_SVM_matrix2[0][0]

# Overall accuracy
Cpi_SVM_tf2_ACC = (Cpi_SVM_tf2_TP + Cpi_SVM_tf2_TN)/(Cpi_SVM_tf2_TP + Cpi_SVM_tf2_FP + Cpi_SVM_tf2_FN + Cpi_SVM_tf2_TN)
print(Cpi_SVM_tf2_ACC)

CpiAccuracyDict.update({'Cpi_SVM_tf2_ACC': Cpi_SVM_tf2_ACC})
print(CpiAccuracyDict)

#############################################
###########  SVM  for tfidf model #3 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

Cpi_tf_SVM_Model3=LinearSVC(C=1)
Cpi_tf_SVM_Model3.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Cpi_tf_svm_predict3 = Cpi_tf_SVM_Model3.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(Cpi_tf_svm_predict3)
print("\nActual:")
print(TestLabels_tf)

Cpi_tf_SVM_matrix3 = confusion_matrix(TestLabels_tf, Cpi_tf_svm_predict3)
print("\nThe confusion matrix is:")
print(Cpi_tf_SVM_matrix3)
print("\n\n")

Cpi_svm_tf_target_names3 = ['0','1']
print(classification_report(TestLabels_tf, Cpi_tf_svm_predict3, target_names = Cpi_svm_tf_target_names3))

Cpi_SVM_tf3_FP = Cpi_tf_SVM_matrix3[0][1] 
Cpi_SVM_tf3_FN = Cpi_tf_SVM_matrix3[1][0]
Cpi_SVM_tf3_TP = Cpi_tf_SVM_matrix3[1][1]
Cpi_SVM_tf3_TN = Cpi_tf_SVM_matrix3[0][0]

# Overall accuracy
Cpi_SVM_tf3_ACC = (Cpi_SVM_tf3_TP + Cpi_SVM_tf3_TN)/(Cpi_SVM_tf3_TP + Cpi_SVM_tf3_FP + Cpi_SVM_tf3_FN + Cpi_SVM_tf3_TN)
print(Cpi_SVM_tf3_ACC)

CpiAccuracyDict.update({'Cpi_SVM_tf3_ACC': Cpi_SVM_tf3_ACC})
print(CpiAccuracyDict)

#############################################
#######  SVM  with different kernels ########
#############################################
## running on Bernoulli model since it had best accuracy above
## i want to see if the accuracy gets improved in the different kernels

#############################################
###############  Sigmoid #1 #################
#############################################

Cpi_sig_SVM_Model=sklearn.svm.SVC(C=10, kernel='sigmoid', 
                           verbose=True, gamma="scale")
Cpi_sig_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
Cpi_sig_svm_predict = Cpi_sig_SVM_Model.predict(TestDFB_nolabels)
print("\nsig SVM prediction:\n")
print(Cpi_sig_svm_predict)
print("\nActual:")
print(TestLabelsB)

Cpi_sig_SVM_matrix = confusion_matrix(TestLabelsB, Cpi_sig_svm_predict)
print("\nThe sig confusion matrix is:")
print(Cpi_sig_SVM_matrix)
print("\n\n")

Cpi_svm_sig_target_names = ['0','1']
print(classification_report(TestLabelsB, Cpi_sig_svm_predict, target_names = Cpi_svm_sig_target_names))

Cpi_SVM_sig_FP = Cpi_sig_SVM_matrix[0][1] 
Cpi_SVM_sig_FN = Cpi_sig_SVM_matrix[1][0]
Cpi_SVM_sig_TP = Cpi_sig_SVM_matrix[1][1]
Cpi_SVM_sig_TN = Cpi_sig_SVM_matrix[0][0]

# Overall accuracy
Cpi_SVM_sig_ACC = (Cpi_SVM_sig_TP + Cpi_SVM_sig_TN)/(Cpi_SVM_sig_TP + Cpi_SVM_sig_FP + Cpi_SVM_sig_FN + Cpi_SVM_sig_TN)
print(Cpi_SVM_sig_ACC)

CpiAccuracyDict.update({'Cpi_SVM_sig_ACC': Cpi_SVM_sig_ACC})
print(CpiAccuracyDict)

#############################################
###############  Sigmoid #2 #################
#############################################

Cpi_sig_SVM_Model2=sklearn.svm.SVC(C=1000, kernel='sigmoid', 
                           verbose=True, gamma="scale")
Cpi_sig_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
Cpi_sig_svm_predict2 = Cpi_sig_SVM_Model2.predict(TestDFB_nolabels)
print("\nsig SVM prediction 2:\n")
print(Cpi_sig_svm_predict2)
print("\nActual:")
print(TestLabelsB)

Cpi_sig_SVM_matrix2 = confusion_matrix(TestLabelsB, Cpi_sig_svm_predict2)
print("\nThe sig 2 confusion matrix is:")
print(Cpi_sig_SVM_matrix2)
print("\n\n")

Cpi_svm_sig_target_names2 = ['0','1']
print(classification_report(TestLabelsB, Cpi_sig_svm_predict2, target_names = Cpi_svm_sig_target_names2))

Cpi_SVM_sig2_FP = Cpi_sig_SVM_matrix2[0][1] 
Cpi_SVM_sig2_FN = Cpi_sig_SVM_matrix2[1][0]
Cpi_SVM_sig2_TP = Cpi_sig_SVM_matrix2[1][1]
Cpi_SVM_sig2_TN = Cpi_sig_SVM_matrix2[0][0]

# Overall accuracy
Cpi_SVM_sig2_ACC = (Cpi_SVM_sig2_TP + Cpi_SVM_sig2_TN)/(Cpi_SVM_sig2_TP + Cpi_SVM_sig2_FP + Cpi_SVM_sig2_FN + Cpi_SVM_sig2_TN)
print(Cpi_SVM_sig2_ACC)

CpiAccuracyDict.update({'Cpi_SVM_sig2_ACC': Cpi_SVM_sig2_ACC})
print(CpiAccuracyDict)

#############################################
###############  Sigmoid #3 #################
#############################################

Cpi_sig_SVM_Model3=sklearn.svm.SVC(C=.001, kernel='sigmoid', 
                           verbose=True, gamma="scale")
Cpi_sig_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
Cpi_sig_svm_predict3 = Cpi_sig_SVM_Model3.predict(TestDFB_nolabels)
print("\nsig SVM prediction 3:\n")
print(Cpi_sig_svm_predict3)
print("\nActual:")
print(TestLabelsB)

Cpi_sig_SVM_matrix3 = confusion_matrix(TestLabelsB, Cpi_sig_svm_predict3)
print("\nThe sig 3 confusion matrix is:")
print(Cpi_sig_SVM_matrix3)
print("\n\n")

Cpi_svm_sig_target_names3 = ['0','1']
print(classification_report(TestLabelsB, Cpi_sig_svm_predict3, target_names = Cpi_svm_sig_target_names3))

Cpi_SVM_sig3_FP = Cpi_sig_SVM_matrix3[0][1] 
Cpi_SVM_sig3_FN = Cpi_sig_SVM_matrix3[1][0]
Cpi_SVM_sig3_TP = Cpi_sig_SVM_matrix3[1][1]
Cpi_SVM_sig3_TN = Cpi_sig_SVM_matrix3[0][0]

# Overall accuracy
Cpi_SVM_sig3_ACC = (Cpi_SVM_sig3_TP + Cpi_SVM_sig3_TN)/(Cpi_SVM_sig3_TP + Cpi_SVM_sig3_FP + Cpi_SVM_sig3_FN + Cpi_SVM_sig3_TN)
print(Cpi_SVM_sig3_ACC)

CpiAccuracyDict.update({'Cpi_SVM_sig3_ACC': Cpi_SVM_sig3_ACC})
print(CpiAccuracyDict)

#############################################
################  Poly #1 ###################
#############################################

Cpi_poly_SVM_Model=sklearn.svm.SVC(C=1000, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(Cpi_poly_SVM_Model)
Cpi_poly_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
Cpi_poly_svm_predict = Cpi_poly_SVM_Model.predict(TestDFB_nolabels)
print("\npoly SVM prediction:\n")
print(Cpi_poly_svm_predict)
print("\nActual:")
print(TestLabelsB)

Cpi_poly_SVM_matrix = confusion_matrix(TestLabelsB, Cpi_poly_svm_predict)
print("\nThe poly confusion matrix is:")
print(Cpi_poly_SVM_matrix)
print("\n\n")

Cpi_svm_poly_target_names = ['0','1']
print(classification_report(TestLabelsB, Cpi_poly_svm_predict, target_names = Cpi_svm_poly_target_names))

Cpi_SVM_poly_FP = Cpi_poly_SVM_matrix[0][1] 
Cpi_SVM_poly_FN = Cpi_poly_SVM_matrix[1][0]
Cpi_SVM_poly_TP = Cpi_poly_SVM_matrix[1][1]
Cpi_SVM_poly_TN = Cpi_poly_SVM_matrix[0][0]

# Overall accuracy
Cpi_SVM_poly_ACC = (Cpi_SVM_poly_TP + Cpi_SVM_poly_TN)/(Cpi_SVM_poly_TP + Cpi_SVM_poly_FP + Cpi_SVM_poly_FN + Cpi_SVM_poly_TN)
print(Cpi_SVM_poly_ACC)

CpiAccuracyDict.update({'Cpi_SVM_poly_ACC': Cpi_SVM_poly_ACC})
print(CpiAccuracyDict)

#############################################
################  Poly #2 ###################
#############################################

Cpi_poly_SVM_Model2=sklearn.svm.SVC(C=10, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(Cpi_poly_SVM_Model2)
Cpi_poly_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
Cpi_poly_svm_predict2 = Cpi_poly_SVM_Model2.predict(TestDFB_nolabels)
print("\npoly SVM 2 prediction:\n")
print(Cpi_poly_svm_predict2)
print("\nActual:")
print(TestLabelsB)

Cpi_poly_SVM_matrix2 = confusion_matrix(TestLabelsB, Cpi_poly_svm_predict2)
print("\nThe poly 2 confusion matrix is:")
print(Cpi_poly_SVM_matrix2)
print("\n\n")

Cpi_svm_poly_target_names2 = ['0','1']
print(classification_report(TestLabelsB, Cpi_poly_svm_predict2, target_names = Cpi_svm_poly_target_names2))

Cpi_SVM_poly2_FP = Cpi_poly_SVM_matrix2[0][1] 
Cpi_SVM_poly2_FN = Cpi_poly_SVM_matrix2[1][0]
Cpi_SVM_poly2_TP = Cpi_poly_SVM_matrix2[1][1]
Cpi_SVM_poly2_TN = Cpi_poly_SVM_matrix2[0][0]

# Overall accuracy
Cpi_SVM_poly2_ACC = (Cpi_SVM_poly2_TP + Cpi_SVM_poly2_TN)/(Cpi_SVM_poly2_TP + Cpi_SVM_poly2_FP + Cpi_SVM_poly2_FN + Cpi_SVM_poly2_TN)
print(Cpi_SVM_poly2_ACC)

CpiAccuracyDict.update({'Cpi_SVM_poly2_ACC': Cpi_SVM_poly2_ACC})
print(CpiAccuracyDict)

#############################################
################  Poly #3 ###################
#############################################

Cpi_poly_SVM_Model3=sklearn.svm.SVC(C=.001, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(Cpi_poly_SVM_Model3)
Cpi_poly_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
Cpi_poly_svm_predict3 = Cpi_poly_SVM_Model3.predict(TestDFB_nolabels)
print("\npoly SVM 3 prediction:\n")
print(Cpi_poly_svm_predict3)
print("\nActual:")
print(TestLabelsB)

Cpi_poly_SVM_matrix3 = confusion_matrix(TestLabelsB, Cpi_poly_svm_predict3)
print("\nThe poly 3 confusion matrix is:")
print(Cpi_poly_SVM_matrix3)
print("\n\n")

Cpi_svm_poly_target_names3 = ['0','1']
print(classification_report(TestLabelsB, Cpi_poly_svm_predict3, target_names = Cpi_svm_poly_target_names3))

Cpi_SVM_poly3_FP = Cpi_poly_SVM_matrix3[0][1] 
Cpi_SVM_poly3_FN = Cpi_poly_SVM_matrix3[1][0]
Cpi_SVM_poly3_TP = Cpi_poly_SVM_matrix3[1][1]
Cpi_SVM_poly3_TN = Cpi_poly_SVM_matrix3[0][0]

# Overall accuracy
Cpi_SVM_poly3_ACC = (Cpi_SVM_poly3_TP + Cpi_SVM_poly3_TN)/(Cpi_SVM_poly3_TP + Cpi_SVM_poly3_FP + Cpi_SVM_poly3_FN + Cpi_SVM_poly3_TN)
print(Cpi_SVM_poly3_ACC)

CpiAccuracyDict.update({'Cpi_SVM_poly3_ACC': Cpi_SVM_poly3_ACC})
print(CpiAccuracyDict)

CpiVisDF = pd.DataFrame(CpiAccuracyDict.items(), index = CpiAccuracyDict.keys(), columns=['Model','Accuracy'])
print(CpiVisDF)
SortedCpiVisDF = CpiVisDF.sort_values('Accuracy', ascending = [True])
print(SortedCpiVisDF)

print(PartyAccuracyDict)
print(WinnerAccuracyDict)
print(SentimentAccuracyDict)
print(IncumCandAccuracyDict)
print(IncumPartyAccuracyDict)
print(UnemploymentAccuracyDict)
print(GDPAccuracyDict)
print(InflationAccuracyDict)
print(SatisfactionAccuracyDict)
print(RealIncomeGrowthAccuracyDict)
print(PresApprovalAccuracyDict)
print(IncomeTaxAccuracyDict)
print(DjiaVolumeAccuracyDict)
print(CpiAccuracyDict)

print(SortedPartyVisDF)
print(SortedWinnerVisDF)
print(SortedSentimentVisDF)
print(SortedIncumCandVisDF)
print(SortedIncumPartyVisDF)
print(SortedUnemploymentVisDF)
print(SortedGDPVisDF)
print(SortedInflationVisDF)
print(SortedSatisfactionVisDF)
print(SortedRealIncomeGrowthVisDF)
print(SortedPresApprovalVisDF)
print(SortedIncomeTaxVisDF)
print(SortedDjiaVolumeVisDF)
print(SortedCpiVisDF)

SortedPartyVisDF.plot.barh(y='Accuracy')
SortedWinnerVisDF.plot.barh(y='Accuracy')
SortedSentimentVisDF.plot.barh(y='Accuracy')
SortedIncumCandVisDF.plot.barh(y='Accuracy')
SortedIncumPartyVisDF.plot.barh(y='Accuracy')
SortedUnemploymentVisDF.plot.barh(y='Accuracy')
SortedGDPVisDF.plot.barh(y='Accuracy')
SortedInflationVisDF.plot.barh(y='Accuracy')
SortedSatisfactionVisDF.plot.barh(y='Accuracy')
SortedRealIncomeGrowthVisDF.plot.barh(y='Accuracy')
SortedPresApprovalVisDF.plot.barh(y='Accuracy')
SortedIncomeTaxVisDF.plot.barh(y='Accuracy')
SortedDjiaVolumeVisDF.plot.barh(y='Accuracy')
SortedCpiVisDF.plot.barh(y='Accuracy')


# running list to see which variables have been completed
#print(IndexList)
#print(TextList)
#print(PartyList)
#print(WinnerList)
#print(SentimentList)
#print(IncumCandList)
#print(IncumPartyList)
#print(UnemploymentList)
#print(GDPList)
#print(InflationList)
#print(SatisfactionList)
#print(RealIncomeGrowthList)
#print(PresApprovalList)
#print(IncomeTaxList)
#print(DjiaVolumeList)
#print(CpiList)
print(CciIndexList)

#################################################################################################
#################################################################################################
### everything from this point down is a matter of find and replace for each of the variables ###
#################################################################################################
#################################################################################################

# the rows weren't all populated for the variables below so an edited text list is needed

### starting to build a model for CciIndex
df_CciIndex = rawfile.copy(deep=True)
df_CciIndex = df_CciIndex.filter(['id', 'cci_index', 'comb_text'])
df_CciIndex = df_CciIndex[df_CciIndex['cci_index'].notna()]
print(df_CciIndex)

CciIndexList = []
TextList = []
IndexList = []

for row in df_CciIndex.itertuples():
    CciIndexlabel = row.cci_index
    textlabel = row.comb_text
    textlabel = textlabel.replace('\n',' ') # remove \n
    textlabel = textlabel.replace("'","'")
    textlabel = textlabel.replace('\t',' ') # remove \t
    #textlabel = textlabel.replace(',',' ') # remove ,
    #textlabel = textlabel.replace('.',' ') # remove .
    #textlabel = textlabel.replace('"',' ') # remove "
    #textlabel = textlabel.replace("'",' ') # remove '
    textlabel = textlabel.replace('nan',' ') # remove nan
    #textlabel = textlabel.replace('-',' ') # remove -
    #textlabel = textlabel.replace(' - ',' ') # remove -
    #textlabel = textlabel.replace('?',' ') # remove ?
    #textlabel = textlabel.replace('!',' ') # remove !
    #textlabel = textlabel.replace(':',' ') # remove :
    #textlabel = textlabel.replace(';',' ') # remove ;
    #textlabel = textlabel.replace('  ',' ') # remove double spaces
    indexlabel = row.id
    TextList.append(textlabel)
    IndexList.append(indexlabel)
    CciIndexList.append(CciIndexlabel)

CciIndexList = [ int(x) for x in CciIndexList ]
print(CciIndexList)

X_text=MyVect.fit_transform(TextList)

Bool_X_text=MyVect.fit_transform(TextList)

tf_X_text=MyVect.fit_transform(TextList)

## Hmm - that's not quite what we want...
## Let's get the feature names which ARE the words
ColumnNamesText=MyVect.get_feature_names()
print(ColumnNamesText)

## OK good - but we want a document topic model A DTM (matrix of counts)
VectDF=pd.DataFrame(X_text.toarray(), columns=ColumnNamesText, index = IndexList)
print(VectDF)

bool_VectDF=pd.DataFrame(Bool_X_text.toarray(), columns=ColumnNamesText, index = IndexList)
print(bool_VectDF)

tf_VectDF=pd.DataFrame(tf_X_text.toarray(), columns=ColumnNamesText, index = IndexList)
print(tf_VectDF)

CciIndexVectDF = VectDF.copy(deep=True)
CciIndexVectDF.insert(loc=0, column='LABEL', value=CciIndexList)
print(CciIndexVectDF)

bool_CciIndexVectDF = bool_VectDF.copy(deep=True)
bool_CciIndexVectDF.insert(loc=0, column='LABEL', value=CciIndexList)
print(bool_CciIndexVectDF)

tf_CciIndexVectDF = tf_VectDF.copy(deep=True)
tf_CciIndexVectDF.insert(loc=0, column='LABEL', value=CciIndexList)
print(tf_CciIndexVectDF)


##--------------------------------------------------------
####################################################################
###################### model Data Build ############################
####################################################################
##--------------------------------------------------------
from sklearn.model_selection import train_test_split

## Copy everything below here to rebuild for CciIndex data
TrainDF, TestDF = train_test_split(CciIndexVectDF, test_size=0.3)
print(TrainDF.head())

## Now we have a training set and a testing set. 
print("\nThe training set is:")
print(TrainDF)
print("\nThe testing set is:")
print(TestDF)

## IMPORTANT - YOU CANNOT LEAVE LABELS ON THE TEST SET
## Save labels
TestLabels=TestDF["LABEL"]
print(TestLabels)
## remove labels
TestDF_nolabels = TestDF.drop(["LABEL"], axis=1)
print(TestDF_nolabels)

TrainDF_nolabels=TrainDF.drop(["LABEL"], axis=1)
print(TrainDF_nolabels)

TrainLabels=TrainDF["LABEL"]
print(TrainLabels)

############# Do the same for the binary dataframe
## Create Train/Test for TrainDFB
TrainDFB, TestDFB = train_test_split(bool_CciIndexVectDF, test_size=0.3)
print(TrainDFB.head())

TrainDFB_nolabels=TrainDFB.drop(["LABEL"], axis=1)
print(TrainDFB_nolabels)
TrainLabelsB=TrainDFB["LABEL"]
print(TrainLabelsB)

TestDFB_nolabels=TestDFB.drop(["LABEL"], axis=1)
print(TestDFB_nolabels)
TestLabelsB=TestDFB["LABEL"]
print(TestLabelsB)

############# Do the same for the tfidf dataframe
## Create Train/Test for TrainDF tfidf
TrainDF_tf, TestDF_tf = train_test_split(tf_CciIndexVectDF, test_size=0.3)
print(TrainDF_tf.head())

TrainDF_tf_nolabels=TrainDF_tf.drop(["LABEL"], axis=1)
print(TrainDF_tf_nolabels)
TrainLabels_tf=TrainDF_tf["LABEL"]
print(TrainLabels_tf)

TestDF_tf_nolabels=TestDF_tf.drop(["LABEL"], axis=1)
print(TestDF_tf_nolabels)
TestLabels_tf=TestDF_tf["LABEL"]
print(TestLabels_tf)



#############################################
###########  SVM  for regular model #1 ######
#############################################
from sklearn.svm import LinearSVC
print(TrainDF_nolabels.head())
print(TrainLabels.head())

CciIndex_SVM_Model=LinearSVC(C=.01)
CciIndex_SVM_Model.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
CciIndex_svm_predict = CciIndex_SVM_Model.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(CciIndex_svm_predict)
print("Actual:")
print(TestLabels)

from sklearn.metrics import confusion_matrix
CciIndex_SVM_matrix = confusion_matrix(TestLabels, CciIndex_svm_predict)
print("\nThe confusion matrix is:")
print(CciIndex_SVM_matrix)
print("\n\n")

from sklearn.metrics import classification_report
CciIndex_svm_target_names = ['0','1']
print(classification_report(TestLabels, CciIndex_svm_predict, target_names = CciIndex_svm_target_names))

CciIndex_SVM_reg_FP = CciIndex_SVM_matrix[0][1] 
CciIndex_SVM_reg_FN = CciIndex_SVM_matrix[1][0]
CciIndex_SVM_reg_TP = CciIndex_SVM_matrix[1][1]
CciIndex_SVM_reg_TN = CciIndex_SVM_matrix[0][0]

# Overall accuracy
CciIndex_SVM_reg_ACC = (CciIndex_SVM_reg_TP + CciIndex_SVM_reg_TN)/(CciIndex_SVM_reg_TP + CciIndex_SVM_reg_FP + CciIndex_SVM_reg_FN + CciIndex_SVM_reg_TN)
print(CciIndex_SVM_reg_ACC)

CciIndexAccuracyDict = {}
CciIndexAccuracyDict.update({'CciIndex_SVM_reg_ACC': CciIndex_SVM_reg_ACC})
print(CciIndexAccuracyDict)

#############################################
###########  SVM  for regular model #2 ######
#############################################

CciIndex_SVM_Model2=LinearSVC(C=1)
CciIndex_SVM_Model2.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
CciIndex_svm_predict2 = CciIndex_SVM_Model2.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(CciIndex_svm_predict2)
print("Actual:")
print(TestLabels)

CciIndex_SVM_matrix2 = confusion_matrix(TestLabels, CciIndex_svm_predict2)
print("\nThe confusion matrix is:")
print(CciIndex_SVM_matrix2)
print("\n\n")

CciIndex_svm_target_names2 = ['0','1']
print(classification_report(TestLabels, CciIndex_svm_predict2, target_names = CciIndex_svm_target_names2))

CciIndex_SVM_reg2_FP = CciIndex_SVM_matrix2[0][1] 
CciIndex_SVM_reg2_FN = CciIndex_SVM_matrix2[1][0]
CciIndex_SVM_reg2_TP = CciIndex_SVM_matrix2[1][1]
CciIndex_SVM_reg2_TN = CciIndex_SVM_matrix2[0][0]

# Overall accuracy
CciIndex_SVM_reg2_ACC = (CciIndex_SVM_reg2_TP + CciIndex_SVM_reg2_TN)/(CciIndex_SVM_reg2_TP + CciIndex_SVM_reg2_FP + CciIndex_SVM_reg2_FN + CciIndex_SVM_reg2_TN)
print(CciIndex_SVM_reg2_ACC)

CciIndexAccuracyDict.update({'CciIndex_SVM_reg2_ACC': CciIndex_SVM_reg2_ACC})
print(CciIndexAccuracyDict)

#############################################
###########  SVM  for regular model #3 ######
#############################################

CciIndex_SVM_Model3=LinearSVC(C=100)
CciIndex_SVM_Model3.fit(TrainDF_nolabels, TrainLabels)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
CciIndex_svm_predict3 = CciIndex_SVM_Model3.predict(TestDF_nolabels)
print("SVM prediction:\n")
print(CciIndex_svm_predict3)
print("Actual:")
print(TestLabels)

CciIndex_SVM_matrix3 = confusion_matrix(TestLabels, CciIndex_svm_predict3)
print("\nThe confusion matrix is:")
print(CciIndex_SVM_matrix3)
print("\n\n")

from sklearn.metrics import classification_report
CciIndex_svm_target_names3 = ['0','1']
print(classification_report(TestLabels, CciIndex_svm_predict3, target_names = CciIndex_svm_target_names3))

CciIndex_SVM_reg3_FP = CciIndex_SVM_matrix3[0][1] 
CciIndex_SVM_reg3_FN = CciIndex_SVM_matrix3[1][0]
CciIndex_SVM_reg3_TP = CciIndex_SVM_matrix3[1][1]
CciIndex_SVM_reg3_TN = CciIndex_SVM_matrix3[0][0]

# Overall accuracy
CciIndex_SVM_reg3_ACC = (CciIndex_SVM_reg3_TP + CciIndex_SVM_reg3_TN)/(CciIndex_SVM_reg3_TP + CciIndex_SVM_reg3_FP + CciIndex_SVM_reg3_FN + CciIndex_SVM_reg3_TN)
print(CciIndex_SVM_reg3_ACC)

CciIndexAccuracyDict.update({'CciIndex_SVM_reg3_ACC': CciIndex_SVM_reg3_ACC})
print(CciIndexAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #1 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

CciIndex_B_SVM_Model=LinearSVC(C=100)
CciIndex_B_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
CciIndex_b_svm_predict = CciIndex_B_SVM_Model.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(CciIndex_b_svm_predict)
print("\nActual:")
print(TestLabelsB)

CciIndex_B_SVM_matrix = confusion_matrix(TestLabelsB, CciIndex_b_svm_predict)
print("\nThe confusion matrix is:")
print(CciIndex_B_SVM_matrix)
print("\n\n")

CciIndex_svm_B_target_names = ['0','1']
print(classification_report(TestLabelsB, CciIndex_b_svm_predict, target_names = CciIndex_svm_B_target_names))

CciIndex_SVM_bool_FP = CciIndex_B_SVM_matrix[0][1] 
CciIndex_SVM_bool_FN = CciIndex_B_SVM_matrix[1][0]
CciIndex_SVM_bool_TP = CciIndex_B_SVM_matrix[1][1]
CciIndex_SVM_bool_TN = CciIndex_B_SVM_matrix[0][0]

# Overall accuracy
CciIndex_SVM_bool_ACC = (CciIndex_SVM_bool_TP + CciIndex_SVM_bool_TN)/(CciIndex_SVM_bool_TP + CciIndex_SVM_bool_FP + CciIndex_SVM_bool_FN + CciIndex_SVM_bool_TN)
print(CciIndex_SVM_bool_ACC)

CciIndexAccuracyDict.update({'CciIndex_SVM_bool_ACC': CciIndex_SVM_bool_ACC})
print(CciIndexAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #2 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

CciIndex_B_SVM_Model2=LinearSVC(C=1)
CciIndex_B_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
CciIndex_b_svm_predict2 = CciIndex_B_SVM_Model2.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(CciIndex_b_svm_predict2)
print("\nActual:")
print(TestLabelsB)

CciIndex_B_SVM_matrix2 = confusion_matrix(TestLabelsB, CciIndex_b_svm_predict2)
print("\nThe confusion matrix is:")
print(CciIndex_B_SVM_matrix2)
print("\n\n")

CciIndex_svm_B_target_names2 = ['0','1']
print(classification_report(TestLabelsB, CciIndex_b_svm_predict2, target_names = CciIndex_svm_B_target_names2))

CciIndex_SVM_bool2_FP = CciIndex_B_SVM_matrix2[0][1] 
CciIndex_SVM_bool2_FN = CciIndex_B_SVM_matrix2[1][0]
CciIndex_SVM_bool2_TP = CciIndex_B_SVM_matrix2[1][1]
CciIndex_SVM_bool2_TN = CciIndex_B_SVM_matrix2[0][0]

# Overall accuracy
CciIndex_SVM_bool2_ACC = (CciIndex_SVM_bool2_TP + CciIndex_SVM_bool2_TN)/(CciIndex_SVM_bool2_TP + CciIndex_SVM_bool2_FP + CciIndex_SVM_bool2_FN + CciIndex_SVM_bool2_TN)
print(CciIndex_SVM_bool2_ACC)

CciIndexAccuracyDict.update({'CciIndex_SVM_bool2_ACC': CciIndex_SVM_bool2_ACC})
print(CciIndexAccuracyDict)

#############################################
###########  SVM  for Bernoulli model #3 ####
#############################################
print(TrainDFB_nolabels.head())
print(TrainLabelsB.head())

CciIndex_B_SVM_Model3=LinearSVC(C=.01)
CciIndex_B_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
CciIndex_b_svm_predict3 = CciIndex_B_SVM_Model3.predict(TestDFB_nolabels)
print("\nBern SVM prediction:\n")
print(CciIndex_b_svm_predict3)
print("\nActual:")
print(TestLabelsB)

CciIndex_B_SVM_matrix3 = confusion_matrix(TestLabelsB, CciIndex_b_svm_predict3)
print("\nThe confusion matrix is:")
print(CciIndex_B_SVM_matrix3)
print("\n\n")

CciIndex_svm_B_target_names3 = ['0','1']
print(classification_report(TestLabelsB, CciIndex_b_svm_predict3, target_names = CciIndex_svm_B_target_names3))

CciIndex_SVM_bool3_FP = CciIndex_B_SVM_matrix3[0][1] 
CciIndex_SVM_bool3_FN = CciIndex_B_SVM_matrix3[1][0]
CciIndex_SVM_bool3_TP = CciIndex_B_SVM_matrix3[1][1]
CciIndex_SVM_bool3_TN = CciIndex_B_SVM_matrix3[0][0]

# Overall accuracy
CciIndex_SVM_bool3_ACC = (CciIndex_SVM_bool3_TP + CciIndex_SVM_bool3_TN)/(CciIndex_SVM_bool3_TP + CciIndex_SVM_bool3_FP + CciIndex_SVM_bool3_FN + CciIndex_SVM_bool3_TN)
print(CciIndex_SVM_bool3_ACC)

CciIndexAccuracyDict.update({'CciIndex_SVM_bool3_ACC': CciIndex_SVM_bool3_ACC})
print(CciIndexAccuracyDict)

#############################################
###########  SVM  for tfidf model #1 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

CciIndex_tf_SVM_Model=LinearSVC(C=.001)
CciIndex_tf_SVM_Model.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
CciIndex_tf_svm_predict = CciIndex_tf_SVM_Model.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(CciIndex_tf_svm_predict)
print("\nActual:")
print(TestLabels_tf)

CciIndex_tf_SVM_matrix = confusion_matrix(TestLabels_tf, CciIndex_tf_svm_predict)
print("\nThe confusion matrix is:")
print(CciIndex_tf_SVM_matrix)
print("\n\n")

CciIndex_svm_tf_target_names = ['0','1']
print(classification_report(TestLabels_tf, CciIndex_tf_svm_predict, target_names = CciIndex_svm_tf_target_names))

CciIndex_SVM_tf_FP = CciIndex_tf_SVM_matrix[0][1] 
CciIndex_SVM_tf_FN = CciIndex_tf_SVM_matrix[1][0]
CciIndex_SVM_tf_TP = CciIndex_tf_SVM_matrix[1][1]
CciIndex_SVM_tf_TN = CciIndex_tf_SVM_matrix[0][0]

# Overall accuracy
CciIndex_SVM_tf_ACC = (CciIndex_SVM_tf_TP + CciIndex_SVM_tf_TN)/(CciIndex_SVM_tf_TP + CciIndex_SVM_tf_FP + CciIndex_SVM_tf_FN + CciIndex_SVM_tf_TN)
print(CciIndex_SVM_tf_ACC)

CciIndexAccuracyDict.update({'CciIndex_SVM_tf_ACC': CciIndex_SVM_tf_ACC})
print(CciIndexAccuracyDict)

#############################################
###########  SVM  for tfidf model #2 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

CciIndex_tf_SVM_Model2=LinearSVC(C=1)
CciIndex_tf_SVM_Model2.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
CciIndex_tf_svm_predict2 = CciIndex_tf_SVM_Model2.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(CciIndex_tf_svm_predict2)
print("\nActual:")
print(TestLabels_tf)

CciIndex_tf_SVM_matrix2 = confusion_matrix(TestLabels_tf, CciIndex_tf_svm_predict2)
print("\nThe confusion matrix is:")
print(CciIndex_tf_SVM_matrix2)
print("\n\n")

CciIndex_svm_tf_target_names2 = ['0','1']
print(classification_report(TestLabels_tf, CciIndex_tf_svm_predict2, target_names = CciIndex_svm_tf_target_names2))

CciIndex_SVM_tf2_FP = CciIndex_tf_SVM_matrix2[0][1] 
CciIndex_SVM_tf2_FN = CciIndex_tf_SVM_matrix2[1][0]
CciIndex_SVM_tf2_TP = CciIndex_tf_SVM_matrix2[1][1]
CciIndex_SVM_tf2_TN = CciIndex_tf_SVM_matrix2[0][0]

# Overall accuracy
CciIndex_SVM_tf2_ACC = (CciIndex_SVM_tf2_TP + CciIndex_SVM_tf2_TN)/(CciIndex_SVM_tf2_TP + CciIndex_SVM_tf2_FP + CciIndex_SVM_tf2_FN + CciIndex_SVM_tf2_TN)
print(CciIndex_SVM_tf2_ACC)

CciIndexAccuracyDict.update({'CciIndex_SVM_tf2_ACC': CciIndex_SVM_tf2_ACC})
print(CciIndexAccuracyDict)

#############################################
###########  SVM  for tfidf model #3 ########
#############################################
print(TrainDF_tf_nolabels.head())
print(TrainLabels_tf.head())

CciIndex_tf_SVM_Model3=LinearSVC(C=1)
CciIndex_tf_SVM_Model3.fit(TrainDF_tf_nolabels, TrainLabels_tf)
#BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
CciIndex_tf_svm_predict3 = CciIndex_tf_SVM_Model3.predict(TestDF_tf_nolabels)
print("\nTF SVM prediction:\n")
print(CciIndex_tf_svm_predict3)
print("\nActual:")
print(TestLabels_tf)

CciIndex_tf_SVM_matrix3 = confusion_matrix(TestLabels_tf, CciIndex_tf_svm_predict3)
print("\nThe confusion matrix is:")
print(CciIndex_tf_SVM_matrix3)
print("\n\n")

CciIndex_svm_tf_target_names3 = ['0','1']
print(classification_report(TestLabels_tf, CciIndex_tf_svm_predict3, target_names = CciIndex_svm_tf_target_names3))

CciIndex_SVM_tf3_FP = CciIndex_tf_SVM_matrix3[0][1] 
CciIndex_SVM_tf3_FN = CciIndex_tf_SVM_matrix3[1][0]
CciIndex_SVM_tf3_TP = CciIndex_tf_SVM_matrix3[1][1]
CciIndex_SVM_tf3_TN = CciIndex_tf_SVM_matrix3[0][0]

# Overall accuracy
CciIndex_SVM_tf3_ACC = (CciIndex_SVM_tf3_TP + CciIndex_SVM_tf3_TN)/(CciIndex_SVM_tf3_TP + CciIndex_SVM_tf3_FP + CciIndex_SVM_tf3_FN + CciIndex_SVM_tf3_TN)
print(CciIndex_SVM_tf3_ACC)

CciIndexAccuracyDict.update({'CciIndex_SVM_tf3_ACC': CciIndex_SVM_tf3_ACC})
print(CciIndexAccuracyDict)

#############################################
#######  SVM  with different kernels ########
#############################################
## running on Bernoulli model since it had best accuracy above
## i want to see if the accuracy gets improved in the different kernels

#############################################
###############  Sigmoid #1 #################
#############################################

CciIndex_sig_SVM_Model=sklearn.svm.SVC(C=10, kernel='sigmoid', 
                           verbose=True, gamma="scale")
CciIndex_sig_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
CciIndex_sig_svm_predict = CciIndex_sig_SVM_Model.predict(TestDFB_nolabels)
print("\nsig SVM prediction:\n")
print(CciIndex_sig_svm_predict)
print("\nActual:")
print(TestLabelsB)

CciIndex_sig_SVM_matrix = confusion_matrix(TestLabelsB, CciIndex_sig_svm_predict)
print("\nThe sig confusion matrix is:")
print(CciIndex_sig_SVM_matrix)
print("\n\n")

CciIndex_svm_sig_target_names = ['0','1']
print(classification_report(TestLabelsB, CciIndex_sig_svm_predict, target_names = CciIndex_svm_sig_target_names))

CciIndex_SVM_sig_FP = CciIndex_sig_SVM_matrix[0][1] 
CciIndex_SVM_sig_FN = CciIndex_sig_SVM_matrix[1][0]
CciIndex_SVM_sig_TP = CciIndex_sig_SVM_matrix[1][1]
CciIndex_SVM_sig_TN = CciIndex_sig_SVM_matrix[0][0]

# Overall accuracy
CciIndex_SVM_sig_ACC = (CciIndex_SVM_sig_TP + CciIndex_SVM_sig_TN)/(CciIndex_SVM_sig_TP + CciIndex_SVM_sig_FP + CciIndex_SVM_sig_FN + CciIndex_SVM_sig_TN)
print(CciIndex_SVM_sig_ACC)

CciIndexAccuracyDict.update({'CciIndex_SVM_sig_ACC': CciIndex_SVM_sig_ACC})
print(CciIndexAccuracyDict)

#############################################
###############  Sigmoid #2 #################
#############################################

CciIndex_sig_SVM_Model2=sklearn.svm.SVC(C=1000, kernel='sigmoid', 
                           verbose=True, gamma="scale")
CciIndex_sig_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
CciIndex_sig_svm_predict2 = CciIndex_sig_SVM_Model2.predict(TestDFB_nolabels)
print("\nsig SVM prediction 2:\n")
print(CciIndex_sig_svm_predict2)
print("\nActual:")
print(TestLabelsB)

CciIndex_sig_SVM_matrix2 = confusion_matrix(TestLabelsB, CciIndex_sig_svm_predict2)
print("\nThe sig 2 confusion matrix is:")
print(CciIndex_sig_SVM_matrix2)
print("\n\n")

CciIndex_svm_sig_target_names2 = ['0','1']
print(classification_report(TestLabelsB, CciIndex_sig_svm_predict2, target_names = CciIndex_svm_sig_target_names2))

CciIndex_SVM_sig2_FP = CciIndex_sig_SVM_matrix2[0][1] 
CciIndex_SVM_sig2_FN = CciIndex_sig_SVM_matrix2[1][0]
CciIndex_SVM_sig2_TP = CciIndex_sig_SVM_matrix2[1][1]
CciIndex_SVM_sig2_TN = CciIndex_sig_SVM_matrix2[0][0]

# Overall accuracy
CciIndex_SVM_sig2_ACC = (CciIndex_SVM_sig2_TP + CciIndex_SVM_sig2_TN)/(CciIndex_SVM_sig2_TP + CciIndex_SVM_sig2_FP + CciIndex_SVM_sig2_FN + CciIndex_SVM_sig2_TN)
print(CciIndex_SVM_sig2_ACC)

CciIndexAccuracyDict.update({'CciIndex_SVM_sig2_ACC': CciIndex_SVM_sig2_ACC})
print(CciIndexAccuracyDict)

#############################################
###############  Sigmoid #3 #################
#############################################

CciIndex_sig_SVM_Model3=sklearn.svm.SVC(C=.001, kernel='sigmoid', 
                           verbose=True, gamma="scale")
CciIndex_sig_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
CciIndex_sig_svm_predict3 = CciIndex_sig_SVM_Model3.predict(TestDFB_nolabels)
print("\nsig SVM prediction 3:\n")
print(CciIndex_sig_svm_predict3)
print("\nActual:")
print(TestLabelsB)

CciIndex_sig_SVM_matrix3 = confusion_matrix(TestLabelsB, CciIndex_sig_svm_predict3)
print("\nThe sig 3 confusion matrix is:")
print(CciIndex_sig_SVM_matrix3)
print("\n\n")

CciIndex_svm_sig_target_names3 = ['0','1']
print(classification_report(TestLabelsB, CciIndex_sig_svm_predict3, target_names = CciIndex_svm_sig_target_names3))

CciIndex_SVM_sig3_FP = CciIndex_sig_SVM_matrix3[0][1] 
CciIndex_SVM_sig3_FN = CciIndex_sig_SVM_matrix3[1][0]
CciIndex_SVM_sig3_TP = CciIndex_sig_SVM_matrix3[1][1]
CciIndex_SVM_sig3_TN = CciIndex_sig_SVM_matrix3[0][0]

# Overall accuracy
CciIndex_SVM_sig3_ACC = (CciIndex_SVM_sig3_TP + CciIndex_SVM_sig3_TN)/(CciIndex_SVM_sig3_TP + CciIndex_SVM_sig3_FP + CciIndex_SVM_sig3_FN + CciIndex_SVM_sig3_TN)
print(CciIndex_SVM_sig3_ACC)

CciIndexAccuracyDict.update({'CciIndex_SVM_sig3_ACC': CciIndex_SVM_sig3_ACC})
print(CciIndexAccuracyDict)

#############################################
################  Poly #1 ###################
#############################################

CciIndex_poly_SVM_Model=sklearn.svm.SVC(C=1000, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(CciIndex_poly_SVM_Model)
CciIndex_poly_SVM_Model.fit(TrainDFB_nolabels, TrainLabelsB)
CciIndex_poly_svm_predict = CciIndex_poly_SVM_Model.predict(TestDFB_nolabels)
print("\npoly SVM prediction:\n")
print(CciIndex_poly_svm_predict)
print("\nActual:")
print(TestLabelsB)

CciIndex_poly_SVM_matrix = confusion_matrix(TestLabelsB, CciIndex_poly_svm_predict)
print("\nThe poly confusion matrix is:")
print(CciIndex_poly_SVM_matrix)
print("\n\n")

CciIndex_svm_poly_target_names = ['0','1']
print(classification_report(TestLabelsB, CciIndex_poly_svm_predict, target_names = CciIndex_svm_poly_target_names))

CciIndex_SVM_poly_FP = CciIndex_poly_SVM_matrix[0][1] 
CciIndex_SVM_poly_FN = CciIndex_poly_SVM_matrix[1][0]
CciIndex_SVM_poly_TP = CciIndex_poly_SVM_matrix[1][1]
CciIndex_SVM_poly_TN = CciIndex_poly_SVM_matrix[0][0]

# Overall accuracy
CciIndex_SVM_poly_ACC = (CciIndex_SVM_poly_TP + CciIndex_SVM_poly_TN)/(CciIndex_SVM_poly_TP + CciIndex_SVM_poly_FP + CciIndex_SVM_poly_FN + CciIndex_SVM_poly_TN)
print(CciIndex_SVM_poly_ACC)

CciIndexAccuracyDict.update({'CciIndex_SVM_poly_ACC': CciIndex_SVM_poly_ACC})
print(CciIndexAccuracyDict)

#############################################
################  Poly #2 ###################
#############################################

CciIndex_poly_SVM_Model2=sklearn.svm.SVC(C=10, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(CciIndex_poly_SVM_Model2)
CciIndex_poly_SVM_Model2.fit(TrainDFB_nolabels, TrainLabelsB)
CciIndex_poly_svm_predict2 = CciIndex_poly_SVM_Model2.predict(TestDFB_nolabels)
print("\npoly SVM 2 prediction:\n")
print(CciIndex_poly_svm_predict2)
print("\nActual:")
print(TestLabelsB)

CciIndex_poly_SVM_matrix2 = confusion_matrix(TestLabelsB, CciIndex_poly_svm_predict2)
print("\nThe poly 2 confusion matrix is:")
print(CciIndex_poly_SVM_matrix2)
print("\n\n")

CciIndex_svm_poly_target_names2 = ['0','1']
print(classification_report(TestLabelsB, CciIndex_poly_svm_predict2, target_names = CciIndex_svm_poly_target_names2))

CciIndex_SVM_poly2_FP = CciIndex_poly_SVM_matrix2[0][1] 
CciIndex_SVM_poly2_FN = CciIndex_poly_SVM_matrix2[1][0]
CciIndex_SVM_poly2_TP = CciIndex_poly_SVM_matrix2[1][1]
CciIndex_SVM_poly2_TN = CciIndex_poly_SVM_matrix2[0][0]

# Overall accuracy
CciIndex_SVM_poly2_ACC = (CciIndex_SVM_poly2_TP + CciIndex_SVM_poly2_TN)/(CciIndex_SVM_poly2_TP + CciIndex_SVM_poly2_FP + CciIndex_SVM_poly2_FN + CciIndex_SVM_poly2_TN)
print(CciIndex_SVM_poly2_ACC)

CciIndexAccuracyDict.update({'CciIndex_SVM_poly2_ACC': CciIndex_SVM_poly2_ACC})
print(CciIndexAccuracyDict)

#############################################
################  Poly #3 ###################
#############################################

CciIndex_poly_SVM_Model3=sklearn.svm.SVC(C=.001, kernel='poly',degree=2,
                           gamma="scale", verbose=True)

print(CciIndex_poly_SVM_Model3)
CciIndex_poly_SVM_Model3.fit(TrainDFB_nolabels, TrainLabelsB)
CciIndex_poly_svm_predict3 = CciIndex_poly_SVM_Model3.predict(TestDFB_nolabels)
print("\npoly SVM 3 prediction:\n")
print(CciIndex_poly_svm_predict3)
print("\nActual:")
print(TestLabelsB)

CciIndex_poly_SVM_matrix3 = confusion_matrix(TestLabelsB, CciIndex_poly_svm_predict3)
print("\nThe poly 3 confusion matrix is:")
print(CciIndex_poly_SVM_matrix3)
print("\n\n")

CciIndex_svm_poly_target_names3 = ['0','1']
print(classification_report(TestLabelsB, CciIndex_poly_svm_predict3, target_names = CciIndex_svm_poly_target_names3))

CciIndex_SVM_poly3_FP = CciIndex_poly_SVM_matrix3[0][1] 
CciIndex_SVM_poly3_FN = CciIndex_poly_SVM_matrix3[1][0]
CciIndex_SVM_poly3_TP = CciIndex_poly_SVM_matrix3[1][1]
CciIndex_SVM_poly3_TN = CciIndex_poly_SVM_matrix3[0][0]

# Overall accuracy
CciIndex_SVM_poly3_ACC = (CciIndex_SVM_poly3_TP + CciIndex_SVM_poly3_TN)/(CciIndex_SVM_poly3_TP + CciIndex_SVM_poly3_FP + CciIndex_SVM_poly3_FN + CciIndex_SVM_poly3_TN)
print(CciIndex_SVM_poly3_ACC)

CciIndexAccuracyDict.update({'CciIndex_SVM_poly3_ACC': CciIndex_SVM_poly3_ACC})
print(CciIndexAccuracyDict)

CciIndexVisDF = pd.DataFrame(CciIndexAccuracyDict.items(), index = CciIndexAccuracyDict.keys(), columns=['Model','Accuracy'])
print(CciIndexVisDF)
SortedCciIndexVisDF = CciIndexVisDF.sort_values('Accuracy', ascending = [True])
print(SortedCciIndexVisDF)

# running list of Accuracy Dicts
print(PartyAccuracyDict)
print(WinnerAccuracyDict)
print(SentimentAccuracyDict)
print(IncumCandAccuracyDict)
print(IncumPartyAccuracyDict)
print(UnemploymentAccuracyDict)
print(GDPAccuracyDict)
print(InflationAccuracyDict)
print(SatisfactionAccuracyDict)
print(RealIncomeGrowthAccuracyDict)
print(PresApprovalAccuracyDict)
print(IncomeTaxAccuracyDict)
print(DjiaVolumeAccuracyDict)
print(CpiAccuracyDict)
print(CciIndexAccuracyDict)

# building a combined dataframe with all of the dicts above
Combined_Accuracy_Dict = { **PartyAccuracyDict, **WinnerAccuracyDict, **SentimentAccuracyDict,
                           **IncumCandAccuracyDict, **IncumPartyAccuracyDict, **UnemploymentAccuracyDict,
                           **GDPAccuracyDict, **InflationAccuracyDict, **SatisfactionAccuracyDict,
                           **RealIncomeGrowthAccuracyDict, **PresApprovalAccuracyDict, **IncomeTaxAccuracyDict,
                           **DjiaVolumeAccuracyDict, **CpiAccuracyDict, **CciIndexAccuracyDict}
print(Combined_Accuracy_Dict)

Combined_Accuracy_VisDF = pd.DataFrame(Combined_Accuracy_Dict.items(), index = Combined_Accuracy_Dict.keys(), columns=['Model','Accuracy'])
print(Combined_Accuracy_VisDF)
SortedCombined_Accuracy_VisDF = Combined_Accuracy_VisDF.sort_values('Accuracy', ascending = [True])
print(SortedCombined_Accuracy_VisDF)

# running list of Accuracy DFs for Vis
print(SortedPartyVisDF)
print(SortedWinnerVisDF)
print(SortedSentimentVisDF)
print(SortedIncumCandVisDF)
print(SortedIncumPartyVisDF)
print(SortedUnemploymentVisDF)
print(SortedGDPVisDF)
print(SortedInflationVisDF)
print(SortedSatisfactionVisDF)
print(SortedRealIncomeGrowthVisDF)
print(SortedPresApprovalVisDF)
print(SortedIncomeTaxVisDF)
print(SortedDjiaVolumeVisDF)
print(SortedCpiVisDF)
print(SortedCciIndexVisDF)
print(SortedCombined_Accuracy_VisDF)

# descriptions of mean accuracy of each model
PartyMean = SortedPartyVisDF['Accuracy'].mean()
WinnerMean = SortedWinnerVisDF['Accuracy'].mean()
SentimentMean = SortedSentimentVisDF['Accuracy'].mean()
IncumCandMean = SortedIncumCandVisDF['Accuracy'].mean()
IncumPartyMean = SortedIncumPartyVisDF['Accuracy'].mean()
UnemploymentMean = SortedUnemploymentVisDF['Accuracy'].mean()
GdpMean = SortedGDPVisDF['Accuracy'].mean()
InflationMean = SortedInflationVisDF['Accuracy'].mean()
SatisfactionMean = SortedSatisfactionVisDF['Accuracy'].mean()
RealIncomeGrowthMean = SortedRealIncomeGrowthVisDF['Accuracy'].mean()
PresApprovalMean = SortedPresApprovalVisDF['Accuracy'].mean()
IncomeTaxMean = SortedIncomeTaxVisDF['Accuracy'].mean()
DjiaVolumeMean = SortedDjiaVolumeVisDF['Accuracy'].mean()
CpiMean = SortedCpiVisDF['Accuracy'].mean()
CciIndexMean = SortedCciIndexVisDF['Accuracy'].mean()
CombinedMean = SortedCombined_Accuracy_VisDF['Accuracy'].mean()
print(CombinedMean)

# creating a dictionary of all the means for each model for vis, excluding Combined Rating
MeansExclCombDict = {}

MeansExclCombDict.update({'PartyMean': PartyMean, 'WinnerMean': WinnerMean,'SentimentMean':SentimentMean,
                          'IncumCandMean':IncumCandMean,'IncumPartyMean':IncumPartyMean,'UnemploymentMean':UnemploymentMean,
                          'GdpMean':GdpMean,'InflationMean':InflationMean,'SatisfactionMean':SatisfactionMean,
                          'RealIncomeGrowthMean':RealIncomeGrowthMean,'PresApprovalMean':PresApprovalMean,'IncomeTaxMean':IncomeTaxMean,
                          'DjiaVolumeMean':DjiaVolumeMean,'CpiMean':CpiMean,'CciIndexMean':CciIndexMean})
print(MeansExclCombDict)


MeanAccuracyVisDF = pd.DataFrame(MeansExclCombDict.items(), index = MeansExclCombDict.keys(), columns=['Variable','Accuracy'])
print(MeanAccuracyVisDF)
SortedMeanAccuracyVisDF = MeanAccuracyVisDF.sort_values('Accuracy', ascending = [True])
print(SortedMeanAccuracyVisDF)
SortedMeanAccuracyVisDF.plot.barh(y='Accuracy')

# creating a dictionary of all the means for each model for vis, including Combined Rating
MeansInclCombDict = {}

MeansInclCombDict.update({'PartyMean': PartyMean, 'WinnerMean': WinnerMean,'SentimentMean':SentimentMean,
                          'IncumCandMean':IncumCandMean,'IncumPartyMean':IncumPartyMean,'UnemploymentMean':UnemploymentMean,
                          'GdpMean':GdpMean,'InflationMean':InflationMean,'SatisfactionMean':SatisfactionMean,
                          'RealIncomeGrowthMean':RealIncomeGrowthMean,'PresApprovalMean':PresApprovalMean,'IncomeTaxMean':IncomeTaxMean,
                          'DjiaVolumeMean':DjiaVolumeMean,'CpiMean':CpiMean,'CciIndexMean':CciIndexMean,'CombinedMean':CombinedMean})
print(MeansInclCombDict)

MeanAccuracyVisDF_withComb = pd.DataFrame(MeansInclCombDict.items(), index = MeansInclCombDict.keys(), columns=['Variable','Accuracy'])
print(MeanAccuracyVisDF_withComb)
SortedMeanAccuracyVisDF_withComb = MeanAccuracyVisDF_withComb.sort_values('Accuracy', ascending = [False])
print(SortedMeanAccuracyVisDF_withComb)
#SortedMeanAccuracyVisDF_withComb.plot.barh(y='Accuracy') Change line 10719 to True to run this line

sns.set(style="darkgrid")
rc={'font.size': 16, 'axes.labelsize': 16, 'legend.fontsize': 16.0, 
    'axes.titlesize': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 12}
sns.set(rc=rc)

# horizontal bar plots of overall model performance
sns.barplot(x="Accuracy", y="Variable", color = 'Blue', data=SortedMeanAccuracyVisDF_withComb)

plt.figure(figsize=(10,6))
sns.barplot(x="Accuracy", y="Variable", color = 'Blue', data=SortedMeanAccuracyVisDF_withComb)


# running list of Accuracy Bar Plots
SortedPartyVisDF.plot.barh(y='Accuracy')
SortedWinnerVisDF.plot.barh(y='Accuracy')
SortedSentimentVisDF.plot.barh(y='Accuracy')
SortedIncumCandVisDF.plot.barh(y='Accuracy')
SortedIncumPartyVisDF.plot.barh(y='Accuracy')
SortedUnemploymentVisDF.plot.barh(y='Accuracy')
SortedGDPVisDF.plot.barh(y='Accuracy')
SortedInflationVisDF.plot.barh(y='Accuracy')
SortedSatisfactionVisDF.plot.barh(y='Accuracy')
SortedRealIncomeGrowthVisDF.plot.barh(y='Accuracy')
SortedPresApprovalVisDF.plot.barh(y='Accuracy')
SortedIncomeTaxVisDF.plot.barh(y='Accuracy')
SortedDjiaVolumeVisDF.plot.barh(y='Accuracy')
SortedCpiVisDF.plot.barh(y='Accuracy')
SortedCciIndexVisDF.plot.barh(y='Accuracy')
#SortedCombined_Accuracy_VisDF.plot.barh(y='Accuracy')

# looking to keep only the top 20 SVM models
top20models = SortedCombined_Accuracy_VisDF.sort_values('Accuracy', ascending = [True]).tail(20)
print(top20models)
top20models.plot.barh(y='Accuracy')

SortedTop20models = top20models.sort_values('Accuracy', ascending = [False])
print(SortedTop20models)

sns.set(style="darkgrid")

## Optional for setting sizes-----------
rc={'font.size': 16, 'axes.labelsize': 16, 'legend.fontsize': 16.0, 
    'axes.titlesize': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 12}
sns.set(rc=rc)

# horizontal bar plots of top 20 models
plt.figure(figsize=(10,6))
sns.barplot(x="Accuracy", y="Model", color = 'Blue', data=SortedTop20models)


