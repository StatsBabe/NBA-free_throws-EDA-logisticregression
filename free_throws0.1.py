# -*- coding: utf-8 -*-
"""
Logistic Regression on NBA free throws data 2006-16
SUCCESS RATES IN GAMES
1. Free throw success rate between regular season and playoffs by period
   Best free throws rate is in 3rd quarter irrespective if regular or playoff
2. Free throw success rate between regular season and playoffs by season
   Success rate for free throws definitely drops in 2014 (lockout??)
   
SUCCESS RATES FOR PLAYERS
3. Evaluate individual player success
   Examples are Lebron James, Kobe Bryan, Carmelo Anthony, Dwight Howard
   They are household names and are referenced in Netflix documentary "The Last Dance" about Michael Jordan
   
PREDICTION MODEL
4. Create Logistic Regression model to evaluate player performance 
   ie whether they can make both of their free throws
5. Evaluate model using confusion matrix, accuracy, precision, recall, F1, AUC of curve

   
Created on Fri Aug  7 18:35:04 2020

@author: Owner# Tanya Reeves
Acknowledgements: Many thanks for inspiration and guidance from other Youtubers and Kagglers

"""
# =============================================================================

# IMPORT LIBRARIES

# =============================================================================
# visualisations

import seaborn as sns
import matplotlib.pyplot as plt

# data manipulation
import pandas as pd
import numpy as np

# Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#==============================================================================

# EYEBALLING DATA

#==============================================================================

# download the data
df = pd.read_csv("free_throws.csv")
# quick visual check
print(df)
# look at top 5 lines
df.head()
# look at bottom 5 lines
df.shape
# (618019, 11)

#==============================================================================

# SUCCESS BY QUARTER

#==============================================================================

# df for creating the chart
success_by_quarter =  df.groupby(['period', 'playoffs']).shot_made.sum().unstack()
total_by_quarter = df.groupby(['period', 'playoffs']).shot_made.count().unstack()
success_by_quarter['playoff_rate'] = success_by_quarter['playoffs']/total_by_quarter['playoffs']
success_by_quarter['regular_rate'] = success_by_quarter['regular'] / total_by_quarter['regular']

success_by_quarter = success_by_quarter.drop([6,7,8])
success_by_quarter = success_by_quarter.assign(period = list(range(1,6)))

print(success_by_quarter.head(10))
print(success_by_quarter.columns)
# Index(['playoffs', 'regular', 'playoff_rate', 'regular_rate', 'period'], dtype='object', name='playoffs')

#create plots
f,ax = plt.subplots(figsize=(10,5))
sns.set(style="darkgrid")
sns.lineplot(x="period", y="playoff_rate",data=success_by_quarter)
sns.lineplot(x="period", y="regular_rate",data=success_by_quarter)
plt.legend(['Playoffs','Regular'])
ax.set_xticks(success_by_quarter["period"])
ax.set_xticklabels(['1Q','2Q','3Q','4Q','OT']) # 0T = overall
ax.set_title('Free throw success rate between regular season and playoffs by period')
# this looks like the highest rate of success is 3rd quarter when settled into game

#==================================================================================

# SUCCESS BY SEASON

#==================================================================================

# df for creating the chart
#categorize seasons column
name_split2 = df.season.str.split(' - ')
df['season_cleaned'] = name_split2.str.get(0)
df['season_cleaned'] = df.season_cleaned.astype(int)

success_by_season =  df.groupby(['season_cleaned', 'playoffs']).shot_made.sum().unstack()
total_by_season = df.groupby(['season_cleaned', 'playoffs']).shot_made.count().unstack()
success_by_season['playoff_rate'] = success_by_season['playoffs']/total_by_season['playoffs']
success_by_season['regular_rate'] = success_by_season['regular'] / total_by_season['regular']


success_by_season = success_by_season.assign(season = list(range(2006,2016)))

print(success_by_season)
print(success_by_season.columns)
# Index(['playoffs', 'regular', 'playoff_rate', 'regular_rate', 'season'], dtype='object', name='playoffs')

#create plots
f,ax = plt.subplots(figsize=(10,5))
sns.set(style="darkgrid")
sns.lineplot(x="season", y="playoff_rate",data=success_by_season)
sns.lineplot(x="season", y="regular_rate",data=success_by_season)
plt.legend(['Playoffs','Regular'])
ax.set_title('Free throw success rate between regular season and playoffs by season')

# =================================================================================

# EVALUATE INDIVIDUAL PLAYER SUCCESS

# =================================================================================

# split end_result into home, away
name_split = df.end_result.str.split(' - ')
df['home_score'] = name_split.str.get(0)
df['away_score'] = name_split.str.get(1)
df['home_score'] = df.home_score.astype(int)
df['away_score'] = df.away_score.astype(int)

#split game into home team and away team 
name_split = df.game.str.split(' - ')
df['home_team'] = name_split.str.get(0)
df['away_team'] = name_split.str.get(1)

team_mapping = {'BOS': 0,'UTAH': 1,'CLE': 2,'GS': 3,'DEN': 4,'LAL': 5,'MIA': 6,'IND': 7,'LAC': 8,'HOU': 9,'CHI': 10,'ORL': 11,'TOR': 12,
                'MEM': 13,'SAC': 14,'SA': 15,'ATL': 16,'DAL': 17, 'WSH': 18,'PHX': 19,'DET': 20,'MIL': 21,'PHI': 22,'NY': 23,'POR': 24,     
                'MIN': 25,'CHA': 26,'NO': 27,'OKC': 28,'NJ': 29,'BKN': 30,'SEA': 31,'EAST': 32,'WEST': 33}

df['home_team_cleaned'] = df.home_team.map(team_mapping)
df['away_team_cleaned'] = df.away_team.map(team_mapping)

#change playoff to numeric value
df['playoffs_int'] = df.playoffs.map({'regular': 0,'playoffs': 1})

#categorize time columns 
name_split1 = df.time.str.split(':')
df['time_cleaned'] = name_split1.str.get(0)
df['time_cleaned'] = df.time_cleaned.astype(int)

#change period value to int 
df['period'] = df.period.astype(int)

#make the ID for each players
df['id'] = df.groupby(['player']).ngroup()

# Make the column that whether or not the player can make both shots 
name_split = df.play.str.split(' ')
df['made'] = name_split.str.get(5)

def int_str(x):
    if len(x) == 1:
        return int(x)

    else:
        return np.nan
    
df['made_cleaned'] = df.made.apply(lambda x:int_str(str(x)))
df['tried'] = name_split.str.get(7)
df['tried_cleaned'] = df.tried.apply(lambda x:int_str(str(x)))

def made_two_generator(made,tried):
    if made == 2 and tried == 2:
        return 1
    elif made <2 and tried == 2:
        return 0
    elif made <=1 and tried == 1:
        return np.nan
    else:
        return np.nan

df['made_two'] = df.apply(lambda x: made_two_generator(x.made_cleaned, x.tried_cleaned), axis=1)

# see if there are any null values
print(df.isna().any())

# see data types
print(df.dtypes)
# This is for next ml model
df_ml = df[['home_score','away_score','home_team_cleaned','away_team_cleaned','player','shot_made','playoffs_int','time_cleaned','season_cleaned','period']]
#Drop nan value on made_two column
df = df.dropna(subset = ['made_two'])
print(len(df))

# In this example, the choices are Lebron James, Kobe Bryan, Carmelo Anthony, Dwight Howard
# They are household names and are referenced in Netflix documentary "The Last Dance" about Michael Jordan
# Lebron James 
lebron_james = df_ml.loc[df_ml['player'].isin(['LeBron James'])]

#Kobe Bryant
kobe_bryant = df_ml.loc[df_ml['player'].isin(['Kobe Bryant'])]

#Carmelo Anthony
carmelo_anthony = df_ml.loc[df_ml['player'].isin(['Carmelo Anthony'])]

#Dwight Howard
dwight_howard = df_ml.loc[df_ml['player'].isin(['Dwight Howard'])]

def logisttic_model(df):
    # divide into features and labels
    features = df[['home_score','away_score','home_team_cleaned','away_team_cleaned','playoffs_int','time_cleaned','season_cleaned','period']]
    labels = df['shot_made']
    
    #divide into train and test sets
    train_data,test_data,train_labels,test_labels = train_test_split(features,labels,test_size = 0.20, random_state = 50)
    
    #normalize data
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    #create and evaluate model
    model = LogisticRegression()
    model.fit(train_scaled,train_labels)
    print(model.score(test_scaled,test_labels))
    print(list(zip(['home_score','away_score','home_team_cleaned','away_team_cleaned','playoffs_int','time_cleaned','season_cleaned','period'],model.coef_[0])))

#Lebron James
logisttic_model(lebron_james)
"""
0.7545284197376639
[('home_score', 0.04601267697818976), ('away_score', 0.027645733857488843), ('home_team_cleaned', 0.09542703315115966), ('away_team_cleaned', 0.06542583918312132), ('playoffs_int', 0.006377988047558601), ('time_cleaned', -0.03325270751379397), ('season_cleaned', -0.02317980399235895), ('period', 0.029586797929549765)]
"""

#Kobe Bryant
logisttic_model(kobe_bryant)
"""
0.838248436103664
[('home_score', 0.031162860185524485), ('away_score', 0.0806645249913902), ('home_team_cleaned', 0.004451704659449326), ('away_team_cleaned', 0.07741225869599547), ('playoffs_int', 0.02299269339523353), ('time_cleaned', -0.05878850080254666), ('season_cleaned', -0.1025615034792667), ('period', -0.02343303626791679)]
"""

#Carmelo Anthony
logisttic_model(carmelo_anthony)
"""
0.8289473684210527
[('home_score', 0.07335642877446376), ('away_score', -0.004076784349741302), ('home_team_cleaned', 0.001994479957716476), ('away_team_cleaned', 0.029765290092051058), ('playoffs_int', 0.05724970898863435), ('time_cleaned', -0.04360970955580643), ('season_cleaned', 0.0842730665826874), ('period', 0.010083147202036135)]
"""

#Dwight Howard
logisttic_model(dwight_howard)
"""
0.5595084087968952
[('home_score', 0.08147367980782536), ('away_score', 0.029501705543779094), ('home_team_cleaned', 0.005377584293273612), ('away_team_cleaned', -0.01618224101450984), ('playoffs_int', -0.005897136607507674), ('time_cleaned', -0.05608441413597346), ('season_cleaned', -0.16100713322821003), ('period', -0.008551479763777358)]
"""

def id_generator(df,name):
    name_df = df.loc[df['player'].isin([name])]
    id_list = []
    name_df = name_df.id.apply(lambda x:id_list.append(x))
    return id_list[0]
    

print(id_generator(df,'LeBron James')) #661
print(id_generator(df,'Ben Wallace')) #93
print(id_generator(df,'Dwight Howard')) #321
print(id_generator(df,'Kobe Bryant')) # 628
print(id_generator(df,'Dirk Nowitzki')) # 303
print(id_generator(df,'Carmelo Anthony')) #150
print(id_generator(df,'Steve Nash')) #997 #1 in count percentage (free_throws0)
print(id_generator(df,'Luol Deng')) #686 #1 in top 10 in player consistency rating (free_throws0)

#===============================================================================

# CREATE LOGISTIC REGRESSION MODEL TO EVALUATE PLAYER PERFORMANCE
# PREDICT WHETHER A PLAYER CAN MAKE BOTH OF THEIR FREE THROWS

#===============================================================================

# divide into features and labels

features = df[['home_team_cleaned','away_team_cleaned','playoffs_int','time_cleaned','season_cleaned','period','id']]
labels = df['made_two']

#divide into train and test sets
train_data,test_data,train_labels,test_labels = train_test_split(features,labels,test_size = 0.20,random_state = 50)

#normalize data

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)


#load sample data for prediction

sample1 = np.array([2,0,0,0,2008,4,661])
sample2 = np.array([20,0,0,0,2008,4,93])
sample3 = np.array([11,0,0,0,2008,4,321])
sample4 = np.array([5,0,0,0,4,2008,628])
sample5 = np.array([17,0,0,0,4,2008,303])
sample6 = np.array([150,0,0,0,4,2008,150])
# sample7 = np.array([150,0,0,0,4,2008,997])
#sample8 = np.array([150,0,0,0,4,2008,686])
sample_score = np.array([sample1,sample2,sample3,sample4,sample5,sample6])
sample = scaler.transform(sample_score)

#create and evaluate model
model = LogisticRegression(C= 0.3,random_state = 50)
model.fit(train_scaled,train_labels)
labels_pred = model.predict(test_scaled)
print(model.score(test_scaled,test_labels))
print(list(zip(['home_team_cleaned','away_team_cleaned','playoffs_int','time_cleaned','season_cleaned','period','id'],model.coef_[0])))

#put the data and predict whther Lebron can make a shot or not
print(model.predict(sample))
free_throw_probability = model.predict_proba(sample)
print(free_throw_probability)
"""
0.49724442916224915
[('home_team_cleaned', 0.003191463717931364), ('away_team_cleaned', -0.0011072211247277557), ('playoffs_int', 0.00044483170092236943), ('time_cleaned', 0.0009503726825001289), ('season_cleaned', 0.0017106883390238003), ('period', 0.0008532599568518129), ('id', -8.174927707624024e-05)]
[0. 1. 0. 1. 1. 1.]
[[0.50104048 0.49895952]
 [0.49936985 0.50063015]
 [0.50020157 0.49979843]
 [0.41703935 0.58296065]
 [0.41596001 0.58403999]
 [0.404267   0.595733  ]]
"""

#Check Lebron's sucess rate of free thorws
made_shot = []
for i in free_throw_probability :
    made_shot.append(i[1])
made_shot = np.array(made_shot)
rate = np.mean(made_shot)*100.0
rate = np.round_(rate)
print(str(rate)+"%") #54.0% ie improved from 49.72%

# =============================================================================

# EVALUATE MODEL

#==============================================================================

# import the metrics class
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(test_labels, labels_pred)
cnf_matrix
"""
array([[25075, 27228],
       [25409, 26985]], dtype=int64)
"""

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

print("Accuracy:",metrics.accuracy_score(test_labels, labels_pred))
print("Precision:",metrics.precision_score(test_labels, labels_pred))
print("Recall:",metrics.recall_score(test_labels, labels_pred))
print("F1:",metrics.f1_score(test_labels, labels_pred, average = 'micro'))
"""
Accuracy: 0.49724442916224915 
Precision: 0.49775884013059596
Recall: 0.5150398900637477
F1: 0.49724442916224915
"""
labels_pred_proba = model.predict_proba(test_scaled)[::,1]
fpr, tpr, _ = metrics.roc_curve(test_labels,  labels_pred_proba)
auc = metrics.roc_auc_score(test_labels, labels_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
# auc = 0.4963 ie it approx the same as to flip a coin which is 50%

