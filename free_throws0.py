# -*- coding: utf-8 -*-
"""
NBA data 2006-16 specifically player performance, free throws, playoffs vs regular games
PLAYERS
1. Best and worst 10 players by count
2. Most consistent 10 players (ie judged by standard deviation, variation over time)
3. Correlation between standard deviation of shooting and std of shooting %
   ie the more you shoot, the less likely your shot is to be inconsistent
4. Plotting the 3 most inconsistent and inconsistent players, by std (variation over time)
   Maybe least consistent players influenced by outside factors as trajectories random?

GAMES
5. Number of games per season (regular vs playoffs)
6. Average numer of free throws per season
7. Distribution of free throws
8. Free throws per quarter (shows number clearly rises at end of quarter, max at end of game)

Created on Sat Aug  8 10:30:49 2020

@author: Owner# Tanya Reeves
Acknowledgements: Many thanks for inspiration from other Youtubers and Kagglers
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

#==============================================================================

# EYEBALLING DATA

#==============================================================================


df = pd.read_csv("free_throws.csv")

df.head(3)
df.columns
"""
Index(['end_result', 'game', 'game_id', 'period', 'play', 'player', 'playoffs',
       'score', 'season', 'shot_made', 'time'],
      dtype='object')
"""
# get a summary of the variables
df.info()
# check for missing values 
df.isnull().sum()
# no missing values, so we can continue
# get an idea of the shape of the data (to get an idea of wrangling needed)
df.shape
# (618019, 11)

#==============================================================================

# PLAYERS

#==============================================================================

# FIRST, LOOK AT BEST AND WORST 10 PLAYERS (by count, made into a %)
# create shooting dataframe by player

shooting = df.groupby(["player"])["shot_made"].agg(["size", "mean"])
shooting = shooting.rename(columns={"size": "ft_count", "mean": "percentage"})

# to make sure the shooting percentages work, considers players with 100+ shots
shooting = shooting[shooting.ft_count>=100]

shooting.head(3)
# top 10 most consistent players by count
topten = shooting.sort_values(by="percentage", ascending=False)[:10]
# 10 least consistent players by count
lowestten = shooting.sort_values(by="percentage")[:10]

print(topten)
"""
                  ft_count  percentage
player                                
Steve Nash            1591    0.913891
Brian Roberts          337    0.910979
Ray Allen             2045    0.903178
Chauncey Billups      2793    0.901540
Peja Stojakovic        455    0.901099
Stephen Curry         2129    0.897605
Brent Barry            172    0.895349
Dirk Nowitzki         4702    0.894938
Eddie House            236    0.894068
J.J. Redick           1540    0.887662
"""

print(lowestten)
"""
                ft_count  percentage
player                              
Joey Dorsey          162    0.376543
Clint Capela         268    0.376866
Andre Drummond      1459    0.378341
Kyrylo Fesenko       155    0.400000
Ben Wallace          837    0.406213
Jan Vesely           157    0.407643
DeAndre Jordan      2630    0.419392
Lou Amundson         174    0.419540
Josh Boone           478    0.447699
Louis Amundson       458    0.454148
"""

shooting_per_season = df.groupby(["player", "season"])["shot_made"].agg(["mean", "size"])

# players who have at least 100 shots per season
shooting_per_season = shooting_per_season[shooting_per_season["size"]>=100]

# dropping level "size"
shooting_per_season = shooting_per_season.drop("size", axis=1).unstack("player")

# removing the hierarchical index "mean"
shooting_per_season.columns = shooting_per_season.columns.droplevel()

# including at least 5 seasons of data
shooting_per_season = shooting_per_season.dropna(axis=1)

shooting_std = shooting_per_season.std()

# adding the overall shooting percentage which can be used as an index
shooting_std = pd.DataFrame({"std": shooting_std})
shooting_std["shooting_percentage"] = shooting.percentage

# 10 most consistent players by standard deviation (ie lack of variation)
shooting_std.sort_values(by="std").head(10)

"""
                      std  shooting_percentage
player                                        
Luol Deng        0.014586             0.775106
Kevin Martin     0.015665             0.874903
Dirk Nowitzki    0.016183             0.894938
Jarrett Jack     0.019644             0.860335
Dwyane Wade      0.020733             0.764390
Chris Bosh       0.020911             0.806629
Carmelo Anthony  0.022007             0.819857
LeBron James     0.025669             0.744532
Jamal Crawford   0.027020             0.872774
Chris Paul       0.027927             0.863994
"""

# 10 least consistent players by standard deviation (ie variation over time)
shooting_std.sort_values(by="std", ascending=False).head(10)

# As expected, there is a negative correlation between standard deviation 
# of shooting and std of shooting %
# ie the more you shoot, the less likely your shot is to be inconsistent
shooting_std.plot(kind="scatter", x="shooting_percentage", y="std", figsize=(8,5))
plt.title("Correlation of Shooting to Shooting Percentage", fontsize=15)

# plotting the 3 most inconsistent and inconsistent players, by std (variation over time)
# parallel lines for the 3 most consistent. For the inconsistent, trajectories are nonlinear
# suggesting they are influenced by outside factors?

most_consistent = shooting_std.sort_values(by="std").head(3).index
most_inconsistent = shooting_std.sort_values(by="std", ascending=False).head(3).index

fig, ax = plt.subplots(1,2, figsize=(20,5), sharey=True)

ax1 = shooting_per_season[most_consistent].plot(marker="o", rot=90, ax=ax[0], title="Top 3 Most Consistent")
ax2 = shooting_per_season[most_inconsistent].plot(marker="o", rot=90, ax=ax[1], title="Top 3 Most Inconsistent")

plt.setp(ax2.get_yticklabels(), visible=True)
plt.suptitle("Shooting Percentages over 10 Seasons", y=1.03, fontsize=20)

#===============================================================================

# GAMES

#===============================================================================

# Looking at the games themselves, number of games per season
# for regular games, there is an obvious drop 2011-12 (when there was a lockout)
# and for playoffs 2010-11 and 2014-15 (due to games played in best of 7 mode)
games = df.drop_duplicates("game_id") \
          .groupby(["season", "playoffs"]).size() \
          .unstack()
games.head()

fig, ax = plt.subplots(1,2, figsize=(15,5))
plt.suptitle("Number of Games per Season", y=1.03, fontsize=20)

games.regular.plot(marker="o", rot=90, title="Regular Season", color="#41ae76", ax=ax[0])
games.playoffs.plot(marker="o", rot=90, title="Playoffs", ax=ax[1])

#==================================================================================

# FREE THROWS

#==================================================================================

# average number of free throws (ft) per season
# the lowest for playoffs was 2011-12, for regular was 2012-13
# historically there was a rule change around 2010-11 which had an effect
# playoffs had more free throws than regular games

ft_total = df.groupby(["season", "playoffs"]).size() \
             .unstack()
ft_total.head(3)

ft_per_game = ft_total / games
ft_per_game.head(3)

ft_per_game.plot(marker="o", rot=90, figsize=(12,5))
plt.title("Average Number of Free Throws per Game", fontsize=20)

# plotting distribution of free throws
ft_per_game = df.groupby(["player", "game_id"]).size() \
                .unstack("player") \
                .mean().sort_values(ascending=False)
        
# adding shooting percentages from the shooting dataframe
ft_per_game = pd.DataFrame({"ft_per_game": ft_per_game})

# dropping those players that had less than 100 shots in the shooting dataframe
ft_per_game = ft_per_game.dropna()

ft_per_game.ft_per_game.hist(bins=50, figsize=(8,5))

#================================================================================

# DISTRIBUTION OF FREE THROWS

#================================================================================

# this graph is approximately normal, positively skewed, median is around 3.1
plt.title("Distribution of Free Throws per Player per Game", fontsize=20)
plt.xlabel("Number of Free Throws")
plt.vlines(x=ft_per_game.ft_per_game.median(), ymin=0, ymax=105, color="red", linestyle="--")
plt.text(x=2.52, y=-5, s="median", color="red")

#=============================================================================================

# FREE THROWS PER QUARTER

#=============================================================================================

# dividing up into minutes and seconds

df['minute'] = df.time.apply(lambda x: int(x[:len(x)-3]))
df['sec'] = df.time.apply(lambda x: int(x[len(x)-2:]))
df['abs_min'] = 12 - df['minute']+12*(df.period -1)
df['abs_time'] = 60*(df.abs_min-1) + 60 - df['sec']

def group_values(df,minute):
    made = len(df[(df.abs_min == minute) & (df.shot_made == 1)])
    total = len(df[df.abs_min == minute])
    return np.true_divide(made,total)

minutes = range(int(max(df.abs_min)))

per_min = []
for minu in minutes:
    per_min.append(group_values(df,minu))

plt.plot(minutes,per_min)
plt.title('Scoring % over time - Scoring improves over time')
plt.xlim([1,48])
plt.ylim([0.65,0.85])
plt.plot([12,12],[0,1], '--', linewidth = 1, color = 'r')
plt.plot([24,24],[0,1], '--', linewidth = 1, color = 'r')
plt.plot([36,36],[0,1], '--', linewidth = 1, color = 'r')
plt.plot([48,48],[0,1], '--', linewidth = 1, color = 'r')
plt.xlabel('Minute')
plt.ylabel('Free Throws %')

minutes_df = pd.DataFrame()
minutes_df['minutes'] = range(int(max(df.abs_min)))
minutes_df['shots'] = minutes_df.minutes.apply(lambda x: len(df[df.abs_min == x]))
minutes_df['players_num'] = minutes_df.minutes.apply(lambda x: len(np.unique(df.player[df.abs_min == x])))

# plot clearly shows free throws increases to the end of the quarter
# at its lowest near beginning of quarter
# major increase at end of game 




