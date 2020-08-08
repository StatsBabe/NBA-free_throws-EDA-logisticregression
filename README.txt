README

Created on Fri Aug  7 18:35:04 2020

@author: Owner# Tanya Reeves
Acknowledgements: Many thanks for inspiration and guidance from other Githubers, Youtubers and Kagglers

There are two files free_throws0, a basic EDA and free_throws0.1 a Logistic Regression model:-

(i) free_throws0

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

(ii) free_throws0.1

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
   
Regrettably, this model was about as accurate as throwing a coin ie 50%
This is initially disappointing but understandable. The model is not sophisticated enough for such complexity.
Matthew Syed in "Bouce, The Myth of Talent and the Power of Practice" (2011, Fourth Estate, London)
comments (p. 48),
"Most sports are characterized by combinatorial explosion: tennis, table tennis, football, ice hockey and so on.
... The complexities are almost to impossible to define, let alone solve."

On to the next model....
   
