# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 19:58:30 2020

@author: rynmc
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

#%%
n_embedding=20
n_hidden1=35
n_hidden2=12
n_epochs=15
p_val=0.1

#%%

def cleanTeamNames(teamNameArray):
    cleanedTeamNameArray=[]
    for team in teamNameArray:
        team=team.replace("Miami (FL)", "Miami-FL")
        team=team.replace("Miami (OH)", "Miami-OH")
        team=team.split(")")
        if len(team)>1:
            team=team[-1].strip()
        else:
            team=team[0]
        cleanedTeamNameArray.append(team)
    return cleanedTeamNameArray

def buildTeamDictionary(winners, losers):
    dictionary = {}
    for team in np.append(winners.values, losers.values):
        if team not in dictionary.keys():
            dictionary[team]=len(dictionary)
        
    return dictionary, len(dictionary)

def buildWindows(df, dictionary):
    
    tensorListX=[]
    tensorListY=[]
    
    for index, row in df.iterrows():
        x1=dictionary[row['Winner']]
        x2=dictionary[row['Loser']]
        x_win=tf.constant([x1,x2], shape=(1,2))
        y_win=tf.constant([int(row['Pts'])], shape=(1,1))
        x_lose=tf.constant([x2,x1],shape=(1,2))
        y_lose=tf.constant([int(row['Pts.1'])], shape=(1,1))
        tensorListX.append(x_win)
        tensorListX.append(x_lose)
        tensorListY.append(y_win)
        tensorListY.append(y_lose)
        
    x_train=tf.stack(tensorListX)
    y_train=tf.stack(tensorListY)
    
    return x_train, y_train
        
    
        
    
#%%
resultDf = pd.read_html("https://www.sports-reference.com/cfb/years/2020-schedule.html")[0]

#remove subheader rows

resultDf=resultDf[resultDf['Rk']!='Rk']

#remove unplayed games

resultDf=resultDf[[not x for x in pd.isna(resultDf['Pts'])]]

#remove rank from team name

resultDf['Winner']=cleanTeamNames(resultDf['Winner'])
resultDf['Loser']=cleanTeamNames(resultDf['Loser'])

#build dictionary of teams

dictionary, n_teams =buildTeamDictionary(resultDf['Winner'], resultDf['Loser'])

#build windows of labels

x_train, y_train = buildWindows(resultDf, dictionary)

#build model

model=keras.Sequential()
model.add(keras.layers.Input((1,2)))
model.add(keras.layers.Embedding(n_teams, n_embedding))
model.add(keras.layers.Reshape(target_shape=(1,-1)))
model.add(keras.layers.Dense(n_hidden1))
model.add(keras.layers.Dense(n_hidden2))
model.add(keras.layers.Dense(1))

model.compile(loss="MeanSquaredError", optimizer='adam')

model.fit(x=x_train, y=y_train, epochs=n_epochs, validation_split=p_val)

#%% predict games

def predictGame(team1, team2, model,dictionary):
    x1=dictionary[team1]
    x2=dictionary[team2]
    points1=model(tf.constant([x1,x2], shape=(1,2)))
    points2=model(tf.constant([x2,x1], shape=(1,2)))
    print(team1 + ":" + str(points1.numpy()[0][0][0]) + " " + team2 + ":" + str(points2.numpy()[0][0][0]))
        
predictGame("Ohio State", "Michigan State", model, dictionary)
        
        