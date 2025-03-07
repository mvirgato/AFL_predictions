import numpy as np
import pandas as pd
import requests
from os.path import isfile
from glob import glob
import re
import os

years = np.arange(1970, 2025)
rounds = np.arange(1, 29)

headers = {'User-Agent':'MV_tipping_predictions'}

# Some helper functions for use later

def get_year(file):
    
    file = re.sub('^Seasonal_Data/data_', '', file)    
    file = file.rstrip('.csv')

    return int(file)

def get_round(file):
    
    file = re.sub('^Standings/\w+/round_', '', file)    
    file = file.rstrip('.csv')

    return int(file)

def read_ranking(file):

    _round = get_round(file)

    _df  = pd.read_csv(file)

    _df.insert(loc=0, column='round', value=_round * np.ones(len(_df), dtype= int))

    return _df


def get_team_info():

    '''
    Returns a dictionary to translate team name to ID number for use 
    in neural nets
    '''
    if not isfile('team_data.csv'):
        team_data = requests.get('https://api.squiggle.com.au/?q=teams;format=csv', headers=headers)

        with open('team_data.csv', 'w') as f:
            f.write(team_data.text)

    team_data = pd.read_csv('team_data.csv', index_col=None)
    # team_dict = team_data[['name', 'id']].to_dict('index')
    names = team_data['name'].to_numpy()
    ids = team_data['id'].to_numpy()

    team_dict = {name : id_ for id_, name in zip(ids, names)}

    return team_dict

def get_seasonal_data():

    for year in years:

        if isfile(f'Seasonal_Data/data_{year:d}.csv'):

            # print('Data already downloaded.')
            
            continue

        response = requests.get(f'https://api.squiggle.com.au/?q=games;year={year:d};format=csv', headers=headers)

        with open(f'Seasonal_Data/data_{year:d}.csv', 'w') as f:
            f.write(response.text)

    rounds = np.arange(1, 29)

def get_standings():

    for year in years:
        try:
            os.mkdir(f'Standings/{year:d}/')
        except:
            pass
        for round_ in rounds:

            if isfile(f'Standings/{year:d}/round_{round_:d}.csv'):

                # print('Data already downloaded.')
                
                continue
            
            response = requests.get(f'https://api.squiggle.com.au/?q=standings;year={year:d};round={round_};format=csv', headers=headers)

            with open(f'Standings/{year:d}/round_{round_:d}.csv', 'w') as f:
                f.write(response.text)

# Function to read in ranking, number of wins and number of losses
# for each team for each round

def standings_data(year):
    rank_files = glob(f'Standings/{year:d}/*')

    standings_frame = pd.concat(read_ranking(file) for file in rank_files)

    # Fill NaN values for rank in finals rounds with the mean of the 
    # team's rank # throughout the season
    standings_frame['rank'] = standings_frame['rank'].fillna(
        standings_frame.groupby('id')['rank'].transform('mean')
        )

    standings_frame = standings_frame[['round', 'id', 'rank', 'wins','losses']]
    standings_frame = standings_frame.sort_values('round')
    standings_frame = standings_frame.reset_index(drop=True)

    return standings_frame

def merge_standings(main_data, standings_data):

    lookup = standings_data[['round', 'id', 'rank', 'wins', 'losses']]

    # Merge with standings_data to filter matching rows
    for prefix in ['h', 'a']:

        filtered_main_data = main_data.merge(lookup, left_on=['round', f'{prefix}teamid'], right_on=['round', 'id'], how='inner')
        filtered_main_data = filtered_main_data.rename(columns= {'rank':f'{prefix}rank', 'wins': f'{prefix}wins', 'losses': f'{prefix}losses'})
        main_data = filtered_main_data.drop('id', axis=1)

    return main_data

def process_season_data(datafile):

    # Select relevant columns

    dataframe = pd.read_csv(datafile)

    dataframe = dataframe[['round', 'hteamid', 'ateamid', 'hscore', 
                           'ascore', 'hgoals', 'agoals', 'hbehinds', 
                           'abehinds', 'is_final', 'winnerteamid']].copy()

    # Fill NaN values in winnerteamid
    dataframe['winnerteamid'] = dataframe['winnerteamid'].fillna(0)

    # Compute hteamwin using np.where()
    dataframe['hteamwin'] = np.where(
        dataframe['winnerteamid'] == 0, 2, 
        np.where(dataframe['winnerteamid'] == dataframe['hteamid'], 1, 0)
    )



    # Compute expanding mean for each team using groupby()
    column_team_mapping = { 
        'hscore': 'hteamid', 'hgoals': 'hteamid', 'hbehinds': 'hteamid', 
        'ascore': 'ateamid', 'agoals': 'ateamid', 'abehinds': 'ateamid' 
        }

    # Compute expanding means efficiently in a single loop
    for col, team_col in column_team_mapping.items():
        dataframe[f'{col}_ppg'] = (
            dataframe.groupby(team_col)[col]
            .expanding().mean()
            .reset_index(level=0, drop=True)
        )
    
    year = get_year(datafile)
    dataframe = merge_standings(dataframe, standings_data(year))

    # Select final columns
    final_data = dataframe[['round', 
                            'hteamid', 'ateamid',
                            'hrank', 'arank', 
                            'hscore_ppg', 'ascore_ppg', 
                            'hgoals_ppg', 'agoals_ppg', 
                            'hbehinds_ppg', 'abehinds_ppg', 
                            'hwins', 'awins', 
                            'hlosses', 'alosses',  
                            'is_final', 'hteamwin']].copy()
    
    
    return final_data

def process_data():

    data_files = glob('Seasonal_Data/data_*.csv')
    full_data = pd.concat(process_season_data(file) for file in data_files)
    return full_data


def current_round_data(_round):

    current_round = requests.get('https://api.squiggle.com.au/?q=games;year=2025;round={_round:d}};format=csv', headers=headers)

    with open('current_round.csv', 'w') as f:
        f.write(current_round.text)

    return pd.read_csv('current_round.csv')[['ateam', 'hteam', 'is_final']]
