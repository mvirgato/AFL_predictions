import pandas as pd
from libs.neural_nets import DenseNN
from torch import load, tensor
from libs.data_processing import get_team_info, process_data, current_round_data, get_seasonal_data
import os


if __name__ == '__main__':

    model = DenseNN(16, 10, 3, 3) 
    model.load_state_dict(load('AFL_prediction_model_DNN.pth', weights_only=True))

    team_dict = get_team_info()

    #Need to speed this up
    get_seasonal_data()
    full_data = process_data()
    full_data.iloc[-1]

    def predict(ateam, hteam, is_final):

        ateamid = team_dict[ateam]
        hteamid = team_dict[hteam]

        away_info = full_data[full_data['ateamid']==ateamid].iloc[-1][['round', 'ateamid', 'arank', 'ascore_ppg', 'agoals_ppg', 'abehinds_ppg', 'awins', 'alosses']]
        home_info = full_data[full_data['hteamid']==hteamid].iloc[-1][['hteamid', 'hrank', 'hscore_ppg', 'hgoals_ppg','hbehinds_ppg', 'hwins', 'hlosses']]

        
        input_data = pd.concat([away_info, home_info])[['round', 'hteamid', 'ateamid', 'hrank', 'arank', 'hscore_ppg', 'ascore_ppg', 'hgoals_ppg', 'agoals_ppg', 'hbehinds_ppg', 'abehinds_ppg', 'hwins', 'awins', 'hlosses', 'alosses']]
        input_data['is_final'] = is_final
        
        input_data = tensor(input_data.values).float()

        model_output = model.predict(input_data).detach().numpy()[0]

        output = model_output.argmax()

        if output == 0:
            return f'{ateam} to win with {model_output[0]:0.2f} probability'
        elif output == 1:
            return f'{hteam} to win with {model_output[1]:0.2f} probability'
        else:
            return f'Draw with {model_output[2]:0.2f} probability'
        

    current_round = int(input("Enter Current Round:"))
    schedule = current_round_data(current_round)

    for row in schedule.iterrows():
        row = row[1]
        print(predict(row['ateam'], row['hteam'], row['is_final']))

    # os.remove('current_round.csv')