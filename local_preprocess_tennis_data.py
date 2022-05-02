import os.path

import pandas as pd
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)



def save_data_wl(u,w,los,player_name):
    y = np.ones(los.shape[0])
    l=los
    r=w
    fn = f'tennis_data_processed_wl_{player_name}'
    if not os.path.exists(fn):
        os.makedirs(fn)
    with open(f'{fn}/u.npy', 'wb') as f:
        np.save(f, u)
    with open(f'{fn}/l_processed.npy', 'wb') as f:
        np.save(f, l)
    with open(f'{fn}/r_processed.npy', 'wb') as f:
        np.save(f, r)
    with open(f'{fn}/y.npy', 'wb') as f:
        np.save(f, y)



if __name__ == '__main__':
    player_name ='d643' #'Djokovic'
    matches = pd.read_csv('tennis_data/match_scores_1991-2016_unindexed.csv')[['tourney_url_suffix','winner_player_id','loser_player_id']]
    env_data = pd.read_csv('tennis_data/tournaments_1877-2017_unindexed.csv')[['tourney_url_suffix','tourney_conditions','tourney_surface']]
    matches = matches.merge(env_data,on='tourney_url_suffix',how='left')

    players = pd.read_csv('tennis_data/player_overviews_unindexed.csv')[['player_id','birth_year','weight_kg','height_cm','handedness','backhand','turned_pro']]
    players['pro_age'] = players['turned_pro']-players['birth_year']
    players['pro_age'] = players['pro_age'].apply(lambda x : 18 if x<0 else x)

    players = players.drop(['turned_pro'],axis=1)
    players['handedness'].fillna(('unknown'), inplace=True)
    players['backhand'].fillna(('unknown'), inplace=True)

    players['birth_year'].fillna((players['birth_year'].mean()), inplace=True)
    players['weight_kg'].fillna((players['weight_kg'].median()), inplace=True)
    players['height_cm'].fillna((players['height_cm'].median()), inplace=True)
    players['pro_age'].fillna((players['pro_age'].median()), inplace=True)
    players = players.dropna()
    players = pd.get_dummies(players,columns=['handedness','backhand'])
    matches = pd.get_dummies(matches,columns=['tourney_conditions','tourney_surface'])
    E = pd.get_dummies(env_data,columns=['tourney_conditions','tourney_surface'])
    S_u = E.drop(['tourney_url_suffix'],axis=1).values

    mask = matches['loser_player_id'].isin([player_name]) #|  matches['winner_player_id'].isin([player_name])
    matches = matches[mask]
    winners = matches['winner_player_id']
    w=winners.to_frame().merge(players, left_on='winner_player_id', right_on='player_id', how='left')
    loosers = matches['loser_player_id']
    los=loosers.to_frame().merge(players, left_on='loser_player_id', right_on='player_id', how='left')
    w = w.drop(['winner_player_id','player_id'],axis=1)
    los = los.drop(['loser_player_id','player_id'],axis=1)
    w = w.values
    los = los.values
    u = matches.drop(['tourney_url_suffix','loser_player_id','winner_player_id'],axis=1).values

    all_relevant_players = pd.concat([loosers,winners],axis=0)
    all_relevant_players=all_relevant_players.drop_duplicates()
    S=all_relevant_players.to_frame().merge(players, left_on=0, right_on='player_id', how='inner')
    S = S.drop(['player_id',0],axis=1).values
    # save_data(w,los)
    save_data_wl(u,w,los,player_name)

    # l,r = [],[]
    # y=[]
    # for i in range(w.shape[0]):
    #     y_val = np.random.choice([-1,1])
    #     y.append(y_val)
    #     if y_val==1:
    #         r.append(w[i,:])
    #         l.append(los[i,:])
    #     else:
    #         r.append(los[i,:])
    #         l.append(w[i,:])
    #
    # y=np.array(y)
    # l=np.stack(l,axis=0)
    # r=np.stack(r,axis=0)
    # # y = np.ones(r.shape[0])
    #
    #
    # if not os.path.exists(fn):
    #     os.makedirs(fn)
    #
    # with open(f'{fn}/y.npy', 'wb') as f:
    #     np.save(f, y)
    # with open(f'{fn}/u.npy', 'wb') as f:
    #     np.save(f, u)
    # with open(f'{fn}/l_processed.npy', 'wb') as f:
    #     np.save(f, l)
    # with open(f'{fn}/r_processed.npy', 'wb') as f:
    #     np.save(f, r)
    # with open(f'{fn}/S.npy', 'wb') as f:
    #     np.save(f, S)
    # with open(f'{fn}/S_u.npy', 'wb') as f:
    #     np.save(f, S_u)











