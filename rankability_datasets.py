import copy
import pickle

import pandas as pd
import numpy as np
from pref_shap.rankability_test import specR
import tqdm
# use_these_columns=['bpid',
#                    'uniquesessionid',
#                    'trans_date',
#                    'click',
#                    'view',
#                    'is_seasonless',
#                    'is_carried_over',
#                    'is_running_item',
#                    'product_type_name',
#                    'product_group_name',
#                    'graphical_appearance_name',
#                    'colour_name',
#                     'assortment_mix_name',
#                    'licence_company_name',
#                    'section_name',
#                    'composition',
#                    'garment_group_name',
#                    ]

def create_adjacency_matrix(a_list):
    a_vec = np.array(a_list)[:,np.newaxis]
    a = a_vec - a_vec.transpose()
    return np.clip(a,-1,1)

def get_spec_r(a_list):
    a=create_adjacency_matrix(a_list)
    return specR(a)

def get_list_of_alist(df):
    sess_id=df['uniquesessionid'].tolist()
    clicks=df['click'].tolist()
    list_of_alist=[]
    init_list=[]
    for i in tqdm.tqdm(range(len(sess_id)-1)):
        cur = sess_id[i]
        next = sess_id[i+1]
        init_list.append(float(clicks[i]))
        if not cur==next:
            if len(init_list)>1 and sum(init_list)>0:
                list_of_alist.append(copy.deepcopy(init_list))
            init_list=[]
    return list_of_alist


def website_rankability():
    df = pd.read_parquet('pref_user_2.parquet')
    df=df.sort_values(by=['bpid', 'uniquesessionid'])
    ls=get_list_of_alist(df)
    spec_r_list =[]
    for a_list in tqdm.tqdm(ls):
        spec_r_val=get_spec_r(a_list)
        spec_r_list.append(spec_r_val)
    data = np.array(spec_r_list)
    return data.mean(0), data.std(0)


def tennis_generate_b_mats(df):
    sess_id = df['tourney_url_suffix'].tolist()
    tourney_indices = []
    spec_R_list = []
    for i in tqdm.tqdm(range(len(sess_id) - 1)):
        cur = sess_id[i]
        next = sess_id[i + 1]
        tourney_indices.append(df.iloc[i,1:].values.tolist())

        if not cur == next:
            matches = pd.DataFrame(tourney_indices,columns=['winner_player_id','loser_player_id'])
            matches,num_players=reform_df_tennis(matches)
            bmat = np.zeros((num_players, num_players))

            for el in matches.values.tolist():
                i,j = el
                bmat[i,j]=1
                bmat[j, i] = -1
            s = specR(bmat)
            spec_R_list.append(s)
            tourney_indices = []
    return np.array(spec_R_list)

def reform_df_tennis(matches):
    winners = matches['winner_player_id'].unique().tolist()
    losers = matches['loser_player_id'].unique().tolist()
    unique_list = winners
    for l in losers:
        if l in unique_list:
            pass
        else:
            unique_list.append(l)
    num_players = len(unique_list)
    player_dict = {el: i for i, el in enumerate(unique_list)}
    matches['winner_player_id'] = matches['winner_player_id'].apply(lambda x: player_dict[x])
    matches['loser_player_id'] = matches['loser_player_id'].apply(lambda x: player_dict[x])
    return matches,num_players
def tennis_rankability():
    matches = pd.read_csv('tennis_data/match_scores_1991-2016_unindexed.csv')[['tourney_url_suffix','winner_player_id','loser_player_id']]
    df=matches.sort_values(by=['tourney_url_suffix'])
    spec_r_list = tennis_generate_b_mats(df)
    return np.mean(spec_r_list),np.std(spec_r_list)

def cartesian_product(*arrays):
    ndim = len(arrays)
    return (np.stack(np.meshgrid(*arrays), axis=-1)
              .reshape(-1, ndim))

def deciding_features(left,right,D):
    a = np.zeros((D,D))
    iu = np.triu_indices(D,1)
    a[iu]=np.arange(1,D+1)
    a=a+np.transpose(a)
    dec_feat_index = a[left,right]
    return dec_feat_index.astype(int)

def one_hot_numpy(a):
    b = np.zeros((a.size, a.max()+1))
    b[np.arange(a.size), a] = 1
    return b

def synthetic_rankability():
    D=3
    players=1000
    hidden_cluster_list = list(range(D))
    hidden_states = np.random.choice(hidden_cluster_list, players)
    total_covs = np.cumsum(np.arange(D))[-1] + 1
    x_cov = np.random.randn(players,
                            total_covs)
    n_matches = 40000
    print(np.corrcoef(x_cov.T))
    a = np.arange(players)
    all_combinations = cartesian_product(a, a)
    all_matches = all_combinations[all_combinations[:, 0] != all_combinations[:, 1]]
    matches = all_matches[np.random.choice(all_matches.shape[0], n_matches, replace=False)]
    left_hidden, left_cov = hidden_states[matches[:, 0]], x_cov[matches[:, 0]]
    right_hidden, right_cov = hidden_states[matches[:, 1]], x_cov[matches[:, 1]]
    d_feat = deciding_features(left_hidden, right_hidden, D)
    y = []

    wl = np.zeros((n_matches,2))

    right_cov = np.concatenate([right_cov, one_hot_numpy(right_hidden)], axis=1)
    left_cov = np.concatenate([left_cov, one_hot_numpy(left_hidden)], axis=1)
    for i, d in enumerate(d_feat):
        y_tmp = np.sign(right_cov[i, d] - left_cov[i, d])
        if y_tmp>0:
            wl[i,0] = matches[i,0]
            wl[i,1] = matches[i,1]
        else:
            wl[i,0] = matches[i,1]
            wl[i,1] = matches[i,0]

    b_train = pd.DataFrame(np.zeros((players, players)),
                           columns=np.arange(players), index=np.arange(players))
    contest = pd.DataFrame(wl,columns=['Loser','Winner'])
    for row in range(contest.shape[0]):
        hold = contest.iloc[row, :]
        i, j = hold["Winner"], hold["Loser"]
        b_train.loc[i, j] += 1
        b_train.loc[j, i] += -1

    B_train = np.array(b_train)
    return specR(B_train)

def chameleon_rankability():
    contest = pd.read_csv("./Chameleons/matches.csv",
                          index_col="Unnamed: 0")
    contest.columns = ["Winner", "Loser"]
    predictors = pd.read_csv(
        "./Chameleons/predictors.csv", index_col="Unnamed: 0")
    num_players = predictors.shape[0]

    # Randomise
    ind = predictors.index.values

    # Create binary matrix representing winning and losing
    b_train = pd.DataFrame(np.zeros((num_players, num_players)),
                           columns=predictors.index, index=predictors.index)

    for row in range(contest.shape[0]):
        hold = contest.iloc[row, :]
        i, j = hold["Winner"], hold["Loser"]
        b_train.loc[i, j] += 1
        b_train.loc[j, i] += -1

    B_train = np.array(b_train)
    return specR(B_train)

def pokemon_rankability():
    contest_1 = pd.read_csv("./Pokemon/combats.csv")
    contest_2 = pd.read_csv("./Pokemon/tests.csv")
    stats = pd.read_csv("./Pokemon/pokemon.csv", index_col="#")
    num_players = stats.shape[0]
    contest = pd.concat([contest_1,contest_2],axis=0).reset_index()
    b_train = pd.DataFrame(np.zeros((num_players, num_players)),
                           columns=stats.index, index=stats.index)
    for i in range(contest.shape[0]):
        hold = contest.iloc[i, :]
        if hold["First_pokemon"] == hold["Winner"]:
            i,j=hold["First_pokemon"], hold["Second_pokemon"]
        else:
            i,j=hold["Second_pokemon"], hold["First_pokemon"]
        i,j = int(i),int(j)
        b_train.loc[i, j] += 1
        b_train.loc[j, i] += -1
    B_train = np.array(b_train)
    return specR(B_train)

if __name__ == '__main__':


    dat = {}
    spec_r_synthetic = synthetic_rankability()
    print(spec_r_synthetic)
    spec_r_mean_tennis,spec_r_std_tennis=tennis_rankability()
    spec_r_mean_web,spec_r_std_web=website_rankability()
    specr_cham = chameleon_rankability()
    specr_pokemon = pokemon_rankability()

    dat['spec_r_mean_tennis'] = spec_r_mean_tennis
    dat['spec_r_std_tennis'] = spec_r_std_tennis
    dat['spec_r_mean_web'] = spec_r_mean_web
    dat['spec_r_std_web'] = spec_r_std_web
    dat['specr_cham'] = specr_cham
    dat['specr_pokemon'] = specr_pokemon
    for k,v in dat.items():
        dat[k]=round(v,3)
    print(dat)
    pickle.dump(dat,
                open( f'spec_r.pickle',
                     "wb"))

    #GET SPECR NUMBER, IF IT'S SMALL WE IN BUSINESS, SHOULD BE (0,1)

