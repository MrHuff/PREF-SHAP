import pandas as pd
import copy
import numpy as np
import tqdm
import os
import pickle
# 'bpid',
# 'uniquesessionid',
# 'trans_date',
# 'click',
# 'view',

categorical_cols_old=[
                   'is_seasonless',
                   'is_carried_over',
                   'is_running_item',
                   'product_type_name',
                   'product_group_name',
                   'graphical_appearance_name',
                   'colour_name',
                    'assortment_mix_name',
                   'licence_company_name',
                   'section_name',
                   'composition',
                   'garment_group_name',
                   ]

item_cols=[
                   'is_seasonless',
                   'is_carried_over',
                   'product_group_name',
                   'graphical_appearance_name',
                    'assortment_mix_name',
                   'garment_group_name',
                   ]

u_cols =['year_of_birth','gender_code']

def extract_winners(losers,winners):
    left,right=[],[]
    y=[]
    for w in winners:
        for l in losers:
            c=np.random.choice([0,1])
            if c==0: #winner goes left
                left.append(w)
                right.append(l)
                y.append(-1)
            else: #winner goes right
                left.append(l)
                right.append(w)
                y.append(1)
    left_chunk=pd.concat(left,axis=1).transpose()
    right_chunk=pd.concat(right,axis=1).transpose()
    Y = np.array(y)
    return left_chunk,right_chunk,Y

def duelling_data(df):
    sess_id = df['uniquesessionid'].tolist()
    clicks = df['click'].tolist()
    init_list_losers = []
    init_list_winners = []
    big_Y=[]
    big_left=[]
    big_right=[]
    for i in tqdm.tqdm(range(len(sess_id) - 1)):
        cur = sess_id[i]
        next = sess_id[i + 1]
        cl = float(clicks[i])
        if cl==1:
            init_list_winners.append(df.iloc[i])
        else:
            init_list_losers.append(df.iloc[i])
        #
        # init_list.append(float(clicks[i]))
        if not cur == next:
            if len(init_list_winners) > 1 and len(init_list_losers) > 1:
                left_chunk, right_chunk, Y = extract_winners(init_list_losers, init_list_winners)
                big_left.append(left_chunk)
                big_right.append(right_chunk)
                big_Y.append(Y)
            init_list_losers = []
            init_list_winners = []
    left=pd.concat(big_left,axis=0)
    right=pd.concat(big_right,axis=0)
    final_Y=pd.Series(np.concatenate(big_Y,axis=0))

    return left,right,final_Y


def parse_correct_age(x):
    if x is None:
        return np.nan
    else:
        p = int(x)
        if p>1950 and p<2010:
            return p
        else:
            np.nan

if __name__ == '__main__':
    df = pd.read_parquet('pref_user_2.parquet')
    df = df.sort_values(by=['bpid', 'uniquesessionid'])
    # df = df.iloc[0:10000,:]
    n_unique  = df[item_cols].nunique().tolist()
    print(n_unique)
    df = pd.get_dummies(df,columns=item_cols)
    if not os.path.exists('website_data_user'):
        os.makedirs('website_data_user')

    if not os.path.exists('website_data_user/l_processed.npy'):
        l, r, y = duelling_data(df)
        l=l.drop(['article_id',
 'bpid',
 'uniquesessionid',
 'trans_date',
 'click',
 'view',
 'is_running_item',
 'product_type_name',
 'colour_name',
 'colour_group_name',
 'licence_company_name',
 'section_name',
 'division_name',
 'composition',
                  'city'
                  ],axis=1)
        r=r.drop(['article_id',
 'bpid',
 'uniquesessionid',
 'trans_date',
 'click',
 'view',
 'is_running_item',
 'product_type_name',
 'colour_name',
 'colour_group_name',
 'licence_company_name',
 'section_name',
 'division_name',
 'composition',
                  'city'
                  ],axis=1)
        u= pd.get_dummies(l[u_cols]).values
        l=l.drop(columns=u_cols,axis=1)
        r=r.drop(columns=u_cols,axis=1)
        l= l.values
        r= r.values
        with open('website_data_user/u.npy', 'wb') as f:
            np.save(f, u)
        with open('website_data_user/l_processed.npy', 'wb') as f:
            np.save(f, l)
        with open('website_data_user/r_processed.npy', 'wb') as f:
            np.save(f, r)
        df = pd.read_parquet('pref_user_2.parquet')
        S = df.drop_duplicates(subset=['article_id'])
        S = pd.get_dummies(S[item_cols]).values

        with open('website_data_user/S.npy', 'wb') as f:
            np.save(f, S)

        df = pd.read_parquet('pref_user_2.parquet')
        df['year_of_birth'] = df['year_of_birth'].apply(lambda x: parse_correct_age(x))
        df['year_of_birth'].fillna((df['year_of_birth'].mean()), inplace=True)
        df['gender_code'].fillna((3), inplace=True)
        S_u = df.drop_duplicates(subset=['bpid'])
        S_u = S_u[['year_of_birth','gender_code']]
        S_u = pd.get_dummies(S_u,'gender_code').values
        with open('website_data_user/S_u.npy', 'wb') as f:
            np.save(f, S_u)
        with open('website_data_user/y.npy', 'wb') as f:
            np.save(f, y.values)




        # l, r, y = duelling_data(df)
        # l.to_csv('website_data_user/left.csv')
        # r.to_csv('website_data_user/right.csv')
        # y.to_csv('website_data_user/y.csv')

    # cat_df = pd.get_dummies(df, columns=categorical_cols)








