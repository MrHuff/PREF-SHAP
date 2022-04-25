import pandas as pd
import copy
import numpy as np
import tqdm
import os
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

categorical_cols=[
                   'is_seasonless',
                   'is_carried_over',
                   'product_group_name',
                   'graphical_appearance_name',
                    'assortment_mix_name',
                   'garment_group_name',
                   ]



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

if __name__ == '__main__':
    df = pd.read_parquet('pref_3.parquet')
    df = df.sort_values(by=['bpid', 'uniquesessionid'])
    n_unique  = df[categorical_cols].nunique().tolist()
    print(n_unique)
    df = pd.get_dummies(df,columns=categorical_cols)
    if not os.path.exists('website_data'):
        os.makedirs('website_data')

    if not os.path.exists('website_data/left.csv'):
        l, r, y = duelling_data(df)
        l=l.drop(['bpid',
                  'article_id',
        'uniquesessionid',
        'trans_date',
        'click',
        'view',
                  ],axis=1)
        r=r.drop(['bpid',
                  'article_id',
                  'uniquesessionid',
        'trans_date',
        'click',
        'view',
                  ],axis=1)

        l.to_csv('website_data/left.csv')
        r.to_csv('website_data/right.csv')
        with open('website_data/y.npy', 'wb') as f:
            np.save(f, y.values)

    if not os.path.exists('website_data/left_processed.npy'):
        l= pd.read_csv('website_data/left.csv',index_col=0)
        r= pd.read_csv('website_data/right.csv',index_col=0)
        with open('website_data/l_processed.npy', 'wb') as f:
            np.save(f, l)
        with open('website_data/r_processed.npy', 'wb') as f:
            np.save(f, r)
        df = pd.read_parquet('pref_3.parquet')
        S = df.drop_duplicates(subset=['article_id'])
        S = pd.get_dummies(S[categorical_cols]).values
        with open('website_data/S.npy', 'wb') as f:
            np.save(f, S)


        # l, r, y = duelling_data(df)
        # l.to_csv('website_data/left.csv')
        # r.to_csv('website_data/right.csv')
        # y.to_csv('website_data/y.csv')

    # cat_df = pd.get_dummies(df, columns=categorical_cols)








