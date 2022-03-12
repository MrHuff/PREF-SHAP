import copy

import pandas as pd
import numpy as np
from pref_shap.rankability_test import specR,edgeR
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
    for i in range(len(sess_id)-1):
        cur = sess_id[i]
        next = sess_id[i+1]
        init_list.append(float(clicks[i]))
        if not cur==next:
            if len(init_list)>1 and sum(init_list)>0:
                list_of_alist.append(copy.deepcopy(init_list))
            init_list=[]
    return list_of_alist


if __name__ == '__main__':
    df = pd.read_parquet('pref_2.parquet')
    df=df.sort_values(by=['bpid', 'uniquesessionid'])
    ls=get_list_of_alist(df)
    spec_r_list =[]
    for a_list in ls:
        spec_r_val=get_spec_r(a_list)
        spec_r_list.append(spec_r_val)
    data = np.array(spec_r_list)
    print(data.mean(0))
    print(data.std(0))



    #GET SPECR NUMBER, IF IT'S SMALL WE IN BUSINESS, SHOULD BE (0,1)

