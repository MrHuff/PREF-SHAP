import numpy as np
import torch
from sklearn.decomposition import PCA,KernelPCA

def cartesian_product(*arrays):
    ndim = len(arrays)
    return (np.stack(np.meshgrid(*arrays), axis=-1)
              .reshape(-1, ndim))
def one_hot_numpy(a):
    b = np.zeros((a.size, a.max()+1))
    b[np.arange(a.size), a] = 1
    return b
def deciding_features(left,right,D):
    a = np.zeros((D,D))
    iu = np.triu_indices(D,1)
    a[iu]=np.arange(1,D+1)
    a=a+np.transpose(a)
    dec_feat_index = a[left,right]
    return dec_feat_index.astype(int)
def generate_clusters(D=3,players=1000,n_matches=40000):
    hidden_cluster_list = list(range(D))
    hidden_states = np.random.choice(hidden_cluster_list,players)
    total_covs = np.cumsum(np.arange(D))[-1]+1
    x_cov = np.random.randn(players,total_covs)# * np.arange(1,total_covs+1)[np.newaxis,:]*0.1  #+ hidden_states[:,np.newaxis] * 1.
    # model = KernelPCA()
    # x_cov = model.fit_transform(x_cov)

    # x_cov = []
    # for i in range(total_covs):
    print(np.corrcoef(x_cov.T))
    a = np.arange(players)
    all_combinations=cartesian_product(a, a)
    all_matches = all_combinations[all_combinations[:,0]!=all_combinations[:,1]]
    matches = all_matches[np.random.choice(all_matches.shape[0], n_matches, replace=False)]
    left_hidden,left_cov = hidden_states[matches[:,0]],x_cov[matches[:,0]]
    right_hidden,right_cov = hidden_states[matches[:,1]],x_cov[matches[:,1]]
    d_feat = deciding_features(left_hidden,right_hidden,D)
    y=[]
    winners=[]
    losers=[]
    right_cov=np.concatenate([right_cov,one_hot_numpy(right_hidden)],axis=1)
    left_cov=np.concatenate([left_cov,one_hot_numpy(left_hidden)],axis=1)
    for i,d in enumerate(d_feat):
        y_tmp=np.sign(right_cov[i, d] - left_cov[i, d])
        # y.append(y_tmp)
        if y_tmp>0:
            winners.append(right_cov[i, :])
            losers.append(left_cov[i, :])
        else:
            winners.append(left_cov[i, :])
            losers.append(right_cov[i, :])

    y= np.ones(n_matches)# np.array(y)#[:,np.newaxis]
    losers =np.stack(losers,axis=0)
    winners =np.stack(winners,axis=0)
    S = np.concatenate([x_cov,one_hot_numpy(hidden_states)],axis=1)

    return losers,winners,y,S
if __name__ == '__main__':
    generate_clusters()











