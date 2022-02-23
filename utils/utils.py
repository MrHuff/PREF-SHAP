import pickle



def load_data(data_dir_load,files):
    with open(data_dir_load + '/' + files, 'rb') as handle:
        data_dict = pickle.load(handle)
    return data_dict

def save_data(data_dir_load,files,u,u_prime,y):
    with open(data_dir_load + '/' + files, 'rb') as handle:
        pickle.dump({ 'X':u,'X_prime':u_prime,'Y':y}, handle, protocol=pickle.HIGHEST_PROTOCOL)

