# Load Libraries
import os.path

from numpy import sign, count_nonzero, cos, sin, pi, ones, zeros, mean, shape, reshape, exp, sort, diag, eye, around, \
    linspace, dot, sqrt, argsort, allclose, bool, fill_diagonal, concatenate
from numpy import shape, savetxt, loadtxt, transpose, shape, reshape, concatenate
from numpy.random import rand, randn, randint
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def new_compute_upset1(C, r):
    C = np.triu(C)
    N_i, N_j = C.shape
    upset = 0
    for i in range(N_i):
        for j in range(i+1, N_j):
            if C[i, j] != 0:
                if np.sign(C[i, j])*np.sign(r[i] - r[j]) < 0:
                    upset += 1
                elif r[i] - r[j] == 0:
                    upset += .5
    total_entries = (C != 0).sum()
    return upset/total_entries


def FLIP(C, η=0.1):
    """
    Given a C matrix, randomly create noise to the problem.

    """
    n, p = C.shape
    flip_ls = np.random.choice([1, -1], n*p, p=(1-η, η))
    flip_ls = flip_ls.reshape(n, p)

    flip_mat = np.triu(flip_ls) + np.triu(flip_ls).T - \
        np.diag(np.diag(flip_ls))

    return C*flip_mat


def full_compute_upset(C, r):
    value1 = new_compute_upset1(C, r)
    value2 = new_compute_upset1(C, -r)

    return np.minimum(np.float64(value1), np.float64(value2))


def include_sparsity(C, sparsity=0.7):
    n, p = C.shape
    zero_ls = np.random.choice([0, 1], n*p, p=(sparsity, 1-sparsity))
    zero_ls = zero_ls.reshape(n, p)

    flip_mat = np.triu(zero_ls) + np.triu(zero_ls).T - \
        np.diag(np.diag(zero_ls))

    return C*flip_mat


def unseen_setup(algo_input, sparsity=0.7):
    C_train, choix_ls, C_test, features, K = algo_input

    full_C = C_train + C_test
    cut = np.int(np.round(full_C.shape[0] * 0.7))
    train_ind = np.random.choice(full_C.shape[0], cut, replace=False)
    test_ind = []
    for i in range(full_C.shape[0]):
        if i not in train_ind:
            test_ind.append(i)
    test_ind = np.array(test_ind)

    train_features = pd.DataFrame(features).iloc[train_ind, :]
    train_features = np.array(train_features)
    test_features = pd.DataFrame(features).iloc[test_ind, :]
    test_features = np.array(test_features)

    training = pd.DataFrame(full_C).iloc[train_ind, train_ind]
    training = np.array(training)
    testing = pd.DataFrame(full_C).iloc[test_ind, test_ind]
    testing = np.array(testing)

    K_train = np.array(pd.DataFrame(K).iloc[train_ind, train_ind])
    K_test = np.array(pd.DataFrame(K).iloc[test_ind, train_ind])

    # Need to work on choix ls too
    train_choix_ls = []
    for i in choix_ls:
        if i[0] in train_ind and i[1] in train_ind:
            train_choix_ls.append(i)

    pre_ls = np.unique(np.array(train_choix_ls).reshape(-1, 1))
    new_ls = np.arange(len(pre_ls))

    new_ls = pd.Series(new_ls)
    new_ls.index = pre_ls

    final_ls = []
    for i in train_choix_ls:
        final_ls.append((new_ls[i[0]], new_ls[i[1]]))

    if sparsity != 0:
        training = include_sparsity(training, sparsity)
        testing = include_sparsity(testing, sparsity)

    return training, testing, K_train, K_test, train_features, test_features, final_ls


def WorldHappiness(split=0.7, matern_length=4.0, noise=0.7):
    sensitive_factors = pd.read_csv("../data/world-happiness/Alsina_ls.csv")

    country_ls = []
    for country in sensitive_factors["Country"]:
        country_ls.append(country[1:])

    sensitive_factors["Country"] = country_ls

    happy = pd.read_csv("../data/world-happiness/2015.csv")

    subset_country_ls = []
    for i in range(happy.shape[0]):
        if happy["Country"][i] in country_ls:
            subset_country_ls.append(i)

    happy = happy.iloc[subset_country_ls, :]
    sensitive_factors = sensitive_factors.set_index("Country")
    sensitive_factors = sensitive_factors.loc[happy["Country"]]

    happy["Ethnic"] = sensitive_factors["Ethnic Fractionalization"].values
    happy["Linguistic"] = sensitive_factors["Linguistic Fractionalization"].values
    happy["Religious"] = sensitive_factors["Religious Fractionalization"].values

    # Create the pairwise comparison matrix first
    r = happy["Happiness Score"]
    r = r.values.reshape(-1, 1)

    C = r @ np.ones_like(r).T - np.ones_like(r)@r.T

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if C[i, j] != 0:
                if C[i, j] > 0:
                    C[i, j] = 1
                elif C[i, j] < 0:
                    C[i, j] = -1

    p = split
    n = C.shape[0]
    flips = np.random.binomial([1 for i in range(n*n)], p)
    flips = flips.reshape(n, n)

    switch_mat = np.triu(flips) + np.triu(flips).T - np.diag(np.diag(flips))

    train = C * switch_mat
    test = C * (-switch_mat + 1)

    features_col = ['Economy (GDP per Capita)', 'Family',
                    'Health (Life Expectancy)', 'Freedom',
                    'Trust (Government Corruption)',
                    'Generosity', "Ethnic"]

    features = happy[features_col]

    scaler = StandardScaler()
    features = scaler.fit_transform(features)



    choix_ls = []
    for i in range(n):
        for j in range(i, n):
            if train[i, j] == 1:
                choix_ls.append((i, j))
            elif train[i, j] == -1:
                choix_ls.append((j, i))

    # Add randomly generated noise
    train = FLIP(train, noise)

    return train, choix_ls, test, np.array(features)


def WorldHappiness_offset(split=0.7, matern_length=4.0, noise=0.3, sparsity=0.7):
    sensitive_factors = pd.read_csv("../data/world-happiness/Alsina_ls.csv")

    country_ls = []
    for country in sensitive_factors["Country"]:
        country_ls.append(country[1:])

    sensitive_factors["Country"] = country_ls

    happy = pd.read_csv("../data/world-happiness/2015.csv")

    subset_country_ls = []
    for i in range(happy.shape[0]):
        if happy["Country"][i] in country_ls:
            subset_country_ls.append(i)

    happy = happy.iloc[subset_country_ls, :]
    sensitive_factors = sensitive_factors.set_index("Country")
    sensitive_factors = sensitive_factors.loc[happy["Country"]]

    happy["Ethnic"] = sensitive_factors["Ethnic Fractionalization"].values
    happy["Linguistic"] = sensitive_factors["Linguistic Fractionalization"].values
    happy["Religious"] = sensitive_factors["Religious Fractionalization"].values

    # Create the pairwise comparison matrix first
    r = happy["Happiness Score"]
    r = r.values.reshape(-1, 1)

    C = r @ np.ones_like(r).T - np.ones_like(r)@r.T
    C = include_sparsity(C, sparsity)

    p = split
    n = C.shape[0]
    flips = np.random.binomial([1 for i in range(n*n)], p)
    flips = flips.reshape(n, n)

    switch_mat = np.triu(flips) + np.triu(flips).T - np.diag(np.diag(flips))

    train = C * switch_mat
    test = C * (-switch_mat + 1)

    features_col = ['Economy (GDP per Capita)', 'Family',
                    'Health (Life Expectancy)', 'Freedom',
                    'Trust (Government Corruption)',
                    'Generosity', "Ethnic"]

    features = happy[features_col]

    scaler = StandardScaler()
    features = scaler.fit_transform(features)


    choix_ls = []
    for i in range(n):
        for j in range(i, n):
            if train[i, j] == 1:
                choix_ls.append((i, j))
            elif train[i, j] == -1:
                choix_ls.append((j, i))

    # Add randomly generated noise
    train = FLIP(train, noise)

    return train, choix_ls, test, np.array(features)


def Insurance(split=0.7, matern_length=0.3, noise=.3):
    """
    Write a function that returns the necessary ingrediants
    to run my ranking algorithms
    """
    insurance = pd.read_csv("../data/Insurance/insurance.csv")

    scaler = StandardScaler()
    transformed_numerics = pd.DataFrame(scaler.fit_transform(np.array(insurance[["age","bmi"]])))
    del insurance["age"]
    del insurance["bmi"]
    transformed_numerics.columns = ["age", "bmi"]
    insurance = pd.concat([insurance, transformed_numerics], 1)

    # Obtaining the "matches" matrix
    cost = insurance.charges.values
    cost = cost.reshape(-1, 1)

    cost_diff = np.dot(cost, np.ones_like(cost).T) - np.dot(np.ones_like(cost), cost.T)

    binarise_diff = np.zeros_like(cost_diff)
    n = cost_diff.shape[0]
    for i in range(n):
        for j in range(n):
            if cost_diff[i, j] > 0:
                binarise_diff[i, j] = 1
            elif cost_diff[i, j] < 0:
                binarise_diff[i, j] = -1

    features_ls = ["bmi", "children", "smoker", "region", "age", "sex"]
    features = insurance[features_ls]
    features = pd.concat([features, pd.get_dummies(features["region"])], 1)
    features["sex"] = (features["sex"] == "male") + 0
    del features["region"]
    features["smoker"] = (features["smoker"] == "yes") + 0

    # Create sparsity
    p = split
    n = binarise_diff.shape[0]
    flips = np.random.binomial([1 for i in range(n*n)], p)
    flips = flips.reshape(n, n)

    switch_mat = np.triu(flips) + np.triu(flips).T - np.diag(np.diag(flips))

    train = binarise_diff * switch_mat
    test = binarise_diff * (-switch_mat + 1)

    # Create kernel matrix

    # choix ls
    choix_ls = []
    for i in range(n):
        for j in range(i, n):
            if train[i, j] == 1:
                choix_ls.append((i, j))
            elif train[i, j] == -1:
                choix_ls.append((j, i))

    # Add randomly generated noise
    train = FLIP(train, noise)

    return train, choix_ls, test, np.array(features)


def Insurance_offset(split=0.7, matern_length=0.3, noise=.3, sparsity=0.7):
    """
    Write a function that returns the necessary ingrediants
    to run my ranking algorithms
    """
    insurance = pd.read_csv("../data/Insurance/insurance.csv")

    scaler = StandardScaler()
    transformed_numerics = pd.DataFrame(scaler.fit_transform(np.array(insurance[["age","bmi"]])))
    del insurance["age"]
    del insurance["bmi"]
    transformed_numerics.columns = ["age", "bmi"]
    insurance = pd.concat([insurance, transformed_numerics], 1)

    # Obtaining the "matches" matrix
    cost = insurance.charges.values
    cost = cost.reshape(-1, 1)

    cost_diff = np.dot(cost, np.ones_like(cost).T) - np.dot(np.ones_like(cost), cost.T)

    cost_diff = include_sparsity(cost_diff, sparsity)

    features_ls = ["bmi", "children", "smoker", "region", "age", "sex"]
    features = insurance[features_ls]
    features = pd.concat([features, pd.get_dummies(features["region"])], 1)
    features["sex"] = (features["sex"] == "male") + 0
    del features["region"]
    features["smoker"] = (features["smoker"] == "yes") + 0

    # Create sparsity
    p = split
    n = cost_diff.shape[0]
    flips = np.random.binomial([1 for i in range(n*n)], p)
    flips = flips.reshape(n, n)

    switch_mat = np.triu(flips) + np.triu(flips).T - np.diag(np.diag(flips))

    train = cost_diff * switch_mat
    test = cost_diff * (-switch_mat + 1)

    # Create kernel matrix
    # choix ls
    choix_ls = []

    # Add randomly generated noise
    train = FLIP(train, noise)

    return train, choix_ls, test, np.array(features)


def Chameleon(split=0.7, matern_length=0.3):
    # Contest
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

    scaler = StandardScaler()
    predictors = scaler.fit_transform(np.float64(predictors))

    reference = pd.DataFrame([i for i in range(len(ind))], index=ind)

    choix_ls = []
    for row in range(contest.shape[0]):
        hold = contest.iloc[row, :]
        choix_ls.append(
            (reference.loc[hold["Winner"]].values[0], reference.loc[hold["Loser"]].values[0]))

    return B_train, choix_ls, predictors


def FlatLizard(split=0.7, matern_length=0.3):

    contest = pd.read_csv("./lizard_data/contest.csv")
    predictors = pd.read_csv(
        "./lizard_data/predictors.csv", index_col="Unnamed: 0")
    del contest["Unnamed: 0"]
    ind = predictors.index.values
    # Create binary matrix representing winning and losing
    b_train = pd.DataFrame(
        np.zeros((len(ind), len(ind))), index=ind, columns=ind)

    for row in range(contest.shape[0]):
        hold = contest.iloc[row, ]
        i, j = hold["winner"], hold["loser"]
        b_train.loc[i, j] = 1
        b_train.loc[j, i] = -1

    B_train = np.array(b_train)

    # fill in the missing values using mean
    for col in predictors.columns[:-1]:
        mean = np.mean(predictors[col])
        predictors[col] = predictors[col].fillna(mean)

    # even spread of "floater" and "resident" so we should not input missing values using mode.
    predictors["repro_resident"] = [1 if predictors["repro.tactic"]
                                    [row] == "resident" else 0 for row in range(predictors.shape[0])]
    predictors["repro_floater"] = [1 if predictors["repro.tactic"]
                                   [row] == "floater" else 0 for row in range(predictors.shape[0])]
    del predictors["repro.tactic"]
    del predictors["id"]

    scaler = StandardScaler()
    predictors = scaler.fit_transform(np.float64(predictors))

    reference = pd.DataFrame([i for i in range(len(ind))], index=ind)

    choix_ls = []
    for row in range(contest.shape[0]):
        hold = contest.iloc[row, :]
        choix_ls.append(
            (reference.loc[hold["winner"]].values[0], reference.loc[hold["loser"]].values[0]))

    return B_train, choix_ls, predictors


def NFL(split=0.7, matern_length=0.3):

    # Load data
    stats = pd.read_csv(
        "./NFL_Data/NFL2009/reg_season_teamstats.csv", index_col="Team")
    game = pd.read_csv("./NFL_Data/NFL2009/regular_season.csv")

    # Use the index of `stats` to create a binary matrix
    ind = stats.index.values
    # there are 32 teams
    B_train = pd.DataFrame(np.zeros((32, 32)), index=ind, columns=ind)

    # input the matrix B
    for match in range(game.shape[0]):
        hold = game.iloc[match, ]
        i, j = hold["Winner/tie"], hold["Loser/tie"]
        B_train.loc[i, j] = 1
        B_train.loc[j, i] = -1

    features = stats.iloc[:, 0:14]
    scaler = StandardScaler()
    features_ = scaler.fit_transform(np.float64(features))


    reference = pd.DataFrame([ind, [i for i in range(len(ind))]]).T
    reference.index = reference.iloc[:, 0]
    del reference[0]

    choix_ls = []
    for rnd in range(game.shape[0]):
        hold = game.iloc[rnd, :]
        choix_ls.append(
            (reference.loc[hold["Winner/tie"]].values[0], reference.loc[hold["Loser/tie"]].values[0]))

    return np.array(B_train), choix_ls,  features_


def NFL_offset(split=0.7, matern_length=0.3):
    # Load data
    stats = pd.read_csv(
        "./NFL_Data/NFL2009/reg_season_teamstats.csv", index_col="Team")
    game = pd.read_csv("./NFL_Data/NFL2009/regular_season.csv")

    # Use the index of `stats` to create a binary matrix
    ind = stats.index.values
    # there are 32 teams
    B_train = pd.DataFrame(np.zeros((32, 32)), index=ind, columns=ind)


    # input the matrix B
    for match in range(game.shape[0]):
        hold = game.iloc[match, ]
        i, j = hold["Winner/tie"], hold["Loser/tie"]
        i_score, j_score = hold["PtsW"], hold["PtsL"]
        B_train.loc[i, j] = i_score - j_score
        B_train.loc[j, i] = j_score - i_score

    features = stats.iloc[:, 0:14]
    scaler = StandardScaler()
    features_ = scaler.fit_transform(np.float64(features))

    reference = pd.DataFrame([ind, [i for i in range(len(ind))]]).T
    reference.index = reference.iloc[:, 0]
    del reference[0]

    choix_ls = []
    for rnd in range(game.shape[0]):
        hold = game.iloc[rnd, :]
        choix_ls.append(
            (reference.loc[hold["Winner/tie"]].values[0], reference.loc[hold["Loser/tie"]].values[0]))

    return np.array(B_train), choix_ls, features_


def Pokemon(split=0.7, matern_length=0.3):
    stats = pd.read_csv("./Pokemon/pokemon.csv", index_col="#")
    contest_1 = pd.read_csv("./Pokemon/combats.csv")
    contest_2 = pd.read_csv("./Pokemon/tests.csv")
    contest = pd.concat([contest_1,contest_2],axis=0).reset_index()


    # Create type columns
    type_col = np.unique(list(stats["Type 2"].unique()) + list(stats["Type 1"].unique()))
    type_col = type_col[type_col != 'nan']
    for col in type_col:
        stats[col] = 0

    for i in stats.index:
        stats.loc[i, stats.loc[i, "Type 1"]] += 1
        if not pd.isna(stats.loc[i, "Type 2"]):
            stats.loc[i, stats.loc[i, "Type 2"]] += 1

    del stats["Type 1"]
    del stats["Type 2"]
    del stats["Generation"]
    name_ls = stats["Name"].values
    del stats["Name"]
    stats["Legendary"] += 0
    numeric = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']

    scaler = StandardScaler()
    stats[numeric] = scaler.fit_transform(np.float64(stats[numeric]))

    # Bradley Terry
    choix_ls = []
    for i in range(contest.shape[0]):
        hold = contest.iloc[i, :]
        if hold["First_pokemon"] == hold["Winner"]:
            choix_ls.append(
                (hold["First_pokemon"]-1, hold["Second_pokemon"]-1))
        else:
            choix_ls.append(
                (hold["Second_pokemon"]-1, hold["First_pokemon"]-1))

    return [], np.array(choix_ls), np.array(stats)


def skillfunction(x):
    return sin(3*pi*x)-1.5*x**2

def generate_matches(f, noisesd=1.0, dropoff_rate=0.3, verbose=True):
    n = shape(f)[0]
    e = ones((n, 1))
    C = f.dot(e.T)-e.dot(f.T) + noisesd*randn(n, n)
    fill_diagonal(C, 0)  # remove noise from the diagonal
    C = (C-C.T)/2  # make the matrix skew-symmetric again
    # some entries are not observed - zero them out
    mask = rand(n, n) < dropoff_rate

    Ctst = zeros((n, n))
    Ctst[mask] = C[mask]
    Ctst[(Ctst.T) != 0] = -Ctst.T[(Ctst.T) != 0]

    C[mask] = 0
    C[C.T == 0] = 0  # make the matrix skew-symmetric again

    if(verbose):
        print("observed matches: %d/%d" % (count_nonzero(C)/2, n*(n-1)/2))
        print("witheld matches (testing): %d/%d" %
              (count_nonzero(Ctst)/2, n*(n-1)/2))

    return C, Ctst

def reshape_data(full_choix_ls,predictors):
    left = []
    right = []
    y = []
    for a,b in full_choix_ls:
        a,b = int(a),int(b)
        winner_cov,looser_cov=predictors[a,:],predictors[b,:]
        direction = np.random.choice([0, 1])
        if direction==1: #winner goes left
            left.append(winner_cov)
            right.append(looser_cov)
            y.append(-1)
        else:#winner goes right
            left.append(looser_cov)
            right.append(winner_cov)
            y.append(1)
    Y = np.array(y)
    left = np.stack(left,axis=0)
    right = np.stack(right,axis=0)
    return left,right,Y

def reshape_data_wl(full_choix_ls,predictors):
    left = []
    right = []
    y = []
    for a,b in full_choix_ls:
        a,b = int(a),int(b)
        winner_cov,looser_cov=predictors[a,:],predictors[b,:]
        right.append(winner_cov)
        left.append(looser_cov)
        y.append(1)
    Y = np.array(y)
    left = np.stack(left,axis=0)
    right = np.stack(right,axis=0)
    return left,right,Y

def save_data(ds_name,choix_ls,predictors):
    l, r, y = reshape_data(choix_ls, predictors)
    if not os.path.exists(f'{ds_name}'):
        os.makedirs(ds_name)
    with open(f'{ds_name}/l_processed.npy', 'wb') as f:
        np.save(f, l)
    with open(f'{ds_name}/r_processed.npy', 'wb') as f:
        np.save(f, r)
    with open(f'{ds_name}/S.npy', 'wb') as f:
        np.save(f, predictors)
    with open(f'{ds_name}/y.npy', 'wb') as f:
        np.save(f, y)

def save_data_wl(ds_name,choix_ls,predictors):
    l, r, y = reshape_data_wl(choix_ls, predictors)
    if not os.path.exists(f'{ds_name}_wl'):
        os.makedirs(ds_name+'_wl')
    with open(f'{ds_name}_wl/l_processed.npy', 'wb') as f:
        np.save(f, l)
    with open(f'{ds_name}_wl/r_processed.npy', 'wb') as f:
        np.save(f, r)
    with open(f'{ds_name}_wl/S.npy', 'wb') as f:
        np.save(f, predictors)
    with open(f'{ds_name}_wl/y.npy', 'wb') as f:
        np.save(f, y)

if __name__ == '__main__':

    for func,ds_name in zip([Chameleon,Pokemon],['chameleon','pokemon']):
        B_train, choix_ls, predictors = func()
        save_data(ds_name,choix_ls,predictors)
        save_data_wl(ds_name,choix_ls,predictors)



    # print(B_train.shape)
    # print(predictors.shape)