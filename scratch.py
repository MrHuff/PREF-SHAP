import pickle

with open('dataset_summary.pickle', 'rb') as handle:
    a = pickle.load(handle)
with open('spec_r.pickle', 'rb') as handle:
    b = pickle.load(handle)

print(a)
print(b)