import json


def flatten(l):
    return [item for sublist in l for item in sublist]


with open('data/samples_topic_nlas_eng.json') as filehandle:
    json_data = json.load(filehandle)
data = {'train': [], 'dev': [], 'test': []}

k = json_data.keys()
keys = []
for key in k:
    keys.append(key)
print(keys)

# 1st fold 10-20-30-40 45 50
fold1 = {'train': [], 'dev': [], 'test': []}
for i in range(40):
    fold1['train'].append(json_data[keys[i]])
for i in range(40, 45):
    fold1['dev'].append(json_data[keys[i]])
for i in range(45, 50):
    fold1['test'].append(json_data[keys[i]])

fold1['train'] = flatten(fold1['train'])
fold1['dev'] = flatten(fold1['dev'])
fold1['test'] = flatten(fold1['test'])

# 2nd fold 10-20-30-50 35 40
fold2 = {'train': [], 'dev': [], 'test': []}
for i in range(30):
    fold2['train'].append(json_data[keys[i]])
for i in range(30, 35):
    fold2['dev'].append(json_data[keys[i]])
for i in range(35, 40):
    fold2['test'].append(json_data[keys[i]])
for i in range(40, 50):
    fold2['train'].append(json_data[keys[i]])

fold2['train'] = flatten(fold2['train'])
fold2['dev'] = flatten(fold2['dev'])
fold2['test'] = flatten(fold2['test'])

# 3rd fold 10-20-50-40 25 30
fold3 = {'train': [], 'dev': [], 'test': []}
for i in range(20):
    fold3['train'].append(json_data[keys[i]])
for i in range(20, 25):
    fold3['dev'].append(json_data[keys[i]])
for i in range(25, 30):
    fold3['test'].append(json_data[keys[i]])
for i in range(30, 50):
    fold3['train'].append(json_data[keys[i]])

fold3['train'] = flatten(fold3['train'])
fold3['dev'] = flatten(fold3['dev'])
fold3['test'] = flatten(fold3['test'])

# 4th fold 10-50-30-40 15 20
fold4 = {'train': [], 'dev': [], 'test': []}
for i in range(10):
    fold4['train'].append(json_data[keys[i]])
for i in range(10, 15):
    fold4['dev'].append(json_data[keys[i]])
for i in range(15, 20):
    fold4['test'].append(json_data[keys[i]])
for i in range(20, 50):
    fold4['train'].append(json_data[keys[i]])

fold4['train'] = flatten(fold4['train'])
fold4['dev'] = flatten(fold4['dev'])
fold4['test'] = flatten(fold4['test'])


# 5th fold 50-20-30-40 5 10
fold5 = {'train': [], 'dev': [], 'test': []}
for i in range(5):
    fold5['dev'].append(json_data[keys[i]])
for i in range(5, 10):
    fold5['test'].append(json_data[keys[i]])
for i in range(10, 50):
    fold5['train'].append(json_data[keys[i]])

fold5['train'] = flatten(fold5['train'])
fold5['dev'] = flatten(fold5['dev'])
fold5['test'] = flatten(fold5['test'])

with open('data/eng/fold1_nlas_eng.json', 'w') as fp:
    json.dump(fold1, fp)

with open('data/eng/fold2_nlas_eng.json', 'w') as fp:
    json.dump(fold2, fp)

with open('data/eng/fold3_nlas_eng.json', 'w') as fp:
    json.dump(fold3, fp)

with open('data/eng/fold4_nlas_eng.json', 'w') as fp:
    json.dump(fold4, fp)

with open('data/eng/fold5_nlas_eng.json', 'w') as fp:
    json.dump(fold5, fp)