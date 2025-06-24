import json

with open('data/samples_final_nlas_eng.json') as filehandle:
    json_data = json.load(filehandle)
data = {'train': [], 'dev': [], 'test': []}

dict={'0':[], '1':[], '2':[], '3':[], '4':[], '5':[], '6':[], '7':[], '8':[], '9':[], '10':[], '11':[], '12':[],
      '13':[], '14':[], '15':[], '16':[], '17':[], '18':[], '19':[]}

for argument in json_data['samples']:
    dict[str(argument[1])].append(argument[0])

for key in dict:
    tr = int(0.8*len(dict[key]))
    de = tr+int(0.1*len(dict[key]))
    train = dict[key][:tr]
    dev = dict[key][tr:de]
    test = dict[key][de:]
    print(len(train), len(dev), len(test))
    print(len(dict[key]))
    for spl in train:
        data['train'].append([spl, int(key)])
    for spl in dev:
        data['dev'].append([spl, int(key)])
    for spl in test:
        data['test'].append([spl, int(key)])
    print(len(data['train']), len(data['dev']), len(data['test']))

with open('data/eng/experiment_nlas_eng.json', 'w') as fp:
    json.dump(data, fp)