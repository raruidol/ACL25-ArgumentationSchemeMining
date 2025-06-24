import csv
import json

'''

code = {'position to know': 0, 'expert opinion': 1, 'direct ad hominem': 2, 'inconsistent commitment': 3,
        'popular practice': 4, 'popular opinion': 5, 'analogy': 6, 'precedent': 7, 'example': 8,
        'established rule': 9, 'cause to effect': 10, 'verbal classification': 11, 'slippery slope': 12, 'sign': 13,
        'ignorance': 14, 'threat': 15, 'waste': 16, 'sunk costs': 17, 'witness testimony': 18, 'best explanation': 19, 
        'consequences': 21, 'practical reasoning': 22, 'allegation of bias': 23, 'random sample to population': 20}
'''

code = {'position to know': 0, 'expert opinion': 0, 'direct ad hominem': 1, 'inconsistent commitment': 1,
        'popular practice': 2, 'popular opinion': 2, 'analogy': 3, 'precedent': 3, 'example': 3,
        'established rule': 4, 'cause to effect': 4, 'verbal classification': 4, 'slippery slope': 5, 'sign': 6,
        'ignorance': 6, 'threat': 7, 'waste': 7, 'sunk costs': 7, 'witness testimony': 0, 'best explanation': 6,
        'consequences': 7, 'practical reasoning': 7, 'allegation of bias': 1, 'random sample to population': 6}


# 0 - Position to Know
# 1 - Ad Hominem Arguments
# 2 - Popular Acceptance
# 3 - Defeasible Rule-based Arguments
# 4 - Based on Cases
# 5 - Chained Arguments with rules and cases
# 6 - Discovery Arguments
# 7 - Practical Reasoning

data = {'train': [], 'test': []}

with open('data/qt-schemes/raw/19Aug2021.csv', newline='') as f:
    reader = csv.reader(f, delimiter=';')
    file_train1 = list(reader)

with open('data/qt-schemes/raw/2Sept2021.csv', newline='') as f:
    reader = csv.reader(f, delimiter=';')
    file_train2 = list(reader)

with open('data/qt-schemes/raw/10June2021.csv', newline='') as f:
    reader = csv.reader(f, delimiter=';')
    file_train3 = list(reader)

with open('data/qt-schemes/raw/14Oct2021.csv', newline='') as f:
    reader = csv.reader(f, delimiter=';')
    file_train4 = list(reader)

with open('data/qt-schemes/raw/18March2021.csv', newline='') as f:
    reader = csv.reader(f, delimiter=';')
    file_test = list(reader)

for arg in file_train1[1:]:
    try:
        sample = [arg[0], code[arg[3].lower()]]
        data['train'].append(sample)
    except:
        print()

for arg in file_train2[1:]:
    try:
        sample = [arg[0], code[arg[3].lower()]]
        data['train'].append(sample)
    except:
        print()

for arg in file_train3[1:]:
    try:
        sample = [arg[0], code[arg[3].lower()]]
        data['train'].append(sample)
    except:
        print()

for arg in file_train4[1:]:
    try:
        sample = [arg[0], code[arg[3].lower()]]
        data['train'].append(sample)
    except:
        print()

for arg in file_test[1:]:
    try:
        sample = [arg[0], code[arg[3].lower()]]
        data['test'].append(sample)
    except:
        print()

print(len(data['train']), len(data['test']))

with open('data/qt-schemes/qt-schemes-families.json', 'w') as fp:
    json.dump(data, fp)
