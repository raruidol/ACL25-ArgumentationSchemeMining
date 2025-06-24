import json
from itertools import combinations
import ast

with open('data/nlas_multi.json') as filehandle:
    json_data = json.load(filehandle)

with open('data/nlas_extra.json') as filehandle:
    json_data_extra = json.load(filehandle)

argument_dataset = []

for a_id in json_data['eng']:
    if json_data['eng'][a_id]['label'] == 'yes':
        argument_dataset.append(json_data['eng'][a_id])

for b_id in json_data_extra['eng']:
    if json_data_extra['eng'][b_id]['label'] == 'yes':
        argument_dataset.append(json_data_extra['eng'][b_id])

'''
code = {'position to know': 0, 'expert opinion': 1, 'direct ad hominem': 2, 'inconsistent commitment': 3,
        'popular practice': 4, 'popular opinion': 5, 'analogy': 6, 'precedent': 7, 'example': 8,
        'established rule': 9, 'cause to effect': 10, 'verbal classification': 11, 'slippery slope': 12, 'sign': 13,
        'ignorance': 14, 'threat': 15, 'waste': 16, 'sunk costs': 17, 'witness testimony': 18, 'best explanation': 19}
'''

code = {'position to know': 0, 'expert opinion': 0, 'direct ad hominem': 1, 'inconsistent commitment': 1,
        'popular practice': 2, 'popular opinion': 2, 'analogy': 3, 'precedent': 3, 'example': 3,
        'established rule': 4, 'cause to effect': 4, 'verbal classification': 4, 'slippery slope': 5, 'sign': 6,
        'ignorance': 6, 'threat': 7, 'waste': 7, 'sunk costs': 7, 'witness testimony': 0, 'best explanation': 6}

# 0 - Position to Know
# 1 - Ad Hominem Arguments
# 2 - Popular Acceptance
# 3 - Defeasible Rule-based Arguments
# 4 - Based on Cases
# 5 - Chained Arguments with rules and cases
# 6 - Discovery Arguments
# 7 - Practical Reasoning

data = {'train': [], 'dev': []}
dev = 356

for argument_data in argument_dataset:
    if dev > 0:
        dev -= 1
        label = code[argument_data['argumentation scheme']]
        # print(label)
        if isinstance(argument_data['argument'], dict):
            argument = argument_data['argument']
        else:
            argument = ast.literal_eval(argument_data['argument'])
            # argument = json.loads(argument_data['argument'])

        text_units = []
        text = ''
        for argcomp in argument:
            text_units.append(argument[argcomp])
            text += argument[argcomp] + ' '
            if 'conclusion' not in argcomp:
                data['dev'].append([argument[argcomp], label])
        data['dev'].append([text, label])

        for combination in combinations(text_units, 2):
            data['dev'].append([combination[0] + ' ' + combination[1], label])

    else:
        label = code[argument_data['argumentation scheme']]
        # print(label)
        if isinstance(argument_data['argument'], dict):
            argument = argument_data['argument']
        else:
            argument = ast.literal_eval(argument_data['argument'])
            # argument = json.loads(argument_data['argument'])

        text_units = []
        text = ''
        for argcomp in argument:
            text_units.append(argument[argcomp])
            text += argument[argcomp]+' '
            if 'conclusion' not in argcomp:
                data['train'].append([argument[argcomp], label])
        data['train'].append([text, label])

        for combination in combinations(text_units, 2):
            data['train'].append([combination[0]+' '+combination[1], label])

print(len(data['train']))
print(len(data['dev']))

class_distr_train = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0}
class_distr_dev = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0}

for sample in data['train']:
    class_distr_train[sample[1]] += 1

for sample in data['dev']:
    class_distr_dev[sample[1]] += 1

print(class_distr_train)
print(class_distr_dev)

with open('data/eng/experiment_nlas_extended_grouped.json', 'w') as fp:
    json.dump(data, fp)
