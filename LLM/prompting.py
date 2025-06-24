import ollama
import csv
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

FAMILIES = 0
#MODEL = 'qwen2.5' #7b
#MODEL = 'llama3.1' #8b
MODEL = 'llama3.3' #70b

#TASK = 'Z-Sh'
TASK = 'Fw-sh-txt'
#TASK = 'Fw-sh-dial'

if TASK == 'Z-Sh':
    if FAMILIES == 0:
        DATA = 'zero-shot.txt'
        LABELS = ['Direct Ad Hominem', 'Inconsistent Commitment', 'Cause To Effect', 'Established Rule', 'Verbal Classification', 'Analogy', 'Example', 'Precedent', 'Best Explanation', 'Ignorance', 'Sign', 'Popular Opinion', 'Popular Practice', 'Expert Opinion', 'Position To Know', 'Witness Testimony', 'Consequences', 'Practical Reasoning', 'Sunk Costs', 'Threat', 'Waste', 'Slippery Slope']
    else:
        DATA = 'zero-shot-families.txt'
        LABELS = ['Ad Hominem Arguments', 'Arguments Based on Cases', 'Defeasible Rule-based Arguments', 'Discovery Arguments', 'Popular Acceptance Arguments', 'Position to Know Arguments', 'Practical Reasoning Arguments', 'Chained Arguments with Rules and Cases']

elif TASK == 'Fw-sh-txt':
    if FAMILIES == 0:
        DATA = 'few-shot-textbook.txt'
        LABELS = ['Direct Ad Hominem', 'Inconsistent Commitment', 'Cause To Effect', 'Established Rule', 'Verbal Classification', 'Analogy', 'Example', 'Precedent', 'Best Explanation', 'Ignorance', 'Sign', 'Popular Opinion', 'Popular Practice', 'Expert Opinion', 'Position To Know', 'Witness Testimony', 'Sunk Costs', 'Threat', 'Waste', 'Slippery Slope']
    else:
        DATA = 'few-shot-textbook-families.txt'
        LABELS = ['Ad Hominem Arguments', 'Arguments Based on Cases', 'Defeasible Rule-based Arguments',
                  'Discovery Arguments', 'Popular Acceptance Arguments', 'Position to Know Arguments',
                  'Practical Reasoning Arguments', 'Chained Arguments with Rules and Cases']

else:
    if FAMILIES == 0:
        DATA = 'few-shot-dialogue.txt'
        LABELS = ['Direct Ad Hominem', 'Inconsistent Commitment', 'Cause To Effect', 'Established Rule', 'Verbal Classification', 'Analogy', 'Example', 'Precedent', 'Best Explanation', 'Ignorance', 'Sign', 'Popular Opinion', 'Popular Practice', 'Expert Opinion', 'Position To Know', 'Witness Testimony', 'Consequences', 'Practical Reasoning', 'Sunk Costs', 'Threat', 'Waste', 'Slippery Slope']

    else:
        DATA = 'few-shot-dialogue-families.txt'
        LABELS = ['Ad Hominem Arguments', 'Arguments Based on Cases', 'Defeasible Rule-based Arguments',
                  'Discovery Arguments', 'Popular Acceptance Arguments', 'Position to Know Arguments',
                  'Practical Reasoning Arguments', 'Chained Arguments with Rules and Cases']

with open('18March2021.csv', newline='') as f:
    reader = csv.reader(f, delimiter=';')
    arg_list = list(reader)

truth = []
predict = []
full_response = []

for argument in arg_list[1:]:
    if 'Default Inference' in argument[3]:
        continue

    if FAMILIES == 1:
        if argument[3] == 'Direct Ad Hominem' or argument[3] == 'Inconsistent Commitment' or argument[3] == 'Allegation Of Bias':
            truth.append('Ad Hominem Arguments'.lower())
        elif argument[3] == 'Cause To Effect' or argument[3] == 'Established Rule' or argument[3] == 'Verbal Classification':
            truth.append('Arguments Based on Cases'.lower())
        elif argument[3] == 'Analogy' or argument[3] == 'Example' or argument[3] == 'Precedent':
            truth.append('Defeasible Rule-based Arguments'.lower())
        elif argument[3] == 'Best Explanation' or argument[3] == 'Ignorance' or argument[3] == 'Sign' or argument[3] == 'Random Sample To Population':
            truth.append('Discovery Arguments'.lower())
        elif argument[3] == 'Popular Opinion' or argument[3] == 'Popular Practice':
            truth.append('Popular Acceptance Arguments'.lower())
        elif argument[3] == 'Expert Opinion' or argument[3] == 'Position To Know' or argument[3] == 'Witness Testimony':
            truth.append('Position to Know Arguments'.lower())
        elif argument[3] == 'Consequences' or argument[3] == 'Practical Reasoning' or argument[3] == 'Sunk Costs' or argument[3] == 'Threat' or argument[3] == 'Waste':
            truth.append('Practical Reasoning Arguments'.lower())
        elif argument[3] == 'Slippery Slope':
            truth.append('Chained Arguments with Rules and Cases'.lower())
        else:
            print(argument[3].lower())
            exit()
    else:
        truth.append(argument[3].lower())

    arg = argument[0]
    print(truth[-1])

    with open(DATA, 'r') as file:
        prompt = file.read()#.replace('\n', '')

    final_prompt = prompt + arg
    #print(final_prompt)

    response = ollama.generate(model=MODEL, prompt=final_prompt)

    full_response.append(response['response'])

    #print(response['response'])

    res = response['response'].split('**')[1]
    predict.append(res.lower())

    print(predict[-1])
    print('----')


print(truth)
print(predict)

x = precision_recall_fscore_support(truth, predict, average='macro', labels=[label.lower() for label in LABELS])

print(x)
