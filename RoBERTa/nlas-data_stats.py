import json
from itertools import combinations
import ast

with open('data/eng/experiment_nlas_eng.json') as filehandle:
    nlas = json.load(filehandle)

with open('data/eng/experiment_nlas_extended.json') as filehandle:
    nlas_ext = json.load(filehandle)

with open('data/eng/experiment_nlas_extended_grouped.json') as filehandle:
    nlas_fam = json.load(filehandle)

class_distr = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0}
class_distr_ext = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0}
class_distr_fam_tr = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0}
class_distr_fam_te = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0}
for split in nlas:
    for arg in nlas[split]:
        class_distr[arg[1]] += 1

for split in nlas_ext:
    for arg in nlas_ext[split]:
        class_distr_ext[arg[1]] += 1

for arg in nlas_fam['train']:
    class_distr_fam_tr[arg[1]] += 1

for arg in nlas_fam['dev']:
    class_distr_fam_te[arg[1]] += 1

print(class_distr)

print(class_distr_ext)

print(class_distr_fam_tr)
print(class_distr_fam_te)