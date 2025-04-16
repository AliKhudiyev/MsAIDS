from ortools.sat.python import cp_model
import os, sys
import numpy as np


show_stats = False if len(sys.argv) < 4 else bool(sys.argv[3])
input_path = sys.argv[1]
output_path = sys.argv[2]

attendees = []
sponsors = []
slots = list(range(12))
weights = []
schedule = {}


def pref_weight(index, n_prefs, max_weight, min_weight):
    return 1


with open(input_path, 'r') as f:
    n_attend_spons = f.readline().split(' ')
    n_spons = int(n_attend_spons[0])
    n_attend = int(n_attend_spons[1])

    for i in range(n_spons):
        sponsors.append(f.readline()[:-1])

    for i in range(n_attend):
        line = f.readline().split(',')
        prefs = line[1:]
        prefs[-1] = prefs[-1][:-1]

        attendees.append(line[0])
        tmp_weights = [-1]*n_spons

        for j, pref in enumerate(prefs):
            spon_index = sponsors.index(pref)
            tmp_weights[spon_index] = pref_weight(j, len(prefs), 1, 0.1)

        weights.append(tmp_weights)

print('sponsors:', sponsors)
print('attendees:', attendees)
# print(np.array(weights))


def main():
    for slot in slots:
        for sponsor in sponsors:
            pass


if __name__ == '__main__':
    main()
