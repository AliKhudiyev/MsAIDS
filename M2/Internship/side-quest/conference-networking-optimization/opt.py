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
        # print(prefs)
        prefs[-1] = prefs[-1][:-1]

        attendees.append(line[0])
        tmp_weights = [-1]*n_spons

        for j, pref in enumerate(prefs):
            spon_index = sponsors.index(pref)
            tmp_weights[spon_index] = pref_weight(j, len(prefs), 1, 0.1)

        weights.append(tmp_weights)

# print('sponsors:', sponsors)
# print('attendees:', attendees)
# print(np.array(weights))


def weight(attendee, sponsor):
    global attendees, sponsors, slots, weights
    return weights[attendees.index(attendee)][sponsors.index(sponsor)]
    
'''
@sponsors    - list of sponsors, e.g., ['Matt Murdock', 'King Pin']
@attendees   - list of attendees, e.g., ['Frank Castle', 'Peter Parker', 'Marry Jane']
@slots       - list of slots, i.e., [0, 1, 2, ..., 11]
@weights     - preference weight matrix
'''
def create_model():
    global attendees, sponsors, slots, weights, schedule
    model = cp_model.CpModel()

    print('defining problem variables')
    # definining scheduling problem variables
    for attendee in attendees:
        for sponsor in sponsors:
            for slot in slots:
                schedule[(attendee, sponsor, slot)] = model.NewBoolVar(f'{attendee} meets {sponsor} @{slot}')

    print('ok\ndefining one-to-one constraint...')
    # defining constraints
    for slot in slots:
        for attendee in attendees:
            # s.t.: each attendee in a given slot has to speak to at most one sponsor
            model.AddAtMostOne(schedule[(attendee, sponsor, slot)] for sponsor in sponsors)
        for sponsor in sponsors:
            # s.t.: each sponsor in a given slot has to speak to at most one attendee
            model.AddAtMostOne(schedule[(attendee, sponsor, slot)] for attendee in attendees)

    print('ok\ndefining preferential election constraint...')
    # s.t.: an attendee can only meet any sponsor from his/her preferences
    for i, attendee in enumerate(attendees):
        tmp = []
        for j, sponsor in enumerate(sponsors):
            if weights[i][j] == -1:
                for slot in slots:
                    tmp.append(schedule[(attendee, sponsor, slot)])
        model.Add(sum(tmp) == 0)

    print('ok\ndefining service quality constraint...')
    satisfacts = [[]] * len(attendees)
    for i, attendee in enumerate(attendees):
        if i and i%(len(attendees)//10) == 0:
            print('+10%')
        for sponsor in sponsors:
            for slot in slots:
                # satisfacts[i].append(weight(attendees[i], sponsor) * schedule[(attendees[i], sponsor, slot)])
                satisfacts[i].append(schedule[(attendees[i], sponsor, slot)])
    print('satisfacts are ready...')
        
    # s.t.: previously registered attendee has to be served at least as good as the next registered ones
    for i in range(len(attendees)-1):
        satisfacts2 = []
        n_pref1 = 1 # np.count_nonzero(np.array(weights[i]) == 1)
        n_pref2 = 1 # np.count_nonzero(np.array(weights[i+1]) == 1)
        for sponsor in sponsors:
            for slot in slots:
                satisfacts2.append(schedule[(attendees[i], sponsor, slot)]*n_pref2 - schedule[(attendees[i+1], sponsor, slot)]*n_pref1)
        model.Add(sum(satisfacts2) >= 0)
        if i % (len(attendees)//10) == 0 and i:
            print('+10%')
        # model.Add(sum(satisfacts[i]) >= sum(satisfacts[i+1]))

            
    print('defining uniqueness constraint...')
    # s.t.: each attendee can meet a sponsor only once
    for attendee in attendees:
        for sponsor in sponsors:
            model.AddAtMostOne(schedule[(attendee, sponsor, slot)] for slot in slots)
    print('ok\nall constraints have been defined... ok')

    # defining  objective function
    satisfactions = []
    for i, attendee in enumerate(attendees):
        for j, sponsor in enumerate(sponsors):
            if weights[i][j] == 1:
                for slot in slots:
                    satisfactions.append(schedule[(attendee, sponsor, slot)]) # weight(attendee, sponsor) * 

    model.Maximize(sum(satisfactions))
    print('objective function has been defined... ok')
    
    return model


if not show_stats:
    print('creating model...')
    model = create_model()
    print('done')
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 8
    print('solving...')
    status = solver.Solve(model)
    print('done')

    f = open(output_path, 'w')
    f.write('slot, attendee, sponsor')
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print('optimal' if status == cp_model.OPTIMAL else 'feasible')
        print(f'Total cost = {solver.ObjectiveValue()}\n')
        for slot in slots:
            for attendee in attendees:
                for sponsor in sponsors:
                    if solver.BooleanValue(schedule[attendee, sponsor, slot]):
                        f.write(f'\n{slot}, {attendee}, {sponsor}')
    else:
        print('No solution found.')
    f.close()

'''
= = = = = = = = = = = = = Statistics = = = = = = = = = = = = = 
'''

import pandas as pd
import matplotlib.pyplot as plt


if show_stats:
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    axs[0,0].set_title('satisfaction rates')
    axs[0,1].set_title('schedule')
    axs[1,0].set_title('sponsor activity')
    axs[1,1].set_title('slot activity')

    axs[0,0].set_xlabel('attendee')
    axs[0,0].set_ylabel('satisfaction')
    axs[0,1].set_xlabel('attendee')
    axs[0,1].set_ylabel('#sponsor')
    axs[1,0].set_xlabel('sponsor')
    axs[1,0].set_ylabel('activity(%)')
    axs[1,1].set_xlabel('slot')
    axs[1,1].set_ylabel('activity(%)')

    axs[1,0].set_ylim(0, 1)
    axs[1,1].set_ylim(0, 1)

    weights = np.array(weights)
    df = pd.read_csv(output_path, skipinitialspace=True)

    satisfaction_rates = np.zeros((len(attendees),))
    schedule_counts = np.zeros((len(attendees),))
    for i, attendee in enumerate(attendees):
        n_scheduled = df[df['attendee'] == attendee].shape[0]
        n_pref = weights[np.where(weights[i,:] == 1)].shape[0]
        satisfaction_rates[i] = n_scheduled / n_pref
        schedule_counts[i] = n_scheduled
    axs[0,0].plot(satisfaction_rates)
    axs[0,1].plot(schedule_counts)

    sponsor_activity = np.zeros((len(sponsors),))
    slot_activity = np.zeros((len(slots),))
    for slot in slots:
        for i, sponsor in enumerate(sponsors):
            if df[(df['slot'] == slot) & (df['sponsor'] == sponsor)].shape[0]:
                sponsor_activity[i] += 1/len(slots)
                slot_activity[slot] += 1/len(sponsors)
    axs[1,0].bar(range(len(sponsors)), sponsor_activity)
    axs[1,0].plot([0, len(sponsors)-1], [np.mean(sponsor_activity), np.mean(sponsor_activity)], c='orange')
    axs[1,1].bar(slots, slot_activity)
    axs[1,1].plot([0, len(slots)-1], [np.mean(slot_activity), np.mean(slot_activity)], c='orange')
    # print(sponsor_activity)

    print('Mean slot activity:', np.mean(slot_activity))
    print('Std slot activity:', np.std(slot_activity))
    print('Mean sponsor activity:', np.mean(sponsor_activity))
    print('Std sponsor activity:', np.std(sponsor_activity))
    print('Attendees unsatisfied(0%):', np.count_nonzero(np.where(schedule_counts == 0)))

    plt.tight_layout()
    plt.show()
