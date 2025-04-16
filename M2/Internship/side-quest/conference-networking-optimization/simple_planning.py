# %% [markdown]
# # Demo: Assignment process for KGC Networking event
# 
# Vocabulary:
# 
# * Event: the full 90 minutes networking event
# * Session: the Event is broken down into one-on-one Sessions of 10 minutes
# * Round: One Round is a 10 minutes time comosed of Sessions

# %%
from random import sample, randint
import string
from pprint import pprint as pp
from collections import defaultdict
import json, sys

# %%
sponsors = [
        'RelationalAI',
        'Katana Graph',
        'Accenture',
        'Oracle',
        'Amazon',
        'Brighthive',
        'Cambridge Semantics',
        'data.world',
        'Datachemist',
        'Datastax',
        'Ontotext',
        'OriginTrail',
        'Semantic Web',
        'Stardog',
        'TerminusDB',
        'Franz',
        'Lymba',
        'Graph Aware',
        'Oxford Semantic',
        'Ontoforce',
        'SeMI Technologies',
        'MonteCarlo',
        'CapGemini',
        'Synaptica',
        'ExperoInc',
        'LARUS Business Automation',
        'Sponsor_0',
        'Sponsor_1',
        'Sponsor_2',
        'Sponsor_3'
    ]


# %%
len(sponsors)

# %%
with open('surnames.txt', 'r', encoding='utf8') as file:
    surnames = [name.strip() for name in file.readlines()]

# %%
len(surnames)

# %%
sample(surnames, 10)

# %%
nb_attendees = 360
nb_sponsors = len(sponsors)
nb_rounds = 12
# sponsors = sample(string.ascii_uppercase, nb_sponsors)
attendees = sample(surnames, nb_attendees)

# %%
attendee_dict = dict()

for attendee in attendees:
    nb_choices = randint(2, nb_rounds)
    attendee_dict[attendee] = sample(sponsors, nb_choices)

attendees_choices = [{'attendee_id': key, 'choices': value} for key, value in attendee_dict.items()]
attendees_choices.sort(reverse=False, key=lambda x: x['attendee_id'])

with open(sys.argv[1], 'w') as f:
    f.write(f'{nb_sponsors} {nb_attendees}\n')
    for sponsor in sponsors:
        f.write(f'{sponsor}\n')
    for attendee in attendee_dict.keys():
        # if len(attendee_dict[attendee]) == 0:
        #     continue
        f.write(f'{attendee}')
        for pref in attendee_dict[attendee]:
            f.write(f',{pref}')
        f.write('\n')

# %%
fully_satisfied_attendees = set()
partially_satisfied_attendees = set()
unsatisfied_attendees = set()

planning = defaultdict(set)
for round_id in range(nb_rounds):
    round_full = False
    i = 0
    remaining_sponsors = set(sponsors.copy())
    while (len(remaining_sponsors) > 0) and (i < len(attendees)):
        choices = attendees_choices[i]['choices']
        attendee_id = attendees_choices[i]['attendee_id']
        
        if len(choices) > 0:
            choice = choices[0]
            if choice in remaining_sponsors:
                planning[f'round_{round_id}'].add((choice, attendee_id))
                attendees_choices[i]['choices'].pop(0)
                remaining_sponsors.remove(choice)
                partially_satisfied_attendees.add(attendee_id)
        
        else:
            fully_satisfied_attendees.add(attendee_id)

        i += 1

# %%
unsatisfied_attendees = set(attendees) - partially_satisfied_attendees
partially_satisfied_attendees = partially_satisfied_attendees - fully_satisfied_attendees

# %%
len(unsatisfied_attendees)

# %%
len(partially_satisfied_attendees)

# %%
len(fully_satisfied_attendees)

# %%
json_planning = dict()
for k, v in planning.items():
    json_planning[k] = list(v)

# %%
with open('planning.json', "w", encoding='utf8') as json_file:
    json.dump(json_planning, json_file)

# %%
for round, assignments in planning.items():
    print(f'{round} --> {len(assignments)}')

