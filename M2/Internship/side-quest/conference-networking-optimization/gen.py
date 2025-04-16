from random import sample, randint
import string, sys
from pprint import pprint as pp
from collections import defaultdict

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
nb_attendees = int(sys.argv[2])
nb_sponsors = len(sponsors)
nb_rounds = 12
# sponsors = sample(string.ascii_uppercase, nb_sponsors)
attendees = sample(surnames, nb_attendees)

# %%
attendee_dict = dict()

for attendee in attendees:
    nb_choices = randint(2, nb_rounds)
    attendee_dict[attendee] = sample(sponsors, nb_choices)

# attendees_choices = [{'attendee_id': key, 'choices': value} for key, value in attendee_dict.items()]
# attendees_choices.sort(reverse=False, key=lambda x: x['attendee_id'])

with open(sys.argv[1], 'w') as f:
    f.write(f'{nb_sponsors} {nb_attendees}\n')
    for sponsor in sponsors:
        f.write(f'{sponsor}\n')
    for attendee in attendee_dict.keys():
        f.write(f'{attendee}')
        for pref in attendee_dict[attendee]:
            f.write(f',{pref}')
        f.write('\n')

