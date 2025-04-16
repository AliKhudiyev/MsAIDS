import pandas as pd
import re, io

from rdflib import Graph, FOAF, RDF 
from rdflib import BNode, URIRef, Literal
from rdflib import Namespace

from SPARQLWrapper import SPARQLWrapper, JSON


def is_valid_ID(word):
    has_capital_letter = False
    has_numeric = False
    has_other_symbol = False

    for c in word:
        if ord(c) >= 48 and ord(c) <= 57:
            has_numeric = True
        elif ord(c) >= 65 and ord(c) <= 90:
            has_capital_letter = True
        else:
            return False
    
    if has_capital_letter and has_numeric:
        return True
    return False

def extract_IDs(text):
    ids = []
    words = text.split(' ')

    for word in words:
        if is_valid_ID(word):
            ids.append(word)

    return ids


# load text file
df = pd.read_csv('data.csv', index_col=0, names=['text'], skiprows=1)
rows = df.values
# print(rows)

# extract IDs
ids = [extract_IDs(text[0]) for text in df.values]
# print(ids)

# create graph
g = Graph()
ns = Namespace("http://example.org/")

# define concepts and relations
con_id = ns.ID
con_text = ns.Text
rel_occursIn = ns.occursIn
rel_hasText = ns.hasText

id_uri_index = 0
text_uri_index = 0

# generate graph
for i, ids_ in enumerate(ids):
    # if i%10:
    #     continue
    con_text_ = URIRef(f"http://example.org/Text{text_uri_index}")
    text_uri_index += 1
    g.add((con_text_, RDF.type, con_text))
    g.add((con_text_, rel_hasText, Literal(df.values[i][0])))
    
    for j, id_ in enumerate(ids_):
        con_id_ = URIRef(f"http://example.org/ID{id_uri_index}")
        id_uri_index += 1
        g.add((con_id_, RDF.type, con_id))
        g.add((con_id_, rel_hasText, Literal(id_)))
        g.add((con_id_, rel_occursIn, con_text_))

# print(g.serialize(format='turtle'))
g.serialize('query.txt', format='turtle')

# run the query
results = g.query("""
        SELECT ?text ?id ?uri_id
        WHERE {
            ?uri_id <http://example.org/occursIn> ?uri_text .
            ?uri_id <http://example.org/hasText> ?id .
            ?uri_text <http://example.org/hasText> ?text .
            }
        """)

# print the results
db = []
for row in results:
    # print(f"{row.text:<100} | {row.id:<50} | {row.uri_id:<30}")
    row = [f"{row.text}", f"{row.id}", f"{row.uri_id}"]
    db.append(row)

pretty_db = pd.DataFrame(db, columns=['text', 'id', 'uri_id'])
print(pretty_db)
