import spacy
from spacy.matcher import Matcher

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

import rdflib as rdf
from rdflib import Graph, FOAF, RDF
from rdflib import BNode, URIRef, Literal
from rdflib import Namespace
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph


df = pd.read_csv('data.csv', index_col=0, names=['text'], skiprows=1)
text = '\n'.join(np.array(df.values).reshape((len(df.values),)))
print(text)

# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

# regex for ID extraction
matcher = Matcher(nlp.vocab)
pattern = [{"TEXT": {"REGEX": "(?<!\S)(([A-Z]+-*[0-9]+)|([0-9]+-*[A-Z]+))[A-Z0-9]*\\b"}}]
# (([A-Z]+[0-9]+)|([0-9]+[A-Z]+))[A-Z0-9]*
matcher.add("IDformat", [pattern])
matches = matcher(doc)

print('\n\n')
ids = []
for match_id, start, end in matches:
    string_id = nlp.vocab.strings[match_id]
    span = doc[start:end]
    # ids.append(span.text)
    print(match_id, string_id, start, end, span.text)


# create rdf graph
grdf = rdf.Graph()
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
    grdf.add((con_text_, RDF.type, con_text))
    grdf.add((con_text_, rel_hasText, Literal(df.values[i][0])))
    
    for j, id_ in enumerate(ids_):
        con_id_ = URIRef(f"http://example.org/ID{id_uri_index}")
        id_uri_index += 1
        grdf.add((con_id_, RDF.type, con_id))
        grdf.add((con_id_, rel_hasText, Literal(id_)))
        grdf.add((con_id_, rel_occursIn, con_text_))

# save rdf graph
grdf.serialize('query.txt', format='turtle')

# transform rdf graph into networkx graph
src = 'query.txt' # 'https://www.w3.org/TeamSubmission/turtle/tests/test-30.ttl' # 'query.txt'
gnx = rdflib_to_networkx_multidigraph(grdf.parse(src, format='turtle'))

pos = nx.spring_layout(gnx, scale=2)
edge_labels = nx.get_edge_attributes(gnx, 'r')
# print('edge labels:', edge_labels)
# nx.draw_networkx_edge_labels(gnx, pos, edge_labels=edge_labels)
# nx.draw(gnx, with_labels=True)

# if not in interactive mode for 
# plt.show()

