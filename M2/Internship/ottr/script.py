import numpy as np
import pandas as pd
import rdflib as rdf
import os, sys


def create_ottr(db):
    pass


db = None
'''
    db format:
    |   text    |   text_str    |   id  |   id_str  |
'''

with open('ins.ottr', 'w') as f:
    f.write('''
        @prefix tp: <http://ontologies.traceparts.com/ontology/> .
        @prefix ottr: <http://ns.ottr.xyz/0.4/> .
        @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
        @prefix ax: <http://tpl.ottr.xyz/owl/axiom/0.1/> .
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix o-rdfs: <http://tpl.ottr.xyz/rdfs/0.2/> .
        @prefix rstr: <http://tpl.ottr.xyz/owl/restriction/0.1/> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        ''')
    for i, row in enumerate(db):
        f.write(f'tp:Text{i}, "{row[0]}", tp:English, (?), (?)')

create_ottr(db)
os.system(f'java -jar ~/Downloads/lutra.jar -m expand -O stottr -o output -L stottr -l lib.stottr -I stottr ins.stottr')
