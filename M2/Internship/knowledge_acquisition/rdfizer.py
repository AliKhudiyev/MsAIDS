import numpy as np
import pandas as pd
import networkx as nx
import random, os, re, io
import configparser
import urllib.parse
import matplotlib.pyplot as plt

import rdflib as rdf
from rdflib import Graph, FOAF, RDF
from rdflib import BNode, URIRef, Literal
from rdflib import Namespace
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph

import spacy
from spacy.matcher import Matcher


# mappings from template arguments to database columns or rdf graph concepts
ottr_database_mappings = {'id_str': 'ID-str',  'text_str': 'Text-str', 'id': 'ID', 'text': 'Text-str', 'texts': 'Text-str', 'language': 'Language'}

# fetching environment variables
try:
    # intermediate file path (database or rdf graph)
    INT_FILE_PATH = os.environ['INT_FILE_PATH']
    print(f'INT FILE PATH: {INT_FILE_PATH}')
except:
    print('no int file path')
try:
    # ottr library path
    OTTR_LIB_PATH = os.environ['OTTR_LIB_PATH']
    print(f'OTTR LIB PATH: {OTTR_LIB_PATH}')
except:
    print('no ottr lib path')


'''
This function takes a raw user query(s) as an input and tokenize it 
then extract relevant information from it. Each query is represented as 
Spacy Doc object.

Inputs:
    @file_path    - raw user query

Outputs:
    @grdf         - intermediate rdf graph (database)
'''
def process_query(file_path):
    print('processing the query...')
    df = pd.read_csv(file_path, index_col=0, names=['text'], skiprows=1)
    text = '\n'.join(np.array(df.values).reshape((len(df.values),)))
    print(text)
    
    ids = []
    # Load English tokenizer, tagger, parser and NER
    nlp = spacy.load("en_core_web_sm")

    for row in df.values:
        text = row[0]
        doc = nlp(text)
    
        # regex for ID extraction
        matcher = Matcher(nlp.vocab)
        pattern = [{"TEXT": {"REGEX": "(?<!\S)(([A-Z]+-*[0-9]+)|([0-9]+-*[A-Z]+))[A-Z0-9]*\\b"}}]
        # (([A-Z]+[0-9]+)|([0-9]+[A-Z]+))[A-Z0-9]*
        matcher.add("IDformat", [pattern])
        matches = matcher(doc)

        print('\n\n')
        tmp_ids = []
        for match_id, start, end in matches:
            string_id = nlp.vocab.strings[match_id]
            span = doc[start:end]
            tmp_ids.append(span.text)
            # tmp_ids.append(f'imid{random.randint(0,1000)}{random.randint(0,1000)}')
            print(match_id, string_id, start, end, span.text)
        ids.append(tmp_ids)


    # create rdf graph
    grdf = rdf.Graph()
    ns = Namespace("https://ontologies.traceparts.com/")

    # define concepts and relations
    con_id = ns.ID
    con_text = ns.Text
    rel_isPartOf = ns.isPartOf
    rel_containedText = ns.containedText
    rel_uniqueText = ns.uniqueText

    id_uri_index = 0
    text_uri_index = 0

    # generate graph
    print('generating graph...')
    for i, ids_ in enumerate(ids):
        # if i >= len(df.values):
        #     break
        # if i%10:
        #     continue
        con_text_ = URIRef(f"https://ontologies.traceparts.com/Text{hash(df.values[i][0])}")
        text_uri_index += 1
        grdf.add((con_text_, RDF.type, con_text))
        grdf.add((con_text_, rel_containedText, Literal(df.values[i][0])))
        
        for j, id_ in enumerate(ids_):
            con_id_ = URIRef(f"https://ontologies.traceparts.com/ID-{urllib.parse.quote(id_)}")
            id_uri_index += 1
            grdf.add((con_id_, RDF.type, con_id))
            grdf.add((con_id_, rel_uniqueText, Literal(id_)))
            grdf.add((con_id_, rel_isPartOf, con_text_))

    # save rdf graph
    print('storing the graph in query.txt and done...')
    grdf.serialize('query.txt', format='turtle')

    return grdf


'''
This function parses the stottr library templates and outputs them.

Inputs:
    @file_path    - ottr template library path

Outputs:
    @templates    - ottr templates formatted as {template_name: [arguments...]}
'''
def parse_ottr_library_templates(file_path, tags=[]):
    print('parsing ottr library...')
    templates = {}
    lines = []

    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        # check if it is main template function (if begins with "tp:M_"
        # print('sampling...', line[:-1])
        match = re.search("(tp:[A-Za-z0-9]+)(\[[\x00-\x7f]+\])", line)
        if match is None:
            # print('match not found :(')
            continue

        # template signature
        sign = match.group(1)
        # template arguments
        arglist = []
        arglist_raw = match.group(2)[1:-1].split(',')

        # print(sign, arglist_raw)
        for arg_raw in arglist_raw:
            arg_raw = arg_raw.replace(' ', '')
            arg = arg_raw.split('?')
            if len(arg) != 2:
                print('error arg len')
            # arg = [type, variable_name]
            arglist.append(arg[1])
            # print('type var:', arg)
        # print('raw arglist:', arglist_raw)

        templates[sign] = arglist

    return templates

'''
This functions generates ottr instance file.

Inputs:
    @file_path    - ottr instance file path (overwritten)
    @templates    - templates parsed from the ottr library file
    @database     - intermediate graph database (gdb) of actual instances

Outputs:
'''
def generate_ottr_instances(file_path, templates, database):
    global ottr_database_mappings

    print('generating ottr instances...')
    # run the query
    results = database.query("""
            SELECT ?text ?uri_text ?id ?uri_id
            WHERE {
                ?uri_id <https://ontologies.traceparts.com/isPartOf> ?uri_text .
                ?uri_id <https://ontologies.traceparts.com/uniqueText> ?id .
                ?uri_text <https://ontologies.traceparts.com/containedText> ?text .
                }
            """)

    # print the results
    db = []
    for row in results:
        # print(f"{row.text:<100} | {row.id:<50} | {row.uri_id:<30}")
        row = [f'{row.text}', f'{row.uri_text}', f'{row.id}', f'{row.uri_id}']
        db.append(row)

    pretty_db = pd.DataFrame(db, columns=['text', 'uri_text', 'id', 'uri_id'])
    print(pretty_db)

    with open(file_path, 'w') as f:
        f.write('''\
@prefix tp: <https://ontologies.traceparts.com/> .
@prefix gist: <https://ontologies.semanticarts.com/gist/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix ottr:  <http://ns.ottr.xyz/0.4/> .
@prefix ax: <http://tpl.ottr.xyz/owl/axiom/0.1/>.
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

tp:Language(tp:English, "English"^^xsd:string).
tp:Language(tp:Turkish, "Turkish"^^xsd:string).
tp:Language(tp:French, "French"^^xsd:string).\n\n''')
        
        for row in db:
            ids = ''
            for row2 in db:
                if row2[0] == row[0]:
                    ids += f'<{row[3]}>,'
            ids = ids[:-1]
            # print(f'ids occuring in "{row[0]}": {ids}')

            f.write(f'tp:ID(<{row[3]}>, "{row[2]}"^^xsd:string).\n')
            f.write(f'tp:Text(<{row[1]}>, "{row[0]}"^^xsd:string, tp:English, ({ids})).\n\n')

        # database = pretty_db
        # for row_index in range(len(database.values)):
            # for temp_sign in templates.keys():
            #     call_str = f'{temp_sign}('
            #     for i, arg in enumerate(templates[temp_sign]):
            #         if temp_sign != 'tp:ID' or temp_sign != 'tp:Text':
            #             continue

            #         col_name = ottr_database_mappings[arg]
            #         val = database.iloc[row_index, list(database.columns).index(col_name)]

            #         if '-str' in col_name:
            #             call_str += '"'
            #         
            #         if temp_sign == 'tp:ID' and arg == 'id':
            #             val = urllib.parse.quote(val)
            #         elif temp_sign == 'tp:ID' and arg == 'textss':
            #             for row_index2 in range(row_index, len(database.values)):
            #                 call_str += ''
            #                 hash(database.iloc[row_index2, list(database.columns).index(col_name)])
            #         elif temp_sign == 'tp:Text' and arg == 'text':
            #             val = 'text' + str(hash(val))

            #         call_str += f'{val}'

            #         if '-str' in col_name:
            #             call_str += '"^^xsd:string'

            #         if i != len(templates[temp_sign])-1:
            #             call_str += ', '
            #     call_str += ').'
            #     f.write(f'{call_str}\n')


if __name__ == '__main__':
    gdb = process_query('../data.csv')
    templates = parse_ottr_library_templates(OTTR_LIB_PATH)
    print(templates)
    # df = pd.read_csv('exdb.csv')
    # print(df)
    # print(list(df.columns))
    # print(df.iloc[1, list(df.columns).index('Text-str')])
    generate_ottr_instances('generated.stottr', templates, gdb)
    os.system(f'java -jar ~/Downloads/lutra.jar -m expand -f -O wottr -o output_rdf -L stottr -l lib.stottr -I stottr generated.stottr')

