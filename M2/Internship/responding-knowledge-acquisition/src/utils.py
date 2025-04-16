import json                 # for json processing
import urllib.parse         # for type safety
import os, sys              # lutra compilation


prefixes_str = '''\
@prefix tp: <https://ontologies.traceparts.com/> .
@prefix gist: <https://ontologies.semanticarts.com/gist/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix ottr:  <http://ns.ottr.xyz/0.4/> .
@prefix ax: <http://tpl.ottr.xyz/owl/axiom/0.1/>.
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

'''

'''
Inputs:
    @text       - text whose language is predicted
    @bias       - likely language of text

Outputs:
    @lang: str  - precited language
'''
def predict_language(text: str, bias: str) -> str:
    # if bias == 'def':
    #     return bias
    # return 'unknown'
    return bias

def generate_id_uri(id_str):
    id_safe = urllib.parse.quote(id_str, safe='')
    return 'tp:ID' + id_safe, id_safe

def generate_text_uri(text_str):
    text_hash = str(hash(text_str))
    return 'tp:Text' + text_hash, text_hash

def generate_pf_uri(pf_str):
    pf_safe = urllib.parse.quote(pf_str, safe='')
    return 'tp:PF' + pf_safe, pf_safe

def generate_pn_uri(pn_str):
    pn_safe = urllib.parse.quote(pn_str, safe='')
    return 'tp:PN' + pn_safe, pn_safe

# def generate_uri(data, concept):
#     ns = 'tp:' + concept
#     if concept == 'Text':
#         return ns+hash(data)
#     elif concept == 'ID' or concept == 'PF' or concept == 'PN':
#         return ns+urllib.parse.quote(data, safe='')
#     elif concept == 'Language':
#         return ns+''

def process_id(id_):
    uri, id_safe = generate_id_uri(id_)
    return f'tp:createIdInstance({uri}, "{id_}"^^xsd:string).\n'

'''
Inputs:
    @nlp        - spacy language model
    @name       - PF/PN-Name
    @language   - language in which PF/PN-Name is expressed in
'''
def process_pf_pn(nlp, name: str, language: str):
    result = ''
    name_hash = hash(name)
    processed_language = 'tp:Language-' + language
    tmp = processed_language
    
    if predict_language(name, language) == 'unknown':
        processed_language = '[]'

    # creating a new doc object
    doc = nlp(name)

    ids_str = ''
    for ent in doc.ents:
        if ent.label_ == 'ID':
            id_ = ent.text
            uri, id_safe = generate_id_uri(id_)
            # create (PF/PN-Name) ID instance [ID-URI, Text-str]
            result += process_id(id_)
            ids_str += f'{uri},'
    if len(ids_str):
        ids_str = ids_str[:-1]

    # create Text instance for PN/PF-Name [Text-URI, Text-str, Lang-URI, IDs]
    name = name.replace('\\', '/') # Be Careful: Changing '\' to '/' in text instances!
    # if name.find('Add-on') == 0:
    #     print(name)
    if processed_language == '[]':
        print(tmp)
        exit(1)
    result += f'tp:createTextInstance(tp:Text{name_hash}, "{name}"^^xsd:string, {processed_language}, ({ids_str})).\n'

    return result

def process_pf(nlp, pf_id: str, pf_name: str, language: str):
    pf_uri, pf_id_safe = generate_pf_uri(pf_id)
    id_uri, id_safe = generate_id_uri(pf_id)
    pf_name_uri, pf_name_hash = generate_text_uri(pf_name)

    tmp =  process_pf_pn(nlp, pf_name, language)
    return process_pf_pn(nlp, pf_name, language) + \
        f'tp:createPartFamilyInstance({pf_uri}, \
        {id_uri}, {pf_name_uri}).\n'

def process_pn(nlp, pn_number: str, pn_name: str, pf_id: str, language: str):
    pn_uri, pn_number_safe = generate_pn_uri(pn_number)
    pn_name_uri, pn_name_hash = generate_text_uri(pn_name)
    pf_uri, pf_safe = generate_pf_uri(pf_id)
    id_uri, id_safe = generate_id_uri(pn_number)

    return process_pf_pn(nlp, pn_name, language) + \
        f'tp:createPartNumberInstance({pn_uri}, {id_uri}, \
        {pf_uri}, {pn_name_uri}).\n'

def compile_ottr(ottr_lib_path, rdf_out_dir, lutra_path):
    print('Massive compilation in progress @[{rdf_out_dir}]...')
    ottr_out_paths = os.listdir(rdf_out_dir)
    ottr_out_paths.remove('lib.stottr')
    ottr_out_paths = [rdf_out_dir+file for file in filter(lambda f: '.stottr' in f, ottr_out_paths)]
    print(ottr_out_paths)
    for ottr_out_path in ottr_out_paths:
        print(f'compiling {ottr_out_path}...')
        index = ottr_out_path[ottr_out_path.find('_'):ottr_out_path.find('.')]
        os.system(f'java -jar {lutra_path} -m expand -f \
                -O wottr -o {rdf_out_dir}output{index}.ttl \
                -L stottr -l {ottr_lib_path} \
                -I stottr {ottr_out_path}')

