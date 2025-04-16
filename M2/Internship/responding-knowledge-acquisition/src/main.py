import numpy as np
import pandas as pd         # sample csv database operations
import os, re               # expression matching, compiling with lutra
import configparser         # config reading/writing
import urllib.parse         # for type safety
import logging              # for debugging
import json                 # sample json es docs operations
import threading            # for processing different parts of the database simultaneously
import random
from pprint import pprint

import spacy
from spacy.language import Language
from spacy.tokenizer import Tokenizer
from spacy.matcher import Matcher

from etl import *
from etlops import *
from extractor import *
from transformer import *
from loader import *

from querier import *
from fileops import *

from opdata import *
from utils import *


config = None

def etl_es2tdb(data, nlp):
    extractor = ESQuery()
    print('Extractor loaded as ESQuery... ok')
    transformer = ES2TDB(nlp)
    print('Transformer loaded as ES2TDB... ok')
    loader = TerminalOps()
    print('Loader loaded as TerminalOps... ok')

    etl = ETL([extractor, transformer, loader])
    print('ETL initialized... ok')

    etl.run(data)
    print('done')
 
def etl_tdb2ses(data, nlp=None):
    extractor = SPARQLQuery()
    print('Extractor loaded as ESQuery... ok')
    transformer = TDB2ES(nlp)
    print('Transformer loaded as ES2TDB... ok')
    loader = JSONOps()
    print('Loader loaded as TerminalOps... ok')

    etl = ETL([extractor, transformer, loader])
    print('ETL initialized... ok')

    etl.run(data)
    print('done')


def main(data_path, ottr_lib_path, lutra_path, rdf_out_dir):
    nlp = Language()
    pipeline = ['entity_ruler']
    for component in pipeline:
        nlp.add_pipe(component)
    nlp.from_disk('./language_model')
    print('Language model loaded... ok')

    data = OperationalData({
        'c1_method': 'extract',
        'c3_method': 'load',
        'body': {
            'size': 40
            },
        'n_chunk': 1,
        'parallelism': 0,

        'sparql_endpoint': 'http://localhost:3030/responding/sparql',
        'ses_path': 'ottr/ses.json'
        })
    print('Operational Data has been initialized... ok')

    c = int(input('ES2TDB(0) / TDB2ES(1)'))
    if not c:
        etl_es2tdb(data, nlp)
    else:
        etl_tdb2ses(data, nlp)


if __name__ == '__main__':
    logging.basicConfig(filename='logs/rdfizer.log', level=logging.INFO)
    logging.info('\n\nFetching data/library/output/lutra paths...')
    # fetching environment variables
    DATA_PATH = None
    OTTR_LIB_PATH = None
    RDF_OUT_DIR = None
    LUTRA_PATH = None
    size = None

    try:
        # csv file path
        DATA_PATH = os.environ['DATA_PATH']
        print(f'DATA PATH: {DATA_PATH}')

        # ottr library path
        OTTR_LIB_PATH = os.environ['OTTR_LIB_PATH']
        print(f'OTTR LIB PATH: {OTTR_LIB_PATH}')

        # rdf output path
        RDF_OUT_DIR = os.environ['RDF_OUT_DIR']
        print(f'RDF OUT DIRECTORY: {RDF_OUT_DIR}')

        # lutra path
        LUTRA_PATH = os.environ['LUTRA_PATH']
        print(f'LUTRA PATH: {LUTRA_PATH}')
    except:
        logging.warning('Environment variables are not properly set... Reading the config.ini')
        try:
            config = configparser.ConfigParser()
            config.read('config.ini')
            defaults = config['DEFAULT']

            DATA_PATH = defaults['data_path']
            OTTR_LIB_PATH = defaults['ottr_lib_path']
            RDF_OUT_DIR = defaults['rdf_out_dir']
            LUTRA_PATH = defaults['lutra_path']

            n_thread = int(defaults['n_thread'])
            n_chunk_per_thread = int(defaults['n_chunk_per_thread'])
            size = n_chunk_per_thread*n_thread

            es = config['ES']
            dstro_es_url = es['dstro_es_url']
            dstro_es_product_idx = es['dstro_es_product_idx']
        except:
            logging.error(f'DATA_PATH[{DATA_PATH}] or OTTR_LIB_PATH[{OTTR_LIB_PATH}] \
                    or RDF_OUT_DIR[{RDF_OUT_DIR}] or LUTRA_PATH[{LUTRA_PATH}] not found!')
            exit(1)

    logging.info(' > done')
    os.system('rm ottr/generated*; rm ottr/output*') # To be removed
    main(DATA_PATH, OTTR_LIB_PATH, LUTRA_PATH, RDF_OUT_DIR)
    logging.info('Success!')
