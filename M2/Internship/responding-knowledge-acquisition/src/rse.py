import spacy
from spacy.language import Language

import configparser
from configparser import ConfigParser

from querier import *


class RSEResult:
    def __init__(self, query='', doc_ids=[]):
        self.query = query
        self.doc_ids = doc_ids

class RSE:
    def __init__(self, config='rse.ini'):
        self.config = ConfigParser()
        self.config.read(config)

        defaults = self.config['DEFAULT']
        tdb_endpoint = defaults['tdb_end_point']
        ses_endpoint = defaults['ses_end_point']

        self.tdb_endpoint = SPARQLQuery()
        self.ses_endpoint = ESQuery()

        self.nlp = Language()
        pipeline = ['entity_ruler']
        for component in pipeline:
            self.nlp.add_pipe(component)
        self.nlp.from_disk('./language_model')

    # returns a list of tokens [{concept: instance}]
    @staticmethod
    def extractable_tokens(query: str) -> []:
        # TODO
        pass

    # returns a list of tokens [{concept: instance}]
    @staticmethod
    def nonextractable_tokens(query: str) -> []:
        tokens = query.split(' ')
        extracted_tokens = RSE.extractable_tokens(query)
        unextracted_tokens = []
        for token in tokens:
            if token not in extracted_tokens:
                unextracted_tokens.append({'unknown': token})
        return unextracted_tokens

    @staticmethod
    def tokens(query: str):
        return RSE.extractable_tokens(query), RSE.nonextractable_tokens(query)

    # e.g., tokens = [{'ID': 'H7EF33'}, {'PartFamily': 'FF34-102-8RJK'}]
    def search_tdb(self, tokens: []) -> []:
        data = OperationalData({
            'body': {},
            'n_chunk': 1
            })
        data = self.tdb_endpoint.run(data)
        result = data['data_chunk']

        output_tokens = []
        # TODO
        return output_tokens

    # this function returns a list of document ids as a bunch of strings
    def search_ses(self, tokens: []) -> [str]:
        data = OperationalData({
            'body': {},
            'n_chunk': 1
            })
        data = self.tdb_endpoint.run(data)
        result = data['data_chunk']

        doc_ids = []
        # TODO
        return doc_ids

    def search(self, query: str) -> RSEResult:
        result = RSEResult(query)
        extractable_tokens, nonextractable_tokens = RSE.tokens(query)
        enriched_tokens = self.search_tdb(extractable_tokens)
        result.doc_ids = self.search_es(nonextractable_tokens+enriched_tokens)
        return result

