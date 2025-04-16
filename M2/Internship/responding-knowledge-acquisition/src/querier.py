import requests
import json
import configparser
import copy
from pprint import pprint

from extractor import *
from loader import *


class Querier(Extractor, Loader):
    def __init__(self, config=None):
        super().__init__()
        self.session = requests.Session()


class SPARQLQuery(Querier):
    def __init__(self):
        super().__init__()
        self.limit = 100
        self.offset = 0

    def build_query(self):
        # PREFIX : <undefined>
        query = {'query':'''
                        PREFIX gist: <https://ontologies.semanticarts.com/gist/>
                        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                        PREFIX tp: <https://ontologies.traceparts.com/>

                        select ?pf_text ?pn_text ?pf_uri ?fid_uri ?nid_uri 
                        ?ftext ?ntext ?flang_uri ?nlang_uri
                        where {{
                          ?pf_uri rdf:type tp:PartFamily.
                          ?pf_uri gist:isIdentifiedBy ?pf_id_uri.
                          ?pf_id_uri gist:uniqueText ?pf_text.
                          ?pn_uri rdf:type tp:PartNumber.
                          ?pn_uri gist:isIdentifiedBy ?pn_id_uri.
                          ?pn_id_uri gist:uniqueText ?pn_text.
                          ?pn_uri tp:hasPartFamily ?pf_uri.
                          ?pf_uri gist:isDescribedIn ?ftext_uri.
                          ?pn_uri gist:isDescribedIn ?ntext_uri.
                          ?fid_uri gist:isPartOf ?ftext_uri.
                          ?nid_uri gist:isPartOf ?ntext_uri.
                          ?ftext_uri gist:containedText ?ftext.
                          ?ntext_uri gist:containedText ?ntext.
                          ?ftext_uri gist:isExpressedIn ?flang_uri.
                          ?ntext_uri gist:isExpressedIn ?nlang_uri.
                        }}
                        limit {}
                        offset {}'''.format(self.limit, self.offset)
                        }
        self.offset += self.limit
        return query

    def extract(self, data: OperationalData) -> OperationalData:
        sparql_endpoint = data.data.get('sparql_endpoint')
        request_body = data.data.get('request_body')
        # e.g., {'query': 'select ?s ?p ?o where { ?s ?p ?o } limit 25'}
        n_chunk = data.data.get('n_chunk', 1)

        request_body = self.build_query()

        response = self.session.request('POST', sparql_endpoint, data=request_body)
        response_json = json.loads(response.text)
        variables = response_json['head']['vars']
        # e.g., ['s', 'p', 'o']
        triples = response_json['results']['bindings']
        # e.g., [{'s': {'type': 'uri', 'value': 'tp:ID-FR32V'}, 
        #         'p': {'type': 'uri', 'value': 'gist:uniqueText'},
        #         'o': {'type': 'literal', 'value': 'FR32V'}}, ...]
        triples = [{variable: triple[variable]['value'] for variable in variables} 
                for triple in response_json['results']['bindings']]
        # e.g., [{'s': 'tp:ID-FR32V', 'p': 'gist:uniqueText', 'o': 'FR32V'}, ...]
        # pprint(triples)

        triple_chunks = []
        n_triple_per_chunk = len(triples) // n_chunk
        chunks = []
        for i, triple in enumerate(triples):
            chunks.append(triple)
            if (i+1) % n_triple_per_chunk == 0:
                triple_chunks.append(chunks)
                chunks = []
        for i in range(len(triples) % n_chunk):
            triple_chunks[-1].append(triples[-i-1])

        data.data['triple_chunks'] = triple_chunks
        # pprint(triple_chunks)
        # print(len(triple_chunks))
        # for chunk in triple_chunks:
        #    print(len(chunk))
        data.data['responses'] = data.data['triple_chunks'][0]

        if not len(triple_chunks):
            self.is_active = False
        return data

    def load(self, data: OperationalData) -> OperationalData:
        return data

    def run(self, data: OperationalData) -> OperationalData:
        if data.data.get('method', 'extract') == 'extract':
            return self.extract(data)
        return self.load(data)


'''
This class is for querying ES database.

Inputs:
    @index                      - elasticsearch index to be queried
    @body                       - query body (i.e., GET request body)
    @batch_size                 - 
    @scroll_cache_timeout       - amount of time the query details to be cached (in minutes)
'''
class ESQuery(Querier):
    def __init__(self, index=None, body=None, batch_size=10, scroll_cache_timeout=1):
        super().__init__()
        self.scroll_id = None

        config = configparser.ConfigParser()
        config.read('config.ini')
        defaults = config['DEFAULT']
        es = config['ES']

        self.url = es['dstro_es_url']
        self.port = es['dstro_es_port']
        self.product_idx = es['dstro_es_product_idx']
        self.scroll_url = self.url + self.port + self.product_idx + f'/_search?scroll={scroll_cache_timeout}m'

        self.index = index
        self.body = body
        self.batch_size = batch_size
        self.scroll_cache_timeout = scroll_cache_timeout

        if index is None:
            self.index = ''
        self.target_url = self.url + self.port + self.index + f'/_search?scroll={scroll_cache_timeout}m'

    def __scroll_query__(self):
        self.target_url = self.url + self.port + '/_search/scroll'
        query = {
                'scroll': f'{self.scroll_cache_timeout}m',
                'scroll_id': self.scroll_id
                }
        return query

    '''
    Inputs:
        @body                   - request body
        @method                 - request method
        @n_chunk                - number of chunks to be returned as response bodies

    Outputs:
        @header                 - response header (i.e., {"took": 2, "_shards": {...}, ...}
        @response_chunks        - response bodies (i.e., [{doc0}, {doc1}, ...]
    '''
    def query(self, body=None, method='GET', n_chunk=1):
        query_body = self.body

        if body is not None:
            self.body = body
            query_body = self.body
        elif self.scroll_id is not None:
            print('building with scroll...')
            query_body = self.__scroll_query__()

        if query_body is None:
            raise Exception('Invalid request body')

        # sending request
        response = self.session.request(method, self.target_url, json=query_body)
        response_json = json.loads(response.text)
        self.scroll_id = response_json['_scroll_id']

        # getting response header
        header = copy.deepcopy(response_json)
        header['hits'].pop('hits', None)

        # gathering 'n_chunk' reponse body chunks
        hits = response_json['hits']['hits']
        chunk_size = len(hits) // n_chunk
        response_chunks = []
        if len(hits):
            response_chunks = [hits[i*chunk_size:(i+1)*chunk_size] for i in range(0, n_chunk-1)]
            response_chunks.append(hits[(n_chunk-1)*chunk_size:])

        print(f'\n\nreturning header and {len(response_chunks)} chunks... {[len(chunk) for chunk in response_chunks]}\n\n')
        return header, response_chunks

    def extract(self, data: OperationalData) -> OperationalData:
        body = data.data.get('body', None)
        n_chunk = data.data.get('n_chunk', 1)

        header, result = self.query(body, n_chunk=n_chunk)
        # data.data['extracted_data'] = result
        data.data['data_chunk'] = result
        data.data['ottr_out_path'] = 'ottr/generated.stottr'

        # temporary: to be removed!
        data.data['ottr_lib_path'] = 'ottr/lib.stottr'
        data.data['rdf_out_dir'] = 'ottr/'
        data.data['lutra_path'] = '~/Downloads/lutra.jar'

        self.is_active = bool(len(result))
        return data

    def load(self, data: OperationalData) -> OperationalData:
        return data

    def run(self, data: OperationalData) -> OperationalData:
        print('extract...', data.data.get('c1_method', 'extract'))
        if data.data.get('c1_method', 'extract') == 'extract':
            return self.extract(data)
        return self.load(data)

