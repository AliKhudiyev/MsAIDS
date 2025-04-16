from etlops import *
from utils import *

import spacy


class Transformer(ETLOperation):
    def __init__(self, nlp):
        super().__init__()
        self.nlp = nlp

    @abstractmethod
    def transform(self, data: OperationalData) -> OperationalData:
        pass

    def run(self, data: OperationalData) -> OperationalData:
        return self.transform(data)


class ES2TDB(Transformer):
    def __init__(self, nlp):
        super().__init__(nlp)

    '''
    This function takes a raw user query(s) as an input and tokenize it 
    then extract relevant information from it. Each query is represented as 
    Spacy Doc object.

    Inputs:
        @data_path        - path to the raw text (user query)
        @ottr_out_path    - path to the ottr instance file

    Outputs:
    '''
    def rdfize(self, data_chunk, ottr_out_path):
        if data_chunk is None:
            return

        # logging.info(f'Running rdfizer with data chunk and output path @[{ottr_out_path}]...')
        results = data_chunk # ['hits']['hits']
        if len(results) == 0:
            print('err 0 chunk is provided!')
            exit(1)

        # logging.info(' > done')
        # logging.info(f'Generating stottr file @ [{ottr_out_path}]...')

        with open(ottr_out_path, 'w') as f:
            # writing prefixes (e.g., @prefix ottr: <http://ns.ottr.xyz/0.4/>.)
            f.write(prefixes_str)

            for result in results:
                pf_id = result['_source']['Family']['ID']
                pn_number = result['_source']['PartNumber']['Number'][0]
                pf_names_dict = result['_source']['Family']['Name']
                pn_names_dict = result['_source']['PartNumber']['Name']

                # create PF ID instance [ID-URI, Text-str]
                f.write(process_id(pf_id))
                # create PN ID instance [ID-URI, Text-str]
                f.write(process_id(pn_number))

                for language, pf_name in pf_names_dict.items():
                    # create PF instance [PF-URI, ID-URI, Text-URI]
                    f.write(process_pf(self.nlp, pf_id, pf_name, language))

                for language, pn_name in pn_names_dict.items():
                    # create PN instance [PN-URI, ID-URI, PF-URI, Text-URI]
                    f.write(process_pn(self.nlp, pn_number, pn_name, pf_id, language))
    
    def transform(self, data: OperationalData) -> OperationalData:
        data_chunk = data.data.get('data_chunk', [])
        if len(data_chunk):
            self.rdfize(data_chunk[0], data.data['ottr_out_path'])
            compile_ottr(data.data['ottr_lib_path'], data.data['rdf_out_dir'], data.data['lutra_path'])
        data.data['method'] = 'load'
        data.data['from_dir'] = 'ottr/*.ttl'
        data.data['to_dir'] = '../data/fuseki-data/'
        return data


class TDB2ES(Transformer):
    def __init__(self, nlp):
        super().__init__(nlp)

    def transform(self, data: OperationalData) -> OperationalData:
        if 'docs' not in data.data:
            data.data['docs'] = []

        docs = data.data['docs']
        old_doc_ids = [old_doc['doc_id'] for old_doc in data.data['docs']]

        for response in data.data['responses']:
            result = {}
            doc_id = response['pf_text'] + ':' + response['pn_text']

            if doc_id not in old_doc_ids:
                docs.append(result)
                old_doc_ids.append(doc_id)
            else:
                for i in range(len(docs)):
                    if doc_id == docs[i]['doc_id']:
                        result = docs[i]
                        break
                # result = filter(lambda doc: doc['doc_id'] == doc_id, docs)

            result['doc_id'] = doc_id
            result['uri'] = response['pf_uri']

            if result.get('concept_uris', None) is None:
                result['concept_uris'] = []
            
            fid_uri = response['fid_uri']
            nid_uri = response['nid_uri']

            if fid_uri not in result['concept_uris']:
                result['concept_uris'].append(fid_uri)
            if nid_uri not in result['concept_uris']:
                result['concept_uris'].append(nid_uri)

            if 'searchable_texts' not in result:
                result['searchable_texts'] = {}

            flang = response['flang_uri']
            nlang = response['nlang_uri']

            result['searchable_texts'][flang] = response['ftext']
            result['searchable_texts'][nlang] = response['ntext']

        data.data['docs'] = docs
        data.data['ses_path'] = 'ottr/ses.json'
        return data

