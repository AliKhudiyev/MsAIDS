from extractor import *
from loader import *

import json


'''
Inputs:
    @data       - dictionary
'''
class FileOps(Extractor, Loader):
    def __init__(self):
        super().__init__()


class TerminalOps(FileOps):
    def __init__(self):
        super().__init__()

    def extract(self, data: OperationalData) -> OperationalData:
        return data

    def load(self, data: OperationalData) -> OperationalData:
        from_dir = data.data.get('from_dir')
        to_dir = data.data.get('to_dir')
        if os.system(f'cp {from_dir} {to_dir}'):
            print('err occured during file loading')
        os.system(f'cp ./ottr/*.ttl ../data/fuseki-data/')
        os.system(f'docker exec docker_fuseki_1 ./tdbloader -loc /fuseki/databases/responding "/staging/*.ttl"')
        return data

    '''
    data dict layout
    ----------------
    "method" : ["extract"/"load"]
    '''
    def run(self, data: OperationalData) -> OperationalData:
        if data.data.get('c3_method', 'extract') == 'extract':
            return self.extract(data)
        return self.load(data)

class JSONOps(FileOps):
    def __init__(self):
        super().__init__()

    def extract(self, data: OperationalData) -> OperationalData:
        return data

    def load(self, data: OperationalData) -> OperationalData:
        with open(data.data.get('ses_path'), 'w') as f:
            f.write(json.dumps(data.data.get('docs')))
        return data

    def run(self, data: OperationalData) -> OperationalData:
        if data.data.get('c3_method', 'extract') == 'extract':
            return self.extract(data)
        return self.load(data)

