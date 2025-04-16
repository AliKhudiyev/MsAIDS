from etlops import *

from extractor import *
from transformer import *
from loader import *

from querier import *
from fileops import *

import threading


'''
ETL class is to perform Extract -> Transform -> Load operations from/to database(s).
This class can be initialized to work with arbitrary databases doing arbitrary transformations 
on the database entities as long as all is given as specified.

Inputs:
    @extractor      - Extractor object used to extract entities from a database (usually Querier)
    @transformer    - Transformer object used to transform the extracted entities
    @loader         - Loader object used to load the transformed entities to a database (usually Querier)

Object of this class should invoke the 'run(...)' function with the specified inputs:
    @parallelism    - [0/1], 0 for no parallelism and 1 for parallelism in ETL process level
'''
class ETL:
    def __init__(self, pipeline=[]):
        self.pipeline = pipeline
        self.entity_buffer = []

    def from_disk(self, file_path):
        pass

    def to_disk(self, file_path):
        pass
    
    def _should_not_terminate_(self):
        return self.pipeline[0].is_active

    def _run_seq_(self, data: OperationalData) -> OperationalData:
        n = 0
        while self._should_not_terminate_():
            # print('epoch:', n, self.pipeline[0].offset, self.pipeline[0].limit)
            n += 1
            for component in self.pipeline:
                data = component.run(data)
            data.data['body'] = None
            break
        return data

    def _run_par_(self, data: OperationalData) -> OperationalData:
        # TODO
        extractor, transformer, laoder = self.pipeline

        while self._should_not_terminate_():
            # mass extraction
            data = extractor(data)

            # parallelized transformations
            threads = [threading.Thread(target=transformer, args=(data,)) for i in range(4)]
            for thread in threads:
                thread.run()
                thread.join()

        # mass load
        data = loader(data)

    def run(self, data: OperationalData) -> OperationalData:
        print('pipeline length:', len(self.pipeline))
        parallelism = data.data.get('parallelism', 0)
        print('parallelism:', parallelism)
        if parallelism == 0:
            return self._run_seq_(data)
        elif parallelism == 1:
            return self._run_par_(data)

