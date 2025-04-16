from etlops import *


class Extractor(ETLOperation):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def extract(self, data: OperationalData) -> OperationalData:
        pass
