from etlops import *


class Loader(ETLOperation):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def load(self, data: OperationalData) -> OperationalData:
        pass
