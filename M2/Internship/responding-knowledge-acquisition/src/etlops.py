from abc import ABC, abstractmethod
import os, sys

from opdata import *


class ETLOperation(ABC):
    def __init__(self):
        self.is_active = True

    def from_config(file_path):
        pass

    def to_config(file_path):
        pass

    @abstractmethod
    def run(self, data: OperationalData) -> OperationalData:
        pass

