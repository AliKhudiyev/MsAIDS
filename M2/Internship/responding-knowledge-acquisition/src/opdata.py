class Data:
    def __init__(self):
        pass

class OperationalData(Data):
    def __init__(self, data=None):
        self.data = data

    def __str__(self):
        return str(self.data)

