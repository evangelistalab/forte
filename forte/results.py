class Results:
    """
    A class used to store the output of a computation.

    Attributes
    ----------
    """
    def __init__(self):
        self._data = dict()

    def add(self, label, value, description, units):
        self._data[label] = {'value': value, 'description': description, 'units': units}

    def value(self, label):
        return self._data[label]['value']

    def __repr__(self):
        return repr(self._data)

    def __str__(self):
        return str(self._data)