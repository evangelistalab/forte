class Results:
    """
    A class used to store the results of a computation.
    """
    def __init__(self):
        self._data = dict()

    def add(self, label, value, description, units):
        """"Add a result"""
        self._data[label] = {'value': value, 'description': description, 'units': units}

    def value(self, label):
        """"Get the value of a result"""
        return self._data[label]['value']

    def description(self, label):
        """"Get the description of a result"""
        return self._data[label]['description']

    def units(self, label):
        """"Get the units of a result"""
        return self._data[label]['units']

    def __repr__(self):
        return repr(self._data)

    def __str__(self):
        return str(self._data)
