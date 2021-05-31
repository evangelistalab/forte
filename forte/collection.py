class Collection:
    """
    A class to that handles forte objects

    Attributes
    ----------
    """
    def __init__(self,mospaceinfo=None,ints=None):
        """
        initialize a Basis object

        Parameters
        ----------
        restricted : bool
            a basis object
        """
        self._collection = {}
        _objects['mospaceinfo'] = mospaceinfo
        _objects['ints'] = ints

    def __repr__(self):
        """
        return a string representation of this object
        """
        return f'Collection()'

    def __str__(self):
        """
        return a string representation of this object
        """        
        return f'Collection()'

    @property
    def mospaceinfo(self):
        return self._objects['mospaceinfo']
