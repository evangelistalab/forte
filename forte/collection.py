class Collection:
    """
    A class to that handles forte objects

    Attributes
    ----------
    """
    def __init__(self, scf_info=None, mospaceinfo=None, ints=None, psi_wfn=None):
        """
        initialize a Basis object

        Parameters
        ----------
        restricted : bool
            a basis object
        """
        self._collection = {}
        self._collection['scf_info'] = scf_info
        self._collection['mospaceinfo'] = mospaceinfo
        self._collection['ints'] = ints
        self._collection['psi_wfn'] = psi_wfn

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
    def scf_info(self):
        return self._collection['scf_info']

    @scf_info.setter
    def scf_info(self, val):
        self._collection['scf_info'] = val

    @property
    def mospaceinfo(self):
        return self._collection['mospaceinfo']

    @property
    def ints(self):
        return self._collection['ints']

    @property
    def psi_wfn(self):
        return self._collection['psi_wfn']

    @psi_wfn.setter
    def psi_wfn(self, val):
        self._collection['psi_wfn'] = val
