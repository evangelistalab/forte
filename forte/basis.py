class Basis:
    """
    A class used to store basis information.

    Attributes
    ----------
    basis : str
        the basis name
    """
    def __init__(self, basis):
        """
        initialize a Basis object

        Parameters
        ----------
        basis : str
            a basis object
        """
        self._basis = basis

    def __repr__(self):
        """
        return a string representation of this object
        """
        return f'Basis(\'{self._basis}\')'

    def __str__(self):
        """
        return a string representation of this object
        """
        return self._basis

    @property
    def basis(self):
        """
        return the basis as a string
        """
        return self._basis
