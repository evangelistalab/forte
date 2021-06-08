# from .results import Results

# class Computation:
#     """
#     A class used to represent a computation.

#     Attributes
#     ----------
#     basis : str
#         the basis name
#     """
#     def __init__(self, model=None, method=None):
#         """
#         initialize a Basis object

#         Parameters
#         ----------
#         basis : str
#             a basis object
#         """
#         self._model = model
#         self._method = method

#     def __repr__(self):
#         """
#         return a string representation of this object
#         """
#         return f'Computation(model={repr(self._model)},method={repr(self._method)})'

#     def __str__(self):
#         """
#         return a string representation of this object
#         """
#         return self._basis

#     @property
#     def model(self):
#         return self._model

#     @property
#     def method(self):
#         return self._method

#     def energy(self):
#         """Compute the energy."""
#         results = Results()
#         self._method.energy(self._model, results=results)
#         return results
