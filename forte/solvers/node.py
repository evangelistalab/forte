from forte.data import Data


class Node():
    """
    Represents a node part of a computational graph.

    A ``Node`` provides only basic functionality and this class
    is specialized in derived classes (Solver, Input,...).
    A ``Node`` stores the list of input nodes and a
    list of features (elements of ``Features``) that are needed and provided.
    The features that are needed must be part of the input node(s).
    This information is used to check the validity of a graph.
    """
    def __init__(self, needs, provides, input_nodes=None, data=None):
        """
        Parameters
        ----------
        needs: list(Feature)
            a list of features required by this solver
        provides: list(Feature)
            a list of features provided by this solver
        input: list(Solver)
            a list of input nodes to this node
        """
        self._needs = needs
        self._provides = provides
        # input_nodes can be None, a single element, or a list
        input_nodes = [] if input_nodes is None else input_nodes
        self._input_nodes = input_nodes if type(input_nodes) is list else [input_nodes]
        self._check_input()
        self._data = Data() if data is None else data

    @property
    def needs(self):
        return self._needs

    @property
    def provides(self):
        return self._provides

    @property
    def input_nodes(self):
        return self._input_nodes

    @property
    def data(self):
        return self._data

    def _check_input(self):
        # verify that this solver can get all that it needs from its inputs
        for need in self.needs:
            need_met = False
            for input in self.input_nodes:
                if need in input.provides:
                    need_met = True
            if not need_met:
                raise AssertionError(
                    f'\n\n  ** The computational graph is inconsistent ** \n\n{self.computational_graph()}'
                    f'\n\n  The solver {self.__class__.__name__} cannot get a feature ({need}) from its input solver: '
                    + ','.join([input.__class__.__name__ for input in self.input_nodes]) + '\n'
                )

    def computational_graph(self):
        """
        Make a simple text representation of the computational graph

        For example

            D
            ├──C
            │  ├──B
            │  │  └──A
            │  └──B
            │     └──A
            └──E

        """
        # all the details are hidden in _make_graph()
        return self._make_graph()

    def _make_graph(self, level=0, prefix=None, last=False):
        # The algorithm uses a recursive code that is a bit hard to follow
        # this will collect string
        graph = []
        # this variable is used to store the text that goes to the left of a node label
        prefix = [] if prefix is None else prefix
        # at the 0-th level just print the class name
        if level == 0:
            graph.append(f'{self.__class__.__name__}')
            # graph = f'{self.__class__.__name__}\n'
        else:
            # at the i-th level print:
            # - the prefix up to the previous level
            # - a connection
            # - the class name
            if last:
                # if we are printing the last node, use the two way connector └
                graph.append(''.join(prefix[:level - 1]) + f'└──{self.__class__.__name__}')
                # graph = ''.join(prefix[:level - 1]) + f'└──{self.__class__.__name__}\n'
            else:
                # if we are printing the any other node, use the three way connector ├
                graph.append(''.join(prefix[:level - 1]) + f'├──{self.__class__.__name__}')
                # graph = ''.join(prefix[:level - 1]) + f'├──{self.__class__.__name__}\n'
        # now let's think what should happen to the prefix after we display this done
        if len(self.input_nodes) > 1:
            # if we have 2 or more input nodes, we should print a vertical line in the next level of printing
            prefix.append('│  ')
        else:
            # otherwise, print empty text
            prefix.append('   ')
        # now we ask to print the input nodes
        last = False
        for k, input in enumerate(self.input_nodes):
            # we have to treat the last node in a special way, so let's flag it
            # and make sure that from now on the prefix is blank, otherwise we would print extra lines
            if k == len(self.input_nodes) - 1:
                last = True
                prefix[-1] = '   '
            graph.append(input._make_graph(level + 1, prefix, last))
        # we are now back one level down, so pop the last prefix added
        prefix.pop()
        # return graph
        return '\n'.join(graph)
