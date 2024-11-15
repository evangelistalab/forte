from typing import List
from forte.modules.module import Module
from forte.data import ForteData


class Workflow(Module):
    """
    A class that represents a workflow of modules. It runs a list of modules

    This class is a subclass of Module, and it can be used as a module in another
    workflow. It can be used to run a list of modules in a sequence.

    If the input data is None, it will create a new ForteData object.
    """

    def __init__(self, modules: List[Module] = None, options=None, name=None):
        super().__init__(options)
        self.modules = modules or []
        self.name = name or "Workflow"

    def add_module(self, module: Module):
        self.modules.append(module)

    def _run(self, data: ForteData = None) -> ForteData:
        if data is None:
            data = ForteData()
        for module in self.modules:
            data = module.run(data)
        return data

    def __repr__(self):
        return f"{self.name}"
