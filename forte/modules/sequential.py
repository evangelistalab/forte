from typing import List, Type, Union

from .module import Module


class Sequential(Module):
    def __init__(
        self, modules: List[Type[Module]] = None, input_modules: Union["Module", List["Module"]] = None, options=None
    ):
        """
        Parameters
        ----------
        modules: List[Type[Module]]
            A list of modules. Defaults to None.
        options: dict
            A dictionary of options. Defaults to None.
        input_modules: Union[Module, List[Module]]
            A single input module or a list of input modules. Defaults to None.
        """
        super().__init__(input_modules, options)
        self.modules = modules or []

    def add(self, module: Module):
        """
        Add a module to the sequential module.
        """
        if self.modules:
            module._input_modules = [self.modules[-1]]
        self.modules.append(module)

    def _run(self, input_data):
        data = input_data
        for module in self.modules:
            data = module.run(data)
        return data

    def __repr__(self):
        return f"Sequential({self.modules})"
