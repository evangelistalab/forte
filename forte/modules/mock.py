from typing import List, Type, Union

from .module import Module


class Ints(Module):
    def __init__(self, input_modules: Union["Module", List["Module"]] = None):
        super().__init__(input_modules)

    def _run(self, input_data):
        data = input_data
        for input_module in self.input_modules:
            data = input_module.run(data)
        print("Running Ints")


class Ints2(Module):
    def __init__(self, input_modules: Union["Module", List["Module"]] = None):
        super().__init__(input_modules)

    def _run(self, input_data):
        data = input_data
        for input_module in self.input_modules:
            data = input_module.run(data)
        print("Running Ints2")


class HF(Module):
    def __init__(self, input_modules: Union["Module", List["Module"]] = None):
        super().__init__(input_modules)

    def _run(self, input_data):
        data = input_data
        for input_module in self.input_modules:
            data = input_module.run(data)
        print("Running HF")


class Localizer(Module):
    def __init__(self, input_modules: Union["Module", List["Module"]] = None):
        super().__init__(input_modules)

    def _run(self, input_data):
        data = input_data
        for input_module in self.input_modules:
            data = input_module.run(data)
        print("Running Localizer")


class FCI(Module):
    def __init__(self, input_modules: Union["Module", List["Module"]] = None):
        super().__init__(input_modules)

    def _run(self, input_data):
        data = input_data
        for input_module in self.input_modules:
            data = input_module.run(data)
        print("Running FCI")
