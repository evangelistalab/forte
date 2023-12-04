from typing import List, Type, Union

from .module import Module


class Ints(Module):
    def __init__(self, input_modules: Union["Module", List["Module"]] = None):
        super().__init__(input_modules)

    def _run(self, data: "ForteData") -> "ForteData":
        print("Running Ints")
        return data


class Ints2(Module):
    def __init__(self, input_modules: Union["Module", List["Module"]] = None):
        super().__init__(input_modules)

    def _run(self, data: "ForteData") -> "ForteData":
        print("Running Ints2")
        return data


class HF(Module):
    def __init__(self, input_modules: Union["Module", List["Module"]] = None):
        super().__init__(input_modules)

    def _run(self, data: "ForteData") -> "ForteData":
        print("Running HF")
        return data


class Localizer(Module):
    def __init__(self, input_modules: Union["Module", List["Module"]] = None):
        super().__init__(input_modules)

    def _run(self, data: "ForteData") -> "ForteData":
        print("Running Localizer")
        return data


class FCI(Module):
    def __init__(self, input_modules: Union["Module", List["Module"]] = None):
        super().__init__(input_modules)

    def _run(self, data: "ForteData") -> "ForteData":
        print("Running FCI")
        return data
