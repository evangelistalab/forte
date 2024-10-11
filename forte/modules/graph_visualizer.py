from typing import Union

from .module import Module
from .workflow import Workflow


class GraphVisualizer:
    """
    Handles the visualization of the computational graph.
    """

    def visualize(self, root_module: Module):
        """
        Creates a visual representation of the computational graph.
        """
        return self._make_graph(root_module)

    def _make_graph(self, module: Module, level: int = 0, prefix=None, last=False):
        graph = []
        prefix = [] if prefix is None else prefix

        if level == 0:
            graph.append(repr(module))
        else:
            connector = "└──" if last else "├──"
            graph.append("".join(prefix[: level - 1]) + f"{connector}{repr(module)}")

        prefix.append("│  " if len(module.input_modules) > 1 else "   ")

        for k, input_module in enumerate(module.input_modules):
            is_last = k == len(module.input_modules) - 1
            prefix[-1] = "   " if is_last else prefix[-1]
            graph.append(self._make_graph(input_module, level + 1, prefix, is_last))

        prefix.pop()
        return "\n".join(graph)


class WorkflowVisualizer:
    """
    Handles the visualization of the computational graph.
    """

    def visualize(self, root: Union[Module, Workflow]):
        """
        Creates a visual representation of the computational graph.
        """
        graph_lines = []
        self._make_graph(root, graph_lines, is_first=True)
        return "\n".join(graph_lines)

    def _make_graph(self, module: Union[Module, Workflow], graph_lines, prefix="", is_last=True, is_first=False):
        if is_first:
            graph_lines.append("Job")
        connector = "└── " if is_last else "├── "
        graph_lines.append(prefix + connector + repr(module))

        # Update the prefix for child nodes
        if is_last:
            prefix += "    "
        else:
            prefix += "│   "

        # Determine if the node is a Workflow or a Module
        if isinstance(module, Workflow):
            child_modules = module.modules
        else:
            # Modules do not have child nodes
            return

        # Iterate over child nodes
        for idx, child_module in enumerate(child_modules):
            is_module_last = idx == len(child_modules) - 1
            self._make_graph(child_module, graph_lines, prefix, is_module_last)
