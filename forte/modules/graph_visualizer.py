from .module import Module


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
