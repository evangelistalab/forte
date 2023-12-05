import forte
from forte.modules import Sequential, HF, FCI, Ints, Ints2, Localizer, GraphVisualizer
from forte import ForteData

data = ForteData()

ints = Ints()
ints2 = Ints2()
seq = Sequential([HF(), Localizer()], ints)
fci = FCI(seq)

graph = GraphVisualizer().visualize(fci)
print(graph)
fci.run(data)
