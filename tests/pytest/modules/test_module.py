import forte
from forte.modules import Sequential, HF, FCI, Ints, Ints2, Localizer, GraphVisualizer
from forte import ForteData

data = ForteData()

ints = Ints()
ints2 = Ints2()
seq = Sequential([Localizer(), Localizer()], ints)
fci = FCI(seq)

graph = GraphVisualizer().visualize(fci)
print(graph)
data = fci.run(data)
