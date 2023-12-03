import forte
from forte.modules import Sequential, HF, FCI, Ints, Ints2, Localizer, GraphVisualizer
from forte import ForteData

data = ForteData()

# ints = Ints()
# hf = HF(ints)
# graph = GraphVisualizer().visualize(hf)
# print(graph)


ints = Ints()
ints2 = Ints2()
ints3 = Ints2()
seq = Sequential([HF(), Localizer()], [ints, ints2])
fci = FCI([seq, ints3])


graph = GraphVisualizer().visualize(fci)
print(graph)
seq.run(data)

print("\n\n")
Ints().run(data)
