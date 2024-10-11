import forte
from forte.modules import Workflow, HF, FCI, Ints, Ints2, Localizer, WorkflowVisualizer
from forte import ForteData

int_mod = Ints()

sub_job = Workflow([Ints2(), int_mod])
job = Workflow([int_mod, FCI(), sub_job])
job.run()

graph = WorkflowVisualizer().visualize(job)
print(graph)
