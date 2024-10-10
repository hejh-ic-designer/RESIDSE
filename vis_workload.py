from residse.visualization.graph.dnn import visualize_dnn_graph
from residse.classes.workload.dnn_workload import DNNWorkload

from residse.inputs.WL.resnet18 import workload


workload = DNNWorkload(workload)

visualize_dnn_graph(workload)