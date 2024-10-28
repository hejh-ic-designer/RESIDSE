import pickle
from residse.classes.cost_model.cost_model import CostModelEvaluation

pkl_path = 'outputs/res18_1--resnet18--True/all_cmes.pickle'
# pkl_path = 'outputs/sesr_1--sesr--True/all_cmes.pickle'

with open(pkl_path, 'rb') as f:
    cmes = pickle.load(f)

cmes: list[CostModelEvaluation]

buf_times_edp = []
buf_list = []

for cme in cmes:
    # print(cme.a_buf_size/1024, cme.edp, cme.tile_size)
    buf_times_edp.append((cme.a_buf_size/1024) * cme.edp)
    buf_list.append(cme.a_buf_size)

# print(buf_times_edp)
mini = min(buf_times_edp)

print(mini)
print(buf_times_edp.index(mini))
print(buf_list[buf_times_edp.index(mini)]/1024)