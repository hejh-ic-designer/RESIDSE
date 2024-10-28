import pickle
from residse.classes.cost_model.cost_model import CostModelEvaluation

# pkl_path = 'outputs/res18_1--resnet18--True--fix_tsize4x2/all_cmes.pickle'
# pkl_path = 'outputs/sesr_1--sesr--True--fix_tsize32x4/all_cmes.pickle'
pkl_path = 'outputs/srgan_1--srgan--True--fix_tsize32x4/all_cmes.pickle'

with open(pkl_path, 'rb') as f:
    cmes = pickle.load(f)

cmes: list[CostModelEvaluation]

buf_times_edp = []
buf_list = []
edps = []

for cme in cmes:
    if cme is None:
        continue
    # print(cme.a_buf_size/1024, cme.edp, cme.tile_size)
    buf_times_edp.append((cme.a_buf_size/1024) * cme.edp)
    buf_list.append(cme.a_buf_size)
    edps.append(cme.edp)

# print(buf_times_edp)
mini = min(buf_times_edp)
id = buf_times_edp.index(mini)
mini_buf_size = buf_list[id]/1024   # 固定的 buffer size
print(mini)
print('minimal buf index is: ', id)
print('buffer size is: ', mini_buf_size)
print('edp of mini is: ', edps[id])


# find edp in not merge
not_merge_pkl = pkl_path.replace('True', 'False')
with open(not_merge_pkl, 'rb') as ff:
    n_cmes: list[CostModelEvaluation] = pickle.load(ff)
for n_cme in n_cmes:
    if n_cme is None:
        continue
    if n_cme.a_buf_size/1024 == mini_buf_size:
        print('in not merge, edp of mini buf size is: ', n_cme.edp)


# find edp in free tile size
free_pkl = pkl_path.replace('--fix_tsize32x4', '')   #! 注意这里的 ?x? 要和上面的pkl_path匹配
with open(free_pkl, 'rb') as fff:
    f_cmes: list[CostModelEvaluation] = pickle.load(fff)
for f_cme in f_cmes:
    if f_cme is None:
        continue
    if f_cme.a_buf_size/1024 == mini_buf_size:
        print('in free tsize, edp of mini buf size is: ', f_cme.edp)
