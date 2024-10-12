from residse.visualization.plot_cme import plot_two_lines_ema, plot_two_lines_edp


#= set experiment_id
experiment_id = 'sesr_1--sesr--True--fix_tsize32x32'
# experiment_id = 'sesr_1--sesr--True'
# experiment_id = 'srgan_1--srgan--True--fix_tsize32x32'

# extract path
other_experiment_id = experiment_id.replace("True", "False")
pkl_path_1 = f'outputs/{experiment_id}/all_cmes.pickle'
pkl_path_2 = f'outputs/{other_experiment_id}/all_cmes.pickle'
save_path_ema = f'outputs/{experiment_id}/two_lines_ema.png'
save_path_edp = f'outputs/{experiment_id}/two_lines_edp.png'
# plot
plot_two_lines_ema(pkl_paths=[pkl_path_1, pkl_path_2], save_path=save_path_ema)
plot_two_lines_edp(pkl_paths=[pkl_path_1, pkl_path_2], save_path=save_path_edp)
print(f'##### PLOT DONE @ path: {save_path_ema}')




