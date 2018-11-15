import os
import sys
from glob import glob
from tqdm import tqdm
from replica_learn.evaluation import get_model_folder_at_epoch

skip_processed = False
BASE_MODEL_DIR = '/home/seguin/cluster-nas/wga_experiments'
OUTPUT_DIR = '/home/seguin/feature_maps_indexes'
DATASET = '/home/seguin/experiment_data_wga/dataset_1_training.pkl'

CMD = "python build_index.py -m {model_dir} -o {output_file} --dataset {dataset}  --feature-maps"

model_dirs = glob(os.path.join(BASE_MODEL_DIR, 'small_vgg_*'))
print(model_dirs)

def compute_index(model_dir, output_file, skip_processed):
    if os.path.exists(output_file):
        if skip_processed:
            return
        else:
            print("Deleting {}".format(output_file))
            os.remove(output_file)
    full_cmd = CMD.format(model_dir=model_dir,
                          output_file=output_file,
                          dataset=DATASET)
    os.system(full_cmd)


for model_dir in tqdm(model_dirs):
    output_file = os.path.join(OUTPUT_DIR, os.path.basename(model_dir)+'.hdf5')
    compute_index(model_dir, output_file, skip_processed)

    output_file = os.path.join(OUTPUT_DIR, os.path.basename(model_dir) + '_untrained.hdf5')
    model_dir_2 = get_model_folder_at_epoch(model_dir, 0)
    compute_index(model_dir_2, output_file, skip_processed)
