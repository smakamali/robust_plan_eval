############### CONSOLIDATE DATA FILES ###############
# Use this script to consolidate multiple labeled data files into a single file
# It assumes all files inside labeled_data_dir are input data files
# Optionally patches the data files before consolidation

import os
import pickle
from util.patch_processed_data import patch

labeled_data_dir = './labeled_data/syn/'
output_file_id = 'job_syn_all'
patch_flag = True

output_path = os.path.join(labeled_data_dir,'labeled_query_plans_{}.pickle'.format(output_file_id))

queries_lists = []
labeled_data_dir_enc = os.fsencode(labeled_data_dir)

for file in os.listdir(labeled_data_dir_enc):
    
    filename = os.fsdecode(file)
    file_path = os.path.join(labeled_data_dir,filename)
    
    # patch files if needed
    if patch_flag:
        print("Patching ", file_path)
        patch(file_path,file_path)

    # load query lists of files and append to the consolidated list
    print("Loading from ", file_path)
    with open(file_path,'rb') as f:
        queries_lists.extend(pickle.load(f))

# write patched file to disk
print("Writing to ", output_path)
with open(output_path,'wb') as f:
    pickle.dump(queries_lists, f)



