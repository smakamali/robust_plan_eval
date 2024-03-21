# This script takes a processed Query/Plan dataset list and fixes the guideline xml and its encoding. This is needed for datasets generated using code prior to commit: cdd7556

import os
import pickle
from util.base_classes import tabid2tab, gltemplatetotree

def patch(file_path, patched_file_path):

    with open(file_path,'rb') as f:
        queries_list = pickle.load(f)

    new_queries_list = []
    for idx,query in enumerate(queries_list):
        for hint_id in query.plans.keys():
            print("query",query.q_id,"hint",hint_id) 
            plan = query.plans[hint_id]
            guideline = tabid2tab(plan.guideline, plan.tab_alias_dict)
            query.plans[hint_id].plan_tree = gltemplatetotree(guideline, plan.id_tab)
        new_queries_list.append(query)

    # write patched file to disk
    with open(patched_file_path,'wb') as f:
        pickle.dump(new_queries_list, f)

if __name__ == '__main__':

    files_id = 'job_syn'
    internal_dir = './internal/'

    file_path = os.path.join(internal_dir,'labeled_query_plans_{}.pickle'.format(files_id))
    patched_file_path = os.path.join(internal_dir,'labeled_query_plans_patched_{}.pickle'.format(files_id))

    patch(file_path, patched_file_path)


