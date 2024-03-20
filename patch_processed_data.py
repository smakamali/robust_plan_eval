# This script takes a processed Query/Plan dataset list and fixes the guideline xml and its encoding. This is needed for datasets generated using code prior to commit: cdd7556

import os
import pickle
from util.util import find_between
from util.base_classes import tabid2tab, binarize_xml_tree
from util.tcnn_util import featurizetree, xmltotree
import xml.etree.ElementTree as ET

def binarize_xml_tree(root):
    if len(root) == 1:
        # Create a new child element for the root
        new_child = ET.Element("Null")
        root.append(new_child)
    
    for elem in root:
        if len(elem) == 1:
            # Create a new child element
            new_child = ET.Element("Null")
            elem.append(new_child)
        else:
            # If the current element has more than one child, recursively call the function
            binarize_xml_tree(elem)

def gltemplatetotree(string,tab_alias_dict):
    gl = find_between(string,'<OPTGUIDELINES>','</OPTGUIDELINES>')
    root = ET.fromstring(gl)
    binarize_xml_tree(root)
    featurizetree(root,tab_alias_dict)
    tree = xmltotree(root)

    return tree


files_id = 'job_syn_p4'
internal_dir = './internal/'

file_path = os.path.join(internal_dir,'labeled_query_plans_{}.pickle'.format(files_id))
patched_file_path = os.path.join(internal_dir,'labeled_query_plans_{}_patched.pickle'.format(files_id))

with open(file_path,'rb') as f:
    queries_list = pickle.load(f)

for idx,query in enumerate(queries_list):
    for hint_id in query.plans.keys():
        print("query",idx,"hint",hint_id) 
        plan = query.plans[hint_id]
        guideline = tabid2tab(plan.guideline, plan.tab_alias_dict)
        query.plans[hint_id].plan_tree = gltemplatetotree(guideline, plan.id_tab)

# write patched file to disk
with open(file_path,'wb') as f:
    pickle.dump(queries_list, f)



