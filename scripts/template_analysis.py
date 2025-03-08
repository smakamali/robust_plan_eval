import os
import pickle
import numpy as np

files_id = 'dsb_1000'
labeled_data_dir = './labeled_data/dsb/'
template_num_chars = 2
write_to_file = False

def get_num_joins(query):
    num_joins = np.array(query.edge_indc).shape[1]/2
    return num_joins

def get_query_template(query):
    if '_' in query.q_id:
        return query.q_id.split('_')[0]
    else:
        return query.q_id[:template_num_chars]

filename = 'labeled_query_plans_{}.pickle'.format(files_id)
file_path = os.path.join(labeled_data_dir,filename)

print("Loading from ", file_path)
with open(file_path, 'rb') as f:
    try:
        f1 = pickle.load(f)
        print("Successfully loaded pickle file!")
    except EOFError:
        print("File is empty or corrupted.")
    except pickle.UnpicklingError:
        print("File is not a valid pickle format.")

query_ids = [query.q_id for query in f1]

print("Number of queries",len(f1))
print(query_ids)

templates = np.unique(np.array([get_query_template(query) for query in f1]))
print("templates",templates)

num_joins_list = []
for query in f1:
    num_joins_list.append(get_num_joins(query))

num_joins_list = np.array(num_joins_list)
print("num_joins_list statistics:")
print("mean", num_joins_list.mean())
print("std", num_joins_list.std())
print("min", num_joins_list.min())
print("25th percentile", np.percentile(num_joins_list,25))
print("50th percentile", np.percentile(num_joins_list,50))
print("75th percentile", np.percentile(num_joins_list,75))
print("max", num_joins_list.max())
print("distinct num_joins", np.unique(num_joins_list))

# summarize the number of queries and joins for each template
for template in templates:
    msk = (np.array([get_query_template(query) for query in f1]) == template)
    template_queries = np.array(f1)[msk]
    first_query = template_queries[0]
    num_joins = get_num_joins(first_query)
    print(f"template {template}, num_queries {msk.sum()}, num_joins {num_joins}")

# unseen_templates = ["query025","query101"]

# unseen_queries = []
# seen_queries = []
# for template in templates:
#     msk = (np.isin(np.array([get_query_template(query) for query in f1]), template))
#     if template in unseen_templates:
#         unseen_queries.extend(np.array(f1)[msk].tolist())
#     else:
#         seen_queries.extend(np.array(f1)[msk].tolist())

# print("unseen_templates",len(unseen_queries))
# print("seen_templates",len(seen_queries))

# if write_to_file:
#     # write unseen_queries and seen_queries to file in labeled_data_dir
#     filename = 'labeled_query_plans_25_101_{}.pickle'.format(files_id)
#     file_path = os.path.join(labeled_data_dir,filename)
#     with open(file_path,'wb') as f:
#         pickle.dump(unseen_queries,f)

#     filename = 'labeled_query_plans_other_{}.pickle'.format(files_id)
#     file_path = os.path.join(labeled_data_dir,filename)
#     with open(file_path,'wb') as f:
#         pickle.dump(seen_queries,f)
