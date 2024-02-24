### ----------------------------------- v3 ----------------------------------- ###
import re
# from table_stats_parser import absoluteFilePaths

# Helper function to return 'n' sized chunks from 'l'
def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

# Helper function to return string between 'first' and 'last' in 's'
def find_between(s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""
 
# Helper function to populate 'ojv' given an access plan 'tree' starting an 'index' 0
# Implements recursive calls
def populate_ojv(ojv, tree, index, tab_alias_dic):
    if "TBSCAN" in tree['data']['operation'] and index != 0:
        if tree['left']['data']['est_cost'] not in tab_alias_dic.keys():
            populate_ojv(ojv, tree['left'], index, tab_alias_dic)
        else:
            ojv[index] = tree['left']['data']['id']
            ojv[index+1] = tree['data']['rows']
    # elif "IXSCAN" in tree['data']['operation'] and index != 0:
    #     if "TEMP" in tree['left']['data']['operation']:
    #         populate_ojv(ojv, tree['left'], index, tab_alias_dic)
    #     else:
    #         # print(tab_alias_dic)
    #         ojv[index] = tab_alias_dic[tree['left']['data']['est_cost']]
    #         ojv[index+1] = tree['data']['rows']
    elif "JOIN" in tree['data']['operation']:
        # adding the join operation
        ojv[index] = tree['data']['operation']
        # adding the cardinality of the join
        ojv[index+1] = tree['data']['rows']
        populate_ojv(ojv, tree['left'], index * 2 + 2, tab_alias_dic) 
        populate_ojv(ojv, tree['right'], index * 2 + 4, tab_alias_dic)
    else:
        populate_ojv(ojv, tree['left'], index, tab_alias_dic)

class ExplainParser:

    def __init__(self, raw_text):
        self.raw_text = raw_text

    #returns a hash with parsed values from a long explain as retrieved from db2
    def parse(self):
        # initialize JSON
        self.json = {}

        # Load original statment
        cut_text = self.raw_text.rsplit('Original Statement:\n------------------',-1)[1]
        og_statment = cut_text.rsplit('Optimized Statement:',-1)[0]
        og_statment = og_statment.strip().split('/')[0]
        # print("og_statment: ", og_statment)
        # print("")
        self.json['raw_sql'] = og_statment
        
        # Load optimized statment
        cut_text = self.raw_text.rsplit('Optimized Statement:\n-------------------',-1)[1]
        op_statment = cut_text.rsplit('Access Plan:',-1)[0]
        op_statment = op_statment.strip().split('/')[0]
        self.json['opt_statement'] = op_statment
        # print("op_statment: ", op_statment)
        # print("")
        
        # Create a dictionary of table aliases
        cut_text = op_statment.rsplit("FROM",-1)[1]
        cut_text = cut_text.rsplit("WHERE",-1)[0]
        cut_text = cut_text.replace('\n','')
        # print("----------->",cut_text)
        tab_raw = cut_text.split(',')
        tab_alias_dic = {}
        for line in tab_raw:
            line = line.split(".")[1]
            tab_alias_dic[str(line.split(" AS ")[1]).strip()] = str(line.split(" AS ")[0]).strip()
        # print("tab_alias_dic: ",tab_alias_dic)
        self.json['tab_alias_dic'] = tab_alias_dic
        # self.json['raw_sql'] = og_statment

        # Load total cost
        cut_text = self.raw_text.rsplit('Access Plan:\n-----------', -1)[1]
        raw_cost = cut_text.strip().split('\n')[0]
        self.json['total_cost'] = raw_cost.split(':')[1].strip()
        # Load access plan
        cut_text = self.raw_text.rsplit('Access Plan:\n-----------', -1)[1]
        raw_access_plan = "\n".join(cut_text.split('\n')[3:])
        raw_access_plan = raw_access_plan.rsplit('Operator Symbols :\n---------------',-1)[0].rstrip()
        raw_access_plan = raw_access_plan.rsplit('Extended Diagnostic Information:\n--------------------------------',-1)[0].rstrip()
        raw_access_plan = raw_access_plan.replace('(', ' ').replace(')', ' ')
        # print("raw_access_plan")
        # print(raw_access_plan)
        # Filter out the rows that contains nothing for the later chunking process
        split_access_plan = list(filter(None, raw_access_plan.split('\n')))
        # print("split_access_plan: ")
        # print(split_access_plan)
        # Grouping every 6 rows because each box except the base tables are represented
        # in the graph as 5 rows and a '|'
        # Example: 
        #                       |
        #                  1.72299e+06
        #                    HSJOIN
        #                    (   5)
        #                    4015.23
        #                      55
        #            /---------+----------\
        chunked_access_plan = list(chunks(split_access_plan, 6))
        # Splitting by two spaces, i.e. '  ' is to break up each row to boxes, and not
        # to split up strings with spaces naturally
        chunked_access_plan = [ [ j.split('  ') for j in i ] for i in chunked_access_plan ]
        chunked_access_plan = [ [ [k.strip() for k in j] for j in i ] for i in chunked_access_plan ]
        chunked_access_plan = [ [ list(filter(None, j)) for j in i ] for i in chunked_access_plan ]
        # print("chunked_access_plan:")
        # print(chunked_access_plan)
        nodes = []
        tree_struct = None
        tree_struct_count = 0
        for i in chunked_access_plan:
            # len(i[0]) stands for number of boxes at this level
            # Differenct boxes in on the same level are separated at this stage
            for j in range(len (i[0])):
                # i[3] stands for the list of cost
                # if it is 'Q' instead of cost, it's the base tables, 
                # therefore, set tree_struct to None
                if 'Q' in i[3][j]:
                    tree_struct = None
                else:
                    tree_struct = i[5][tree_struct_count]
                    tree_struct_count += 1
                nodes += [{ 'rows': i[0][j], 'operation': i[1][j], 'id': i[2][j], 'est_cost': i[3][j], 'tree_struct': tree_struct}]
            tree_struct_count = 0
        access_plan = {'data': nodes[0], 'left': None, 'right': None}
        # Remove information of the top box from the list 'nodes'
        nodes.pop(0)
        # Dictionaries are used as references in list, so the changes made to 
        # elements of queued_nodes will be applied to the nodes even if they 
        # are removed from the list queued_nodes
        queued_nodes = [access_plan]
        for node in nodes:
            new_node = {'data': node, 'left': None, 'right': None}
            # Add new node to 'left' as default, remove old node from list queued_nodes
            # New node is added to list queued_nodes if it's not base table
            # List queued_nodes is implemented as an probe here
            if queued_nodes[0]['data']['tree_struct'] == '|':
                queued_nodes[0]['left'] = new_node
                queued_nodes.pop(0)
                # If tree_struct is None, then it means it's base table
                if bool(new_node['data']['tree_struct']):
                    queued_nodes += [new_node]
                    #print('!!!! these are not interested, tree_structure !!!!')
                    #print(queued_nodes)
            # If the probe points to a box that branches on a lower level
            # Add first branch to 'left', add second branch to 'right'
            # Remove current node after both branches are added
            # The code deals with all left branches first and goes back to 
            # deal with right branches on previous levels as chronicle order
            elif '-+-' in queued_nodes[0]['data']['tree_struct']:
                # If 'left' has already been assigned value, assign value
                # to 'right' and remove previous node
                if bool(queued_nodes[0]['left']):
                    queued_nodes[0]['right'] = new_node
                    queued_nodes.pop(0)
                    if bool(new_node['data']['tree_struct']):
                        #print('**** Another addition to the tree structure ****')
                        #print(queued_nodes)
                        queued_nodes += [new_node]
                else:
                    queued_nodes[0]['left'] = new_node
                    if bool(new_node['data']['tree_struct']):
                        #print('2222 Another type of nodes 222222')
                        #print([new_node])
                        queued_nodes += [new_node]
        self.json['access_plan'] = access_plan
        # print("access_plan")
        # print(access_plan)
        #------------- END of CHECKING --------------------------------#
        # Load OJV (Order Join Vector)
        # ojv = [None] * 1024 # Current tree size of max = sum(2^i) for i in range(0,max_join_num)
        # populate_ojv(ojv, access_plan, 0, tab_alias_dic)
        # # print('_____________OJV______________')
        # # print(ojv)
        # self.json['ojv'] = ojv
        # Load time taken for a given query 
        try:
            cut_text = self.raw_text.rsplit('ZWallClock',-1)[1]
            time = float(cut_text.split(' ')[2])
            self.json['time'] = time
        except:
            self.json['time'] = 0
        
        # Parse out columns, tables, and predicates involved
        raw_columns_queried = find_between(self.json['raw_sql'], 'SELECT', 'FROM').strip()
        # print(raw_columns_queried)
        try:
            columns_queried = [x.strip().split('.')[2].replace(',','') for x in raw_columns_queried.split('\n')    ]
        except IndexError: 
            columns_queried = []

        tables_queried = [tab_alias_dic[i] for i in tab_alias_dic.keys()]

        # raw_predicates = self.json['raw_sql'].split('FROM')[1].strip()
        # if '\n' in raw_predicates:
        #     raw_predicates = re.sub('\n','',raw_predicates)
        #if '/*' in raw_predicates:
        #    print('@@@@@@@@')
        #    raw_predicates = re.sub('/.*/','',raw_predicates)
        #split_predicates = raw_predicates.split(' ')
        # split_predicates = raw_predicates
        # print('________split_predicates', split_predicates)
        # try:
        #     thing_index = split_predicates.index('join')
        #     tables_queried = [split_predicates[thing_index - 1]]
        # except ValueError:
        #     tables_queried = []
        
        # indices = [j + 1 for j, x in enumerate(split_predicates) if x == 'join' ]
        # for i in indices:
        #     tables_queried += [split_predicates[i]]
        # predicates = []
        # indices = [j for j, x in enumerate(split_predicates) if x == '=' ]
        # for i in indices:
        #     pred_a = split_predicates[i-1]
        #     pred_b = split_predicates[i+1]
        #     predicates += [{'col_a': pred_a,
        #                     'col_b': pred_b,
        #                     'operator': '='}]

        self.json['tables_queried'] = tables_queried
        self.json['columns_queried'] = columns_queried
        # self.json['predicates'] = predicates
        # parse out heap usage
        try:
            sort_mem_usage = 0
            cut_text = self.raw_text.rsplit('ZWallClock', -1)[1]
            for line in cut_text.split("\n"):
                if "consumer" in line:
                    temp = line.split(' ')
                    sort_mem_usage += int(max(temp[4], temp[6]))

            self.json['sort_mem_usage'] = sort_mem_usage
        except IndexError:
            self.json['sort_mem_usage'] = 0

        return self.json

