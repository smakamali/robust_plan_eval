import os
from sqlglot import parse_one
from sqlglot import exp
from sqlglot.optimizer.scope import build_scope

def nodes_eq(node1, node2):
    if str(node1) == str(node2):
        return True
    return False

def transformer(node, node_to_remove):
    if nodes_eq(node, node_to_remove):
        return None
    return node

def remove_node(tree,node_to_remove):
    return tree.transform(transformer,node_to_remove=node_to_remove)


def parse_query(sql,verbose=False):
    supported_ops = (exp.EQ, exp.NEQ, exp.Between,
                    exp.LT,exp.GT, exp.LTE, 
                    exp.GTE, exp.Like, exp.In,
                    exp.Is,exp.Not,exp.Or)

    ast = parse_one(sql)
    # print(repr(ast))
    root = build_scope(ast)

    tables = [
        source

        # Traverse the Scope tree, not the AST
        for scope in root.traverse()

        # `selected_sources` contains sources that have been selected in this scope, e.g. in a FROM or JOIN clause.
        # `alias` is the name of this source in this particular scope.
        # `node` is the AST node instance
        # if the selected source is a subquery (including common table expressions), 
        #     then `source` will be the Scope instance for that subquery.
        # if the selected source is a table, 
        #     then `source` will be a Table instance.
        for alias, (node, source) in scope.selected_sources.items()
        if isinstance(source, exp.Table)
    ]

    tables_dict = {}
    for table in tables:
        tb_name = str(table.args['this'])
        if table.find(exp.TableAlias):
            tb_id = str(table.args['alias'])
        else:
            tb_id = str(table.args['this'])
        tables_dict[tb_id]=tb_name

    join_preds = []
    local_preds = []

    for join in ast.find_all(exp.Join):
        for pred in join.find_all(supported_ops):
            join_preds.append(str(pred))

    wh = ast.find(exp.Where)
    if verbose:
        print("Original Where",str(wh))
    while wh.find(supported_ops):
        pred = wh.find(supported_ops)
        if pred.find(exp.Literal,exp.Null):
            local_preds.append(str(pred))
        else:
            join_preds.append(str(pred))
        # if pred.find((exp.Not)):
        wh = remove_node(wh,pred)
        if verbose:
            print("------- > Removing", str(pred))
            print("Updated Where",str(wh))
    
    return tables_dict,join_preds,local_preds

input_dir = './input'
input_dir_enc = os.fsencode(input_dir)

queries = []
query_ids = []

for file in os.listdir(input_dir_enc):
    filename = os.fsdecode(file)
    if filename.endswith(".sql"):
        query_ids.append(filename)
        with open(os.path.join(input_dir, filename)) as f:
            file_lines = f.readlines()
            file_content = []
            for line in file_lines:
                if line.strip('\n').strip(' ') != '':
                    file_content.append(line)
            file_content=''.join(file_content)
            queries.extend(['SELECT '+query for query in file_content.upper().split('SELECT ')[1:]])

for idx,sql in enumerate(queries): 
    tables_dict,join_preds,local_preds=parse_query(sql)
    
    print("Query ID: ",query_ids[idx])
    print(sql)
    
    print("Tables {<alias>:<table_name>}:")
    print(tables_dict)

    print("Join Predicate:")
    print(join_preds)

    print("Local Predicate:")
    print(local_preds)