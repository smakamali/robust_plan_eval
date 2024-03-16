from sqlglot import parse_one
from sqlglot import exp
from sqlglot.optimizer.scope import build_scope

def nodes_eq(node1, node2):
    if str(node1) == str(node2):
        return True
    return False

def delete_node(node, node_to_remove):
    if nodes_eq(node, node_to_remove):
        return None
    return node

def remove_node(tree,node_to_remove):
    return tree.transform(delete_node,node_to_remove=node_to_remove)

# def node_is_id(node):
#     if isinstance(node,exp.Identifier):
#         print((repr(node)))
#         node.this = node.this.upper()
#     return node

# def tree_upper(tree):
#     return tree.transform(node_is_id)

def upper(input):
    if isinstance(input,dict):
        output = {}
        for key in input:
            output[key.upper()]=input[key].upper()
        return output
    if isinstance(input,list):
        for idx,l in enumerate(input):
            if isinstance(l,list):
                input[idx]=upper(l)
            elif isinstance(l,str):
                input[idx]=l.upper()
            else:
                raise Exception("the input must of list of lists of strings")
        return input

def split_col_list(input):
    res = []
    if isinstance(input,list):
        for item in input:
            res.append(split_col_list(item))
    elif isinstance(input,exp.Column):
        res.extend([str(i) for i in input.parts])
    else:
        res.append(str(input))
    return res

def local_pred_to_upper(input):
    input.this
    print(repr(input))
    return input

def parse_query(sql,verbose=False):
    supported_ops = (exp.Predicate,exp.Unary,exp.Or)

    ast = parse_one(sql)
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

    print("-----------------------------")
    join_preds = []
    local_preds = []
    pred_cols = []

    for join in ast.find_all(exp.Join):
        for pred in join.find_all(supported_ops):
            join_preds.append([pred.this,pred.expression,pred.key])
            for col in pred.find_all(exp.Column):
                pred_cols.append(col)

    wh = ast.find(exp.Where)
    if verbose:
        print("Original Where",str(wh))
    while wh.find(supported_ops):
        pred = wh.find(supported_ops)
        if pred.find(exp.Literal,exp.Null):
            # pred=tree_upper(pred)
            local_preds.append(str(pred))
        else:
            join_preds.append([pred.this,pred.expression,pred.key])

        for col in pred.find_all(exp.Column):
            pred_cols.append(col)
        wh = remove_node(wh,pred)
        if verbose:
            print("------- > Removing", str(pred))
            print("Updated Where",str(wh))
    
    # get unique predicate columns
    pred_cols=list(dict.fromkeys(pred_cols))
    pred_cols = split_col_list(pred_cols)
    join_preds = split_col_list(join_preds)

    return upper(tables_dict),upper(join_preds),(local_preds),upper(pred_cols)

############ THE FOLLOWING LINES FOR TESTING ##############
# sql = """SELECT MIN(mc.note) AS production_note,
#        MIN(t.title) AS movie_title,
#        MIN(t.production_year) AS movie_year
# FROM company_type AS ct
#     INNER JOIN movie_companies AS mc
#     ON ct.id = mc.company_type_id,
#      info_type AS it,
#      movie_info_idx AS mi_idx,
#      title AS t
# WHERE ct.kind = 'production companies'
#   AND it.info = 'top 250 rank'
#   AND mc.note NOT LIKE '%(as Metro-Goldwyn-Mayer Pictures)%'
#   AND (mc.note LIKE '%(co-production)%'
#        OR mc.note LIKE '%(presents)%')
#   AND t.id = mc.movie_id
#   AND t.id = mi_idx.movie_id
#   AND mc.movie_id = mi_idx.movie_id
#   AND it.id = mi_idx.info_type_id;"""
# parse_query(sql,verbose=True)


# print(upper([['sdsA','sadf'],['gdfg','rwer',['erwe']]]))