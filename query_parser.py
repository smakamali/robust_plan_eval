# what is needed? 
#   - the adjacency matrix
#   - join type
#   - join operator
#   - column cardinalities -> colcard / card
#   - table cardinalities -> colcard / card
#   - column skewness
#   - join cardinality from sample -> join selectivity
#   - distinct values from each side of a join
#   - number of matches between the two sides -> inclusion factor
#   - local predicate selectivites (from either the sample or the optimizer)
#   - pairwise correlation between predicate columns

import os
from sqlglot import parse_one
from sqlglot import exp
from sqlglot.optimizer.scope import build_scope

supported_ops = (exp.EQ,exp.Between,
                exp.LT,exp.GT,
                exp.LTE, exp.GTE, exp.Like)


def parse_query(sql):
    print(sql)

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
        # print(repr(table))
        tb_name = str(table.args['this'])
        if table.find(exp.TableAlias):
            tb_id = str(table.args['alias'])
        else:
            tb_id = str(table.args['this'])
        tables_dict[tb_id]=tb_name

    join_preds = []
    local_preds = []

    print("ON Preds: ")
    for join in ast.find_all(exp.Join):
        for pred in join.find_all(supported_ops):
            join_preds.append(str(pred))
            # print(repr(pred))

    print("Where Preds: ")
    wh = ast.find(exp.Where)
    for pred in wh.find_all(supported_ops):
        if pred.find(exp.Literal):
            local_preds.append(str(pred))
        else:
            join_preds.append(str(pred))
        # print(repr(pred))

    print("Tables:")
    print(tables_dict)

    print("Join Predicate")
    print(join_preds)

    print("Local Predicate:")
    print(local_preds)


# sql="""SELECT MIN(mc.note) AS production_note
# FROM 
#     movie_info_idx AS mi_idx 
#     INNER JOIN movie_companies AS mc 
#         ON mc.movie_id = mi_idx.movie_id
#     INNER JOIN title AS t
#         ON t.id = mc.movie_id,
#     info_type AS it,
#     company_type AS ct
# WHERE 
#     ct.id = mc.company_type_id
#     AND it.id = mi_idx.info_type_id
#     AND ct.kind BETWEEN 'production companies' AND 'z'
#     AND ct.kind like '%production companies%';"""
input_dir = './input'
input_dir_enc = os.fsencode(input_dir)
queries = []
for file in os.listdir(input_dir_enc):
    filename = os.fsdecode(file)
    if filename.endswith(".sql"):
        print(filename)
        with open(os.path.join(input_dir, filename)) as f:
            file_lines = f.readlines()
            file_content = []
            for line in file_lines:
                if line.strip('\n').strip(' ') != '':
                    file_content.append(line)
            file_content=''.join(file_content)
            # print(file_content.upper().split('SELECT '))
            queries.extend(['SELECT '+query for query in file_content.upper().split('SELECT ')[1:]])
print(queries)

for sql in queries:
    parse_query(sql)
    # sql = file.readlines()
    # sql=''.join(sql)
    # print(sql)
    
    # with open('./input/1a.sql') as f:
    #     sql = f.readlines()
    # sql=''.join(sql)
