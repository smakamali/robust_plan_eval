import os
from subprocess import Popen, PIPE
import pandas as pd
import ibm_db
import ibm_db_dbi
from util.util import find_between, load_queries_from_folder
from util.explain_parser import ExplainParser as explain_parser
import xml.etree.ElementTree as ET

def connect_to_db(conn_str, verbose=False):
    ibm_db_conn = ibm_db.connect(conn_str, "", "")
    ibm_db_dbi_conn = ibm_db_dbi.Connection(ibm_db_conn)
    db_conn_exists = 'ibm_db_conn' in locals() or 'ibm_db_conn' in globals()
    if db_conn_exists and verbose:
        print('Connected to the database!')
    return ibm_db_conn, ibm_db_dbi_conn

def close_connection_to_db(ibm_db_conn, verbose = False):
    rc = ibm_db.close(ibm_db_conn)
    if verbose:
        if rc:
            print('Connection is closed.')
        else:
            print('Closing the connection failed or connection did not exist.')
    return rc

def get_card_sel(conn_str):

    ibm_db_conn, ibm_db_dbi_conn = connect_to_db(conn_str, verbose=False)

    exp_sql = """

    -- Branch for non-index-scan operators (table scans, joins, group-by, etc.)
    SELECT 
        EO.OPERATOR_ID,
        EO.OPERATOR_TYPE,
        MAX(ESO.STREAM_COUNT) AS OUTPUT_CARD,
        CASE 
        -- For join operators, compute join selectivity as:
        -- output / (outer_input * inner_input)
        WHEN EO.OPERATOR_TYPE IN ('HSJOIN','NLJOIN') 
            AND MAX(ESI.STREAM_COUNT) IS NOT NULL 
            AND MIN(ESI.STREAM_COUNT) IS NOT NULL THEN
            MAX(ESO.STREAM_COUNT) / ( MAX(ESI.STREAM_COUNT) * MIN(ESI.STREAM_COUNT) )
        -- For other operators, use: selectivity = output / input
        WHEN MAX(ESI.STREAM_COUNT) <> 0 THEN MAX(ESO.STREAM_COUNT) / MAX(ESI.STREAM_COUNT)
        ELSE NULL 
        END AS SELECTIVITY,
        EO.TOTAL_COST,
        MAX(ESI.OBJECT_NAME) AS TABLE_NAME,
        -- LEFT_INPUT: the operator id from the first input stream row.
        (SELECT MIN(t.SOURCE_ID)
        FROM SYSTOOLS.EXPLAIN_STREAM t
        WHERE t.TARGET_ID = EO.OPERATOR_ID) AS LEFT_INPUT,
        -- RIGHT_INPUT: if more than one input exists, take the other operator id; otherwise NULL.
        (SELECT CASE WHEN COUNT(*) > 1 THEN MAX(t.SOURCE_ID) ELSE NULL END
        FROM SYSTOOLS.EXPLAIN_STREAM t
        WHERE t.TARGET_ID = EO.OPERATOR_ID) AS RIGHT_INPUT
    FROM SYSTOOLS.EXPLAIN_OPERATOR EO 
    LEFT JOIN SYSTOOLS.EXPLAIN_STREAM ESI 
        ON ESI.TARGET_ID = EO.OPERATOR_ID
    LEFT JOIN SYSTOOLS.EXPLAIN_STREAM ESO 
        ON ESO.SOURCE_ID = EO.OPERATOR_ID
    WHERE EO.OPERATOR_TYPE IN ('TBSCAN', 'HSJOIN', 'NLJOIN', 'MSJOIN', 'GRPBY')
    GROUP BY EO.OPERATOR_ID,
            EO.OPERATOR_TYPE,
            EO.TOTAL_COST

    UNION ALL

    -- Branch for index scans: collapse the IXSCAN and use the corresponding FETCH operator’s id.
    SELECT 
        -- Use the FETCH operator's id as the operator id for the index scan.
        FOP.OPERATOR_ID AS OPERATOR_ID,
        'IXSCAN' AS OPERATOR_TYPE,
        MAX(IX_OUT.STREAM_COUNT) AS OUTPUT_CARD,
        MAX(IX_OUT.STREAM_COUNT) / MAX(IX_IN.STREAM_COUNT) AS SELECTIVITY,
        MAX(FOP.TOTAL_COST) AS TOTAL_COST,
        MAX(FES.OBJECT_NAME) AS TABLE_NAME,
        MIN(LEFT_STREAM.SOURCE_ID) AS LEFT_INPUT,
        CASE WHEN COUNT(LEFT_STREAM.SOURCE_ID) > 1 THEN MAX(LEFT_STREAM.SOURCE_ID) ELSE NULL END AS RIGHT_INPUT
    FROM SYSTOOLS.EXPLAIN_OPERATOR O_IX
    -- Get the input and output cardinalities from the IXSCAN operator’s streams.
    JOIN SYSTOOLS.EXPLAIN_STREAM IX_IN 
        ON IX_IN.TARGET_ID = O_IX.OPERATOR_ID
    JOIN SYSTOOLS.EXPLAIN_STREAM IX_OUT 
        ON IX_OUT.SOURCE_ID = O_IX.OPERATOR_ID
    -- Join to obtain the input operator ids for the IXSCAN operator.
    JOIN SYSTOOLS.EXPLAIN_STREAM LEFT_STREAM
        ON LEFT_STREAM.TARGET_ID = O_IX.OPERATOR_ID
    -- Locate the linking row that connects the IXSCAN operator to its FETCH operator.
    JOIN SYSTOOLS.EXPLAIN_STREAM LINK
        ON LINK.SOURCE_ID = O_IX.OPERATOR_ID
    -- Join to the FETCH operator; this row supplies the operator id, cost, etc.
    JOIN SYSTOOLS.EXPLAIN_OPERATOR FOP 
        ON FOP.OPERATOR_ID = LINK.TARGET_ID
        AND FOP.OPERATOR_TYPE = 'FETCH'
    -- Join again to EXPLAIN_STREAM to obtain the table name from the FETCH row.
    JOIN SYSTOOLS.EXPLAIN_STREAM FES 
        ON FES.TARGET_ID = FOP.OPERATOR_ID
        AND FES.OBJECT_NAME IS NOT NULL
    WHERE O_IX.OPERATOR_TYPE = 'IXSCAN'
    GROUP BY FOP.OPERATOR_ID
        
    ORDER BY OPERATOR_ID;

    """

    exp_res = pd.read_sql(exp_sql,ibm_db_dbi_conn)
    print("exp_res:\n",exp_res)

    _ = close_connection_to_db(ibm_db_conn, verbose = False)
    return exp_res


def get_last_plan_cost(conn_str):

    if conn_str is None:
        with open("conn_str", "r") as conn_str_f:
            conn_str = conn_str_f.read()

    ibm_db_conn, ibm_db_dbi_conn = connect_to_db(conn_str)

    sql = "select total_cost from systools.explain_operator where operator_type = 'RETURN'"
    
    plan_cost = pd.read_sql(sql,ibm_db_dbi_conn)
    print("plan_cost",plan_cost)
    plan_cost = plan_cost.values.squeeze()
    _=close_connection_to_db(ibm_db_conn)

    return plan_cost

def db2_explain(schema_name,sql,query_id,
    hintset=None, hintset_id=None,
    opt_plan_path='./optimizer_plans', gen_exp_output=False, 
    gen_guideline=False, return_cost = False, conn_str=None, cmd_verbose=False):
    
    if hintset_id is None and gen_guideline:
        hintset_id = 'defualt'

    if not os.path.isdir(opt_plan_path):
        os.mkdir(opt_plan_path)
    file_path = os.path.join(opt_plan_path,'temp.sql')

    stmt= "explain plan for {}\n".format(sql)
    with open(file_path,'w') as f:
        f.write(stmt)
    
    # cmd='~/db2cserv -d on\n'
    # cmd+='db2stop force'
    # cmd+='db2start'
    cmd='db2 connect to {};\n'.format(schema_name)
    cmd+='db2 set current schema {};\n'.format(schema_name)
    cmd+='db2 .opt set enable nljn;\n'
    cmd+='db2 delete from systools.explain_instance;\n'
    cmd+='db2 commit;\n'
    
    if isinstance(hintset,list):
        cmd += ''.join(hintset)
    elif isinstance(hintset,str):
        cmd += hintset
    
    if gen_guideline:
        cmd+= 'db2 .opt set lockplan on my.guideline;\n'

    cmd+='db2 -tvf {};\n'.format(file_path)

    if gen_guideline:
        guidlineFile = os.path.join(opt_plan_path,'query#{}-{}guide'.format(query_id,hintset_id))
        cmd+='db2 .opt set lockplan off;\n'
        cmd+= 'cp ~/sqllib/my.guideline {}\n'.format(guidlineFile)

    if gen_exp_output:
        exp_name = os.path.join(opt_plan_path,'query#{}-{}.ex'.format(query_id, hintset_id))
        cmd+='db2exfmt -d {} -1 -o {};\n'.format(schema_name, exp_name)

    cmd+='connect reset;\n'
    
    process = Popen( "/bin/bash", shell=False,
                    universal_newlines=True,
                    stdin=PIPE, stdout=PIPE, stderr=PIPE)
    
    output,_ = process.communicate(cmd)
    
    if cmd_verbose:
        cmds = cmd.split('\n')
        for cmd in cmds:
            print('>> ',cmd)
        print(output)

    if return_cost:
        plan_cost = get_last_plan_cost(conn_str)

    result =[]

    if gen_exp_output:
        result.append(exp_name)
    else:
        result.append(None)
    if gen_guideline:
        with open(guidlineFile,'r') as f:
            guideline = f.read()
            guideline = find_between(guideline, '<OPTGUIDELINES>', '</OPTGUIDELINES>').replace('\n','').replace('\t','')
        result.append(guideline)
    else:
        result.append(None)
    if return_cost:
        result.append(plan_cost)
    else:
        result.append(None)

    return tuple(result)

def get_default_plan(sql,q_id,schema_name,conn_str,opt_plan_path):
    schema_name=schema_name.upper()

    exp_path,guideline,plan_cost=db2_explain(schema_name,sql,query_id=q_id,opt_plan_path=opt_plan_path,gen_exp_output=True,gen_guideline=True, return_cost=True,cmd_verbose=True, conn_str=conn_str)
    
    res = get_card_sel(conn_str)
    
    return exp_path, guideline, plan_cost, res


def annotate_guideline(exp_res: pd.DataFrame, guideline: str, tab_alias_dict: dict) -> str:
    # Convert the dataframe to a list of dictionaries sorted by OPERATOR_ID.
    rows = exp_res.sort_values("OPERATOR_ID").to_dict(orient="records")
    
    def get_matching_row(tag: str, node_attrs: dict):
        """
        Find and remove the first row in 'rows' matching the given tag.
        For TBSCAN and IXSCAN nodes, if a TABID attribute is present and found in tab_alias_dict,
        enforce a TABLE_NAME match; for all other operators (even if they have a TABID),
        match solely on operator type.
        """
        for i, row in enumerate(rows):
            # Check if the operator type matches.
            if row["OPERATOR_TYPE"].strip() != tag.strip():
                continue

            # Only enforce table name matching for TBSCAN and IXSCAN.
            if tag in {"TBSCAN", "IXSCAN"} and "TABID" in node_attrs:
                alias = node_attrs["TABID"]
                expected_table = tab_alias_dict.get(alias)
                # If the alias is found in the dictionary, then require a match.
                if expected_table is not None and row.get("TABLE_NAME") != expected_table:
                    continue

            # Debug print if needed:
            # print(f"Matched node {tag} with row: {row}")
            return rows.pop(i)
        # No match found.
        # Debug print if needed:
        # print(f"No match found for node {tag} with attributes {node_attrs}")
        return None

    def format_value(val):
        """Format a numeric value to six decimal places if it is a float; otherwise, return as string."""
        if isinstance(val, float):
            return f"{val:.6f}"
        return str(val)
    
    def annotate_node(node: ET.Element):
        tag = node.tag
        # Attempt to get a matching row for this node.
        matching_row = get_matching_row(tag, node.attrib)
        if matching_row is not None:
            node.set("OUTPUT_CARD", format_value(matching_row["OUTPUT_CARD"]))
            node.set("SELECTIVITY", format_value(matching_row["SELECTIVITY"]))
            node.set("TOTAL_COST", format_value(matching_row["TOTAL_COST"]))
        # Otherwise, we leave the node unannotated (or optionally, log a warning).
        
        # Recursively annotate child nodes.
        for child in node:
            annotate_node(child)
    
    # Parse the guideline XML.
    root = ET.fromstring(guideline)
    annotate_node(root)
    
    # Return the annotated XML as a unicode string.
    annotated_xml = ET.tostring(root, encoding="unicode")
    return annotated_xml

def default_compile(sql,q_id,schema,opt_plan_path='./optimizer_plans/'):
    conn_str_path='./conn_str'
    with open(conn_str_path, "r") as conn_str_f:
        conn_str = conn_str_f.read()

    # get the default plan
    exp_path, guideline, plan_cost, exp_res= get_default_plan(sql,q_id,schema,conn_str,opt_plan_path)

    # parse the explain to get the mapping between aliases used in the guidelines and the table names
    parsed_exp = explain_parser(open(exp_path,'r').read()).parse()
    tab_alias_dict = parsed_exp['tab_alias_dic']

    print("guideline\n",guideline)
    print("plan_cost\n",plan_cost)
    print("tab_alias_dict\n",tab_alias_dict)

    # Annotate the entire guideline tree.
    annotated_guideline = annotate_guideline(exp_res, guideline, tab_alias_dict)

    print("guideline\n",annotated_guideline)
    

if __name__ == '__main__':
    input_dir = './input/temp'
    queries, query_ids=load_queries_from_folder(input_dir)

    for idx,sql in enumerate(queries):
        default_compile(sql, query_ids[idx], "IMDB")
        break

