import os
from subprocess import Popen, PIPE
import time
import numpy as np
import pandas as pd
import ibm_db
import ibm_db_dbi
import subprocess as sp
from util.util import find_between
import xml.etree.ElementTree as ET

def connect_to_db(conn_str, verbose=False):
    try:
        ibm_db_conn = ibm_db.connect(conn_str, "", "")
        ibm_db_dbi_conn = ibm_db_dbi.Connection(ibm_db_conn)
        db_conn_exists = 'ibm_db_conn' in locals() or 'ibm_db_conn' in globals()
        if db_conn_exists and verbose:
            print('Connected to the database!')
        return ibm_db_conn, ibm_db_dbi_conn
    except:
        if verbose:
            print('Connection to the database could not be established.')
        return None, None

def close_connection_to_db(ibm_db_conn, verbose = False):
    rc = ibm_db.close(ibm_db_conn)
    if verbose:
        if rc:
            print('Connection is closed.')
        else:
            print('Closing the connection failed or connection did not exist.')
    return rc

def create_explain_tables(schema_name,cmd_verbose=False):
    
    dir='./temp'
    
    if not os.path.isdir(dir):
        os.mkdir(dir)
    file_path = os.path.join(dir,'temp.sql')
    
    process = Popen( "/bin/bash", shell=False,
                    universal_newlines=True,
                    stdin=PIPE, stdout=PIPE, stderr=PIPE)
        
    stmt= """connect to {};
CALL SYSPROC.SYSINSTALLOBJECTS('EXPLAIN', 'C', 
        CAST (NULL AS VARCHAR(128)), CAST (NULL AS VARCHAR(128)));
commit;
connect reset;""".format(schema_name)

    with open(file_path,'w') as f:
        f.write(stmt)
    
    cmd='db2 -tvf {};\n'.format(file_path)
    
    output,_ = process.communicate(cmd)
    
    if cmd_verbose:
        cmds = cmd.split('\n')
        for cmd in cmds:
            print('>> ',cmd)
    print(output)

def load_db_schema(schema_name, conn_str):
        # database using the provided schema name and connection string. It returns a dictionary where the keys are table names, and the values are lists of column names for each table.
    schema_name=schema_name.upper()
    ibm_db_conn, ibm_db_dbi_conn = connect_to_db(conn_str, verbose=True)

    tab_card_dict = load_tab_card(schema_name, conn_str)
    table_list = np.array(list(tab_card_dict.keys()))
    tab_sizes = np.array([int(tab_card_dict[tab]) for tab in tab_card_dict])
    table_list = table_list[np.flip(np.argsort(tab_sizes))]

    table_dict = {}
    for table in table_list:
        sql = "Select trim(c.tabschema) || '.' || trim(c.tabname) || '.' || trim(c.colname) as column_name from syscat.columns c inner join syscat.tables t on t.tabschema = c.tabschema and t.tabname = c.tabname where t.type = 'T' and c.tabschema = \'" + table.split('.')[0] + "\' and c.tabname = \'" + table.split('.')[1] + "\';"
        columns = pd.read_sql(sql,ibm_db_dbi_conn).values.flatten().tolist()
        table_dict[table] = columns
    _ = close_connection_to_db(ibm_db_conn, verbose = True)
    return table_dict

def load_tab_card(schema_name, conn_str):
    schema_name=schema_name.upper()
    tab_card_dict = {}
    ibm_db_conn, ibm_db_dbi_conn = connect_to_db(conn_str, verbose=False)
    sql = "SELECT  trim(TABSCHEMA)||'.'||trim(TABNAME) FROM SYSCAT.TABLES WHERE TABSCHEMA=\'" + schema_name+"\' AND TYPE = 'T';"
    table_list = pd.read_sql(sql,ibm_db_dbi_conn).values.flatten().tolist()
    for table in table_list:
        sql = "select count(*) from "+schema_name+"."+table+";"
        tab_card = pd.read_sql(sql,ibm_db_dbi_conn).values.flatten().tolist()[0]
        tab_card_dict[table] = tab_card
    _ = close_connection_to_db(ibm_db_conn, verbose = False)
    print("Base table cards collected!")
    return tab_card_dict

def load_col_types(schema_name, conn_str):
    schema_name=schema_name.upper()
    ibm_db_conn, ibm_db_dbi_conn = connect_to_db(conn_str, verbose=True)
    sql = "select tbcreator, tbname, name, coltype from sysibm.syscolumns where tbcreator = \'{}\';".format(schema_name)
    column_types = pd.read_sql(sql,ibm_db_dbi_conn)
    _ = close_connection_to_db(ibm_db_conn, verbose = True)
    return column_types
    

def load_pkfk(schema_name, conn_str):
    schema_name=schema_name.upper()
    ibm_db_conn, ibm_db_dbi_conn = connect_to_db(conn_str, verbose=True)
    sql = "select trim(creator), trim(tbname), trim(creator)|| '.' ||trim(tbname)|| '.' ||trim(fkcolnames), trim(reftbcreator), trim(reftbname), trim(reftbcreator)|| '.' ||trim(reftbname)|| '.' ||trim(pkcolnames) from sysibm.sysrels where creator = \'{}\';".format(schema_name)
    pk_fk = pd.read_sql(sql,ibm_db_dbi_conn)
    _ = close_connection_to_db(ibm_db_conn, verbose = True)
    return pk_fk
    
def load_freq_vals(schema_name, conn_str):
    schema_name=schema_name.upper()
    ibm_db_conn, ibm_db_dbi_conn = connect_to_db(conn_str, verbose=True)
    sql = "SELECT * FROM (SELECT CD.TABSCHEMA, CD.TABNAME, CD.COLNAME, CD.COLVALUE, CD.VALCOUNT, ROW_NUMBER() OVER (PARTITION BY CD.TABSCHEMA, CD.TABNAME, CD.COLNAME) AS RN FROM SYSSTAT.COLDIST AS CD INNER JOIN SYSIBM.SYSCOLUMNS AS SC ON SC.TBCREATOR = CD.TABSCHEMA AND SC.TBNAME = CD.TABNAME AND SC.NAME = CD.COLNAME WHERE CD.COLVALUE IS NOT NULL AND CD.TABNAME NOT LIKE 'SYS%' AND CD.TABSCHEMA NOT LIKE 'SYS%' AND SC.COLTYPE IN ('INTEGER','REAL','BIGINT','DECIMAL','DOUBLE','SMALLINT') ORDER BY CD.TABSCHEMA, CD.TABNAME, CD.COLNAME, CD.VALCOUNT DESC);"
    freq_vals = pd.read_sql(sql,ibm_db_dbi_conn)
    _ = close_connection_to_db(ibm_db_conn, verbose = True)
    return freq_vals

def db_result(statement):
    # Executes a DB2 statement and writes the result to a file named 'db2_dmp'. It returns the lines of the result file.
    with open('./db2_tmp', 'w') as f:
        f.write(statement)
    rc = sp.run('db2 -tvf ./db2_tmp | cat > db2_dmp', shell = True)
    rc = sp.run('rm ./db2_tmp', shell = True)
    with open('./db2_dmp', 'r') as f:
        lines = f.readlines()
    return lines

def db2_execute(sql,ibm_db_conn,guieline=None,
    def_timeout_threshold = 100):

    # add guideline to raw_sql
    sql = sql.replace('\n',' ').replace(';','\n') +' /* '+ guieline + ' */;'
    print(sql)
    stmt = ibm_db.prepare(ibm_db_conn, sql)
    rc = ibm_db.set_option(stmt, {ibm_db.SQL_ATTR_QUERY_TIMEOUT : def_timeout_threshold}, 0)
    
    errorMsg = ''
    if stmt is False:
        print("\nERROR: Unable to prepare the SQL statement specified.")
        errorMsg = ibm_db.stmt_errormsg()
        print("\n" + errorMsg + "\n")

    tic = time.time()
    _ = ibm_db.execute(stmt)
    toc = time.time()

    latency = toc-tic

    return latency, errorMsg

# A function to explain query plans, and optionally generate explain outputs and/or guidelines
# inputs:
    # query
    # query_id
    # hints: a list containing a set of hints
    # gen_exp_output: whether to generate an explain output
    # opt_plan_path: a path to store the explain outputs and guidelines
    # gen_guideline: whether to generate guidelines
    # cmd_verbose: whether to print outputs from the executed commands
# generates 
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
    # print("exp_res:\n",exp_res)

    _ = close_connection_to_db(ibm_db_conn, verbose = False)
    return exp_res

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

    # def format_value(val):
    #     """Format a numeric value to six decimal places if it is a float; otherwise, return as string."""
    #     if isinstance(val, float):
    #         return f"{val:.6f}"
    #     return str(val)
    
    def annotate_node(node: ET.Element):
        tag = node.tag
        # Attempt to get a matching row for this node.
        matching_row = get_matching_row(tag, node.attrib)
        if matching_row is not None:
            node.set("OUTPUT_CARD", str(matching_row["OUTPUT_CARD"]))
            node.set("SELECTIVITY", str(matching_row["SELECTIVITY"]))
            node.set("TOTAL_COST", str(matching_row["TOTAL_COST"]))
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