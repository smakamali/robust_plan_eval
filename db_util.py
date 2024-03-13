import os
from subprocess import Popen, PIPE
import time
import numpy as np
import pandas as pd
import ibm_db
import ibm_db_dbi
import subprocess as sp
from util import find_between

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
    sql = sql +' /* '+ guieline + ' */;'

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
    
    plan_cost = pd.read_sql(sql,ibm_db_dbi_conn).values.squeeze()
    _=close_connection_to_db(ibm_db_conn)

    return plan_cost

def get_card_sel(conn_str):

    ibm_db_conn, ibm_db_dbi_conn = connect_to_db(conn_str, verbose=False)

    exp_sql = """
    SELECT EO.OPERATOR_ID, 
        EO.OPERATOR_TYPE,
        ESI.STREAM_COUNT AS INPUT_CARD, 
        ESO.STREAM_COUNT AS OUTPUT_CARD, 
        EO.TOTAL_COST, 
        ESI.OBJECT_NAME AS INPUT_OBJECT
    FROM SYSTOOLS.EXPLAIN_OPERATOR AS EO 
        LEFT OUTER JOIN SYSTOOLS.EXPLAIN_STREAM AS ESI 
        ON ESI.TARGET_ID = EO.OPERATOR_ID
        LEFT OUTER JOIN SYSTOOLS.EXPLAIN_STREAM AS ESO 
        ON ESO.SOURCE_ID = EO.OPERATOR_ID
    WHERE EO.OPERATOR_TYPE IN ('TBSCAN','FETCH')
        AND ESI.OBJECT_NAME <> '-';
        """

    exp_res = pd.read_sql(exp_sql,ibm_db_dbi_conn)
    exp_res["SELECTIVITY"] = exp_res.OUTPUT_CARD/exp_res.INPUT_CARD
    
    _ = close_connection_to_db(ibm_db_conn, verbose = False)

    return exp_res[["INPUT_OBJECT","INPUT_CARD","OUTPUT_CARD","SELECTIVITY"]]