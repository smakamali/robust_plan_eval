import pandas as pd
import ibm_db
import ibm_db_dbi


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

def load_db_schema(schema_name, conn_str):
    schema_name=schema_name.upper()
    table_dict = {}
    ibm_db_conn, ibm_db_dbi_conn = connect_to_db(conn_str, verbose=True)
    sql = "SELECT TABNAME FROM SYSCAT.TABLES WHERE TABSCHEMA=\'" + schema_name+"\' AND TYPE = 'T'"
    table_list = pd.read_sql(sql,ibm_db_dbi_conn).values.flatten().tolist()
    for table in table_list:
        sql = "Select trim(c.tabschema) || '.' || trim(c.tabname) || '.' || trim(c.colname) as column_name from syscat.columns c inner join syscat.tables t on t.tabschema = c.tabschema and t.tabname = c.tabname where t.type = 'T' and c.tabschema = \'" + schema_name + "\' and c.tabname = \'" + table + "\';"
        columns = pd.read_sql(sql,ibm_db_dbi_conn).values.flatten().tolist()
        table_dict[schema_name+'.'+table] = columns
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
    sql = "select tbcreator, tbname, name, coltype from sysibm.syscolumns where tbcreator not like 'SYS%';"
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

    
