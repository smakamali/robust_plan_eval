# ----------------------------- V2.2 ----------------------------------# 
# Captures explain and guideline
    # Note that quantifiers in the guidelines are numbered based on the order 
    # in which they appear in the rewritten query. This can be arbitrary.
    # To make the numbering non-arbitrary, we take the guideline 
    # and relabel the quantifiers so that a predetermined
    # numbering is used.
    # This requires a method that transforms numbering based on order: original guidelines, to numbering based on table sizes: guideline templates. 
    # This will be used when capturing the possible plan options and storing them along with the queries and hint sets used for guideline generation in glTemplateDF.
    # A reverse transformation will be needed before using 
    # the guidelines for execution.

# Compiles each query with multiple hintsets each producing a guideline
# Guidelines and hints are linked to the queries and stored in a dataframe
# Captures number of joins (num_joins) and stores it along with the guideline templates, so that in labeling phase only relevant guideline templates are used. -> This is no longer needed. Keeping for now in case needed later.

from explain_parser import ExplainParser as ep
import pandas as pd
import subprocess as sp
from subprocess import Popen, PIPE
import numpy as np
import os
# the following line is needed only on windows
# os.add_dll_directory('C:\\Program Files\\IBM\\SQLLIB\\clidriver\\bin')
import ibm_db
import ibm_db_dbi
import io
import shutil
import re

def connect_to_db(conn_str, verbose=False):
    # Establishes a connection to the database using the provided connection string. Returns the IBM DB2 connection and DBI connection objects.
    try:
        ibm_db_conn = ibm_db.connect(conn_str, "", "")
        ibm_db_dbi_conn = ibm_db_dbi.Connection(ibm_db_conn)
        db_conn_exists = 'ibm_db_conn' in locals() or 'ibm_db_conn' in globals()
        db_dbi_conn_exists = 'ibm_db_dbi_conn' in locals() or 'ibm_db_dbi_conn' in globals()
        if db_conn_exists and db_dbi_conn_exists and verbose:
            print('Connected to the database!')
        return ibm_db_conn, ibm_db_dbi_conn
    except:
        if verbose:
            print('Connection to the database could not be established.')
        return None, None

def close_connection_to_db(ibm_db_conn, verbose = False):
    #  Closes the connection to the database specified by the ibm_db_conn object.
    rc = ibm_db.close(ibm_db_conn)
    if verbose:
        if rc:
            print('Connection is closed.')
        else:
            print('Closing the connection failed or connection did not exist.')
    return rc

def execute_sql(sql, conn_str):
    # Executes SQL statements on the database using the provided SQL code and connection string. It connects to the database, executes each line of SQL code, and prints the executed SQL statement. If an error occurs, it prints the error message.
    ibm_db_conn, _ = connect_to_db(conn_str, verbose=False)
    buf = io.StringIO(sql)
    try:
        for line in buf.readlines():
            print(line.strip('\n'))
            _ = ibm_db.exec_immediate(ibm_db_conn, line)
            print("Transaction successful!")
    except:
        err = ibm_db.stmt_errormsg()
        print ("Transaction couldn't be completed:" , err)
    _ = close_connection_to_db(ibm_db_conn, verbose = False)

def fetch_sql_results(sql, conn_str):
    # Executes an SQL query on the database using the provided SQL code and connection string. It returns the results of the query as a Pandas DataFrame. If the query fails, it prints a message and returns None.
    ibm_db_conn, ibm_db_dbi_conn = connect_to_db(conn_str, verbose=False)
    try:
        results = pd.read_sql(sql,ibm_db_dbi_conn)
    except:
        print ("Query failed!")
        results = None
    _ = close_connection_to_db(ibm_db_conn, verbose = False)
    return results

def call_db2(statement):
    # Executes a DB2 statement using the subprocess module. It writes the provided statement to a temporary file, runs the DB2 command with the file, and then removes the temporary file. It returns the subprocess's return code.
    with open('./db2_tmp', 'w') as f:
        f.write(statement)
    rc = sp.run('db2 -tvf ./db2_tmp', shell = True)
    rc = sp.run('rm ./db2_tmp', shell = True)
    return rc

def db_result(statement):
    # Executes a DB2 statement and writes the result to a file named 'db2_dmp'. It returns the lines of the result file.
    with open('./db2_tmp', 'w') as f:
        f.write(statement)
    rc = sp.run('db2 -tvf ./db2_tmp | cat > db2_dmp', shell = True)
    rc = sp.run('rm ./db2_tmp', shell = True)
    with open('./db2_dmp', 'r') as f:
        lines = f.readlines()
    return lines

def key_value_swap(dictionary):
    # Swaps the keys and values of a dictionary and returns the new dictionary.
    new_dic = {}
    keys = dictionary.keys()
    for i in keys:
        new_dic[dictionary[i]] = i
    return new_dic

def raw_sql_process(raw_sql):
    # Processes raw SQL code by removing comments and normalizing whitespace. It returns the processed SQL code.
    # If pattern '/*' is found, remove everything between two slashes
    if '/*' in raw_sql:
        raw_sql = re.sub('/.*/','',raw_sql)
    raw_sql = ' '.join([i.strip() for i in raw_sql.splitlines()])
    # raw_sql = raw_sql.split('Normalized')[0]
    raw_sql = raw_sql.strip()
    return raw_sql

def stripAttribute(xml,attr):
    # Removes a specific attribute from an XML string. It searches for the attribute and its value between two slashes and replaces it with '/>'. It returns the modified XML string.
    while xml.find(attr) != -1:
        Start = xml.find(attr)
        End = xml.find('/>',Start)+2
        Sub = xml[Start:End]
        xml = xml.replace(Sub,'/>')
    return xml

def transformGuideline(origGuide, tab_alias_dic, table_id):
    # Replaces table quantifier IDs in a guideline template with actual table aliases. It takes the guideline template and the alias-template dictionary and returns the updated guideline.
    updatedGuide = origGuide
    tableQs = []
    tableids = []
    for Q in tab_alias_dic.keys():
        tableQs.append(Q)
        tableids.append(table_id[tab_alias_dic[Q]])
    tableQs = np.array(tableQs)
    tableids = np.array(tableids)
    newQs = ['T'+i[1:] for i in tableQs[tableids.argsort()]]
    # print('tableQs',tableQs)
    # print('newQs',newQs)
    alias_template_dict = {}
    for idx,Q in enumerate(tableQs):
        updatedGuide = updatedGuide.replace("\'"+Q+"\'", "\'"+newQs[idx]+"\'")
        alias_template_dict[newQs[idx]] = Q
    # removes index references
    updatedGuide = stripAttribute(updatedGuide,' INDEX=')
    updatedGuide = stripAttribute(updatedGuide,' IXNAME=')
    return updatedGuide, alias_template_dict

def templateToGuideline(glt, alias_template_dict):
    # Replaces table quantifier IDs in a guideline template with actual table aliases. It takes the guideline template and the alias-template dictionary and returns the updated guideline.
    for T in alias_template_dict.keys():
        # print(T,'->',alias_template_dict[T])
        glt = glt.replace("\'"+T+"\'", "\'"+alias_template_dict[T]+"\'")
    return glt

def find_between(s, first, last ):
    # Helper function that returns a substring between two given strings (first and last) in a larger string (s).
    try:
        start = s.index( first )
        end = s.index( last ) + len(last)
        return s[start:end]
    except ValueError:
        return ""

def load_db_schema(schema_name, conn_str):
    # database using the provided schema name and connection string. It returns a dictionary where the keys are table names, and the values are lists of column names for each table.
    table_dict = {}
    ibm_db_conn, ibm_db_dbi_conn = connect_to_db(conn_str, verbose=True)
    sql = "SELECT TABNAME FROM SYSCAT.TABLES WHERE TABSCHEMA=\'" + schema_name+"\' AND TYPE = 'T'"
    table_list = pd.read_sql(sql,ibm_db_dbi_conn).values.flatten()
    # print("table_list1",table_list)
    tab_sizes = []
    for table in table_list:
        sql = "select count(*) from "+ schema_name +"."+table+";"
        tab_size = pd.read_sql(sql,ibm_db_dbi_conn).values.flatten().tolist()[0]
        tab_sizes.append(tab_size)
    tab_sizes = np.array(tab_sizes)
    # print("tab_sizes",tab_sizes)
    # print(np.argsort(tab_sizes))
    table_list = table_list[np.flip(np.argsort(tab_sizes))]
    # print("table_list2",table_list)
    for table in table_list:
        sql = "Select trim(c.tabschema) || '.' || trim(c.tabname) || '.' || trim(c.colname) as column_name from syscat.columns c inner join syscat.tables t on t.tabschema = c.tabschema and t.tabname = c.tabname where t.type = 'T' and c.tabschema = \'" + schema_name + "\' and c.tabname = \'" + table + "\';"
        columns = pd.read_sql(sql,ibm_db_dbi_conn).values.flatten().tolist()
        table_dict[table] = columns
    _ = close_connection_to_db(ibm_db_conn, verbose = True)
    return table_dict

def generateExplains(max_num_queries, schema_name, encFileID, queriesFile):
    # Generates explain plans for SQL queries. It takes the maximum number of queries to process, the schema name, an encoding file ID, and a file containing the SQL queries. The function connects to the database, loads the table schema, and then generates explain plans for each query using different hint sets. It saves the explain plans and associated information to files and returns a Pandas DataFrame containing the generated explain plans.

    internalPath = './internal/'
    if not os.path.isdir(internalPath):
        os.mkdir(internalPath)
    histSets=[
            [], # represents optimizer's plan using no hintsets - always keep this at top of the list
            ['db2 .opt set disable nljn;\n'],
            ['db2 .opt set disable nljn;\n','db2 .opt set disable iscan;\n'],
            ['db2 .opt set disable hsjn;\n'],
            ['db2 .opt set disable hsjn;\n','db2 .opt set disable iscan;\n'],
            ['db2 .opt set disable mgjn;\n'],
            ['db2 .opt set disable mgjn;\n','db2 .opt set disable iscan;\n'],
            ['db2 .opt set disable nljn;\n', 'db2 .opt set disable mgjn;\n'],
            ['db2 .opt set disable nljn;\n', 'db2 .opt set disable mgjn;\n','db2 .opt set disable iscan;\n'],
            ['db2 .opt set disable nljn;\n', 'db2 .opt set disable hsjn;\n'],
            ['db2 .opt set disable nljn;\n', 'db2 .opt set disable hsjn;\n','db2 .opt set disable iscan;\n'],
            ['db2 .opt set disable mgjn;\n', 'db2 .opt set disable hsjn;\n'],
            ['db2 .opt set disable mgjn;\n', 'db2 .opt set disable hsjn;\n','db2 .opt set disable iscan;\n'],
            ]
    glTemplateDF = pd.DataFrame(columns = ['query_id','sql','hintset','num_joins','guideline','alias_template_dict','cost'])

    with open("./conn_str", "r") as conn_str_f:
        conn_str = conn_str_f.read()
    
    table_dict = load_db_schema(schema_name, conn_str)

    id_table = {}
    for idx,item in enumerate(list(table_dict.keys())):
        id_table[idx] = item
    table_id = key_value_swap(id_table)

    rc = sp.run('. ~/sqllib/db2profile', shell = True)
    rc = sp.run('~/db2cserv -d on', shell = True)
    rc = sp.run('db2stop force', shell = True)
    rc = sp.run('db2start', shell = True)

    opt_plan_PATH = './optimizer_plans/'
    if not os.path.exists(opt_plan_PATH):
        os.mkdir(opt_plan_PATH)
        print("\n" + opt_plan_PATH + " Created...\n")
    else:
        shutil.rmtree(opt_plan_PATH)
        os.mkdir(opt_plan_PATH)
        print("\n" + opt_plan_PATH + " Deleted and created again......\n")


    err_files_PATH = internalPath+'error_files'+encFileID
    with open(err_files_PATH, 'w') as f:
        f.write("Explain errors for "+ encFileID+"\n")
    print("\n" + err_files_PATH + " Created...\n")
    
    with open(queriesFile, "r") as qFile:
        queries = qFile.readlines()
    queries = [i for i in queries if i != '\n']
    counter = 0
    query_success_id=0
    guide_success_id=0
    
    ibm_db_conn, ibm_db_dbi_conn = connect_to_db(conn_str)

    for i in queries:
        line = i.strip('\n')
        # print(line)
        if "SELECT" in line and counter < max_num_queries:
            # count number of joins in the query
            num_joins = line.count('JOIN')
            print("processing query", counter+1, "out of", max_num_queries)
            
            minOneSuccess = False
            opt_plan_success = False
            counter+=1
            for hintidx, hintset in enumerate(histSets):
                command='db2 connect to tpcds;\n'
                command+='db2 .opt set enable nljn;\n'
                command+='db2 delete from systools.explain_instance;\n'
                # for hint in hintset: # replace with .join
                #     command+=hint
                command += ''.join(hintset)
                guidlineFile = opt_plan_PATH+'query#' + str(query_success_id) + '-' + str(hintidx) + 'guide'
                command+= 'db2 .opt set lockplan on my.guideline;\n'
                command+='db2 \"explain plan for ' + line + ';\"\n'
                exp_name = opt_plan_PATH+'query#' + str(query_success_id) + '-' + str(hintidx) + '.ex'
                command+='db2 .opt set lockplan off;\n'
                command+= 'cp ~/sqllib/my.guideline '+guidlineFile+'\n'
                if hintidx == 0:
                    command+='db2exfmt -d tpcds -1 -o ' + exp_name + ";\n"
                commands = command.split('\n')
                # for cmd in commands:
                    # print('>> ',cmd)
                process = Popen( "/bin/bash", shell=False, universal_newlines=True,
                    stdin=PIPE, stdout=PIPE, stderr=PIPE)
                output, _ = process.communicate(command)
                # print(output)

                if hintidx == 0:
                    try:
                        data = ep(open(exp_name, 'r').read()).parse()
                        tab_alias_dict = data['tab_alias_dic']
                        opt_plan_success = True
                    except(TypeError):
                        with open(err_files_PATH, 'a') as err:
                            print(exp_name, '............. Type Error Exception', TypeError)
                            err.write(exp_name + ' Type Error! \n')
                    except(IndexError):
                        print(exp_name,'............. Index Error Exception', IndexError)
                        with open(err_files_PATH, 'a') as err:
                            err.write(exp_name + ' Index Error! \n')
                    except(FileNotFoundError):
                        print(exp_name,'............. FileNotFoundError Exception', FileNotFoundError)
                        with open(err_files_PATH, 'a') as err:
                            err.write(exp_name + ' FileNotFound Error! \n')
                    except:
                        pass
                
                # extract plan cost
                sql = "select total_cost from systools.explain_operator where operator_type = 'RETURN'"
                plan_cost = pd.read_sql(sql,ibm_db_dbi_conn).values.squeeze()
                # print('plan_cost',plan_cost)

                if opt_plan_success and plan_cost != '[]':
                    with open(guidlineFile,'r') as f:
                        origGuide = f.read()
                        origGuide = find_between(origGuide, '<OPTGUIDELINES>', '</OPTGUIDELINES>').replace('\n','').replace('\t','')
                        
                        # print("----> hintset",hintset)
                        # print("origGuide ---------> ",origGuide)
                        newGuide, alias_template_dict = transformGuideline(origGuide, tab_alias_dict, table_id)
                        # print("newGuide ---------> ",newGuide)
                    
                    # write the sql, hint set, guideline template, guideline, cost to a dataframe
                    glTemplateDF = glTemplateDF.append({'query_id':query_success_id,'sql':line,'hintset':''.join(hintset),
                                                        'num_joins':num_joins,'guideline': newGuide,
                                                        'alias_template_dict':alias_template_dict,
                                                        'cost':plan_cost}, ignore_index=True)
                    
                    minOneSuccess = True
            
            if minOneSuccess:
                query_success_id+=1
            
            print('queries successfully processed:{}'.format(query_success_id))
            print('Progress:{:.2%}'.format(counter/max_num_queries))

        if query_success_id%100:
            # checkpoint - write to disk
            glTemplateDF.to_pickle(internalPath+'query_hint_gl_'+encFileID+'.pickle')

        if query_success_id%10:
            # delete dump files: '/home/db2inst1/sqllib/db2dump/DIAG0000/FODC_AppErr_*'
            command = 'sudo rm -r -f /home/db2inst1/sqllib/db2dump/DIAG0000/FODC_AppErr_*'
            process = Popen( "/bin/bash", shell=False, universal_newlines=True,
                        stdin=PIPE, stdout=PIPE, stderr=PIPE)
            output, _ = process.communicate(command)

    _ = close_connection_to_db(ibm_db_conn, verbose = True)

    # final write to disk
    glTemplateDF.to_pickle(internalPath+'query_hint_gl_'+encFileID+'.pickle')
    
    print("****Processing all explains is done!****")
    print(counter, "queries were processes.")
    print(query_success_id, "queries had at least one valid guideline.")
    print(guide_success_id, "explains were created successfully.")
    print("Number of guideline templates generated: ",len(glTemplateDF.guideline.values.tolist()))
    print()

if __name__ == '__main__':
    generateExplains(max_num_queries = 5, # Specify the max number of queries to explain
                    schema_name = 'TPCDS', # schema name
                    encFileID = "1-4j_0-2extjp_2-5lp_tmp",
                    queriesFile = "./input/input_queries.sql")