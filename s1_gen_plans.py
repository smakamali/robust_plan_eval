# ----------------------------- V2.2 ----------------------------------# 
#  TODO: update plan_parser to properly extract table aliases -> DONE!
#  TODO: abandon the idea of converting guidelines to templates. Instead only store alias_table_dict

import os
import io
import re
import json
import shutil
import subprocess as sp
from subprocess import Popen, PIPE
import numpy as np
import pandas as pd
# the following line is needed only on windows
# os.add_dll_directory('C:\\Program Files\\IBM\\SQLLIB\\clidriver\\bin')
import ibm_db
import ibm_db_dbi
from explain_parser import ExplainParser as ep
from util import load_input_queries
from db_util import db2_explain, load_db_schema

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

def generateExplains(max_num_queries, schema_name, encFileID, input_dir):
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
    glTemplateDF = pd.DataFrame(columns = ['query_id','sql','hintset','guideline','alias_template_dict','cost'])

    with open("./conn_str", "r") as conn_str_f:
        conn_str = conn_str_f.read()
    
    table_dict = load_db_schema(schema_name, conn_str)
    table_id = {}
    for idx,item in enumerate(list(table_dict.keys())):
        table_id[item.split('.')[1]] = idx

    rc = sp.run('. ~/sqllib/db2profile', shell = True)
    rc = sp.run('~/db2cserv -d on', shell = True)
    rc = sp.run('db2stop force', shell = True)
    rc = sp.run('db2start', shell = True)

    opt_plan_path = './optimizer_plans/'
    if not os.path.exists(opt_plan_path):
        os.mkdir(opt_plan_path)
        print("\n" + opt_plan_path + " Created...\n")
    else:
        shutil.rmtree(opt_plan_path)
        os.mkdir(opt_plan_path)
        print("\n" + opt_plan_path + " Deleted and created again......\n")


    err_files_path = os.path.join(internalPath,'error_files_{}'.format(encFileID))
    with open(err_files_path, 'w') as f:
        f.write("Explain errors for "+ encFileID+"\n")
    print("\n" + err_files_path + " Created...\n")
    
    # load input queries
    queries, query_ids = load_input_queries(input_dir)

    counter = 0
    query_success_id=0
    guide_success_id=0

    for idx,i in enumerate(queries):
        line = i.replace('\n',' ')
        query_id = query_ids[idx].split('.')[0]
        print("query_id:",query_id)
        print(i)
        # print(line)
        if counter < max_num_queries:
            # count number of joins in the query
            print("processing query", counter+1, "out of", max_num_queries)
            
            minOneSuccess = False
            opt_plan_success = False
            counter+=1
            for hintidx, hintset in enumerate(histSets):
                print("Hint Set:",hintset)

                if hintidx == 0:
                    gen_exp_output = True
                else:
                    gen_exp_output = False

                exp_name, guidlineFile, plan_cost = db2_explain(schema_name='imdb', sql=line, query_id=query_id, hintset=hintset, hintset_id=hintidx,opt_plan_path=opt_plan_path,gen_exp_output=gen_exp_output, gen_guideline=True, return_cost=True, conn_str=conn_str,cmd_verbose=True)

                if hintidx == 0:
                    try:
                        data = ep(open(exp_name, 'r').read()).parse()
                        tab_alias_dict = data['tab_alias_dic']
                        print("tab_alias_dict",tab_alias_dict)
                        opt_plan_success = True
                    except(TypeError):
                        with open(err_files_path, 'a') as err:
                            print(exp_name, '............. Type Error Exception', TypeError)
                            err.write(query_id+'\n'+line+'\n'+exp_name + ' Type Error! \n')
                    except(IndexError):
                        print(exp_name,'............. Index Error Exception', IndexError)
                        with open(err_files_path, 'a') as err:
                            err.write(query_id+'\n'+line+'\n'+exp_name + ' Index Error! \n')
                    except(FileNotFoundError):
                        print(exp_name,'............. FileNotFoundError Exception', FileNotFoundError)
                        with open(err_files_path, 'a') as err:
                            err.write(query_id+'\n'+line+'\n'+exp_name + ' FileNotFound Error! \n')
                    except:
                        pass

                if opt_plan_success:
                    with open(guidlineFile,'r') as f:
                        origGuide = f.read()
                        origGuide = find_between(origGuide, '<OPTGUIDELINES>', '</OPTGUIDELINES>').replace('\n','').replace('\t','')
                        print("Guideline",origGuide)
                                            
                    # write the sql, hint set, guideline template, guideline, cost to a dataframe
                    new_row = {'query_id':[query_id],'sql':[line],
                               'hintset':[''.join(hintset)],'guideline': [origGuide],
                               'alias_template_dict':[json.dumps(tab_alias_dict)],
                               'cost':plan_cost}
                    glTemplateDF = pd.concat([glTemplateDF,pd.DataFrame(new_row)],ignore_index=True)
                    
                    minOneSuccess = True
                
                    guide_success_id+=1
            
            if minOneSuccess:
                query_success_id+=1
            
            print('queries successfully processed:{}'.format(query_success_id))
            print('Progress:{:.2%}'.format(counter/max_num_queries))

        if query_success_id%100:
            # checkpoint - write to disk
            glTemplateDF.to_pickle(os.path.join(internalPath,'query_hint_gl_{}.pickle'.format(encFileID)))

        if query_success_id%10:
            # delete dump files: '/home/db2inst1/sqllib/db2dump/DIAG0000/FODC_AppErr_*'
            cmd = 'rm -r -f ~/sqllib/db2dump/DIAG0000/FODC_AppErr_*'
            process = Popen( "/bin/bash", shell=False, universal_newlines=True,
                        stdin=PIPE, stdout=PIPE, stderr=PIPE)
            output, _ = process.communicate(cmd)

    # final write to disk
    glTemplateDF.to_pickle(os.path.join(internalPath,'query_hint_gl_{}.pickle'.format(encFileID)))

    print("****Processing all explains is done!****")
    print(counter, "queries were processes.")
    print(query_success_id, "queries had at least one valid guideline.")
    print(guide_success_id, "explains were created successfully.")
    print()

if __name__ == '__main__':
    generateExplains(max_num_queries = 114, # Specify the max number of queries to explain
                    schema_name = 'imdb', # schema name
                    encFileID = "job", # a unique id for the dataset
                    input_dir = "./input") # the directory that contains query.sql file(s)