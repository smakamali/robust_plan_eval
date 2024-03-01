import os
import time
import datetime
import subprocess as sp
import pickle
import ibm_db

def templateToGuideline(glt, alias_template_dict):
    for T in alias_template_dict.keys():
        # print(T,'->',alias_template_dict[T])
        glt = glt.replace("\'"+T+"\'", "\'"+alias_template_dict[T]+"\'")
    return glt

def labelQueries(max_num_queries, def_timeout_threshold, dynamic_timeout,dynamic_timeout_factor, encFileID):
    tic0 = time.time()

    internalPath = './internal/' 
    outputPath = './output/'
    # if not os.path.isdir(internalPath): #This should exist beforehand!
    #     os.mkdir(internalPath)
    if not os.path.isdir(outputPath):
        os.mkdir(outputPath)
    
    setup = ". ~/sqllib/db2profile\n"
    db_text = setup+ "db2start\n" 
    rc = sp.call(db_text, shell = True)
    
    with open("./conn_str", "r") as conn_str_f:
        conn_str = conn_str_f.read()
    
    ibm_db_conn = ibm_db.pconnect(conn_str, "", "")
    if ibm_db_conn:
        print('Connected to the database!')

    with open(internalPath+'query_hint_gl_'+encFileID+'.pickle',"rb") as f:
        query_hint_gl = pickle.load(f)
    query_hint_gl=query_hint_gl[query_hint_gl.query_id < max_num_queries]
    queries_to_process = query_hint_gl.query_id.unique().shape[0]

    # add a new columns to the pandas dataframe to store runtime and flags
    query_hint_gl['elapsed_time'] = 0
    query_hint_gl['optChoiceFlag'] = ''
    
    date_time = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    with open(internalPath+'log_'+encFileID+'.out', 'w') as f:
        f.write(str(date_time)+"log file for labeling "+encFileID+' created.\n')
    logFile =  open(internalPath+'log_'+encFileID+'.out', 'a')

    query_counter=0
    timed_out=0
    query_gl = 0

    for idx in range(query_hint_gl.shape[0]):
        row = query_hint_gl.loc[idx]

        query_id = row.query_id
        
        raw_sql= row.sql
        hintset = row.hintset
        guideline = row.guideline
        alias_template_dict = row.alias_template_dict
        print('hintset',hintset)
        
        hint_count=0

        date_time = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        log = str(date_time)+" : Processing query "+str(query_id)+" started."
        logFile.write(log + "\n")

        print("query: ", query_id)
        print(raw_sql)

        # get the optimal plan: the first guideline per query
        print('hintset',hintset)
        if hintset == '':
            print('>>>> 1st')
            hint_count=0

            sql = raw_sql

            logFile.write(sql + "\n")

            stmt = ibm_db.prepare(ibm_db_conn, sql)
            rc = ibm_db.set_option(stmt, {ibm_db.SQL_ATTR_QUERY_TIMEOUT : def_timeout_threshold}, 0)
            _ = ibm_db.execute(stmt)
            tic = time.time()
            _ = ibm_db.execute(stmt)
            toc = time.time()
            optPlanTime = toc - tic
            if dynamic_timeout:
                timeout_threshold = max(int(optPlanTime*dynamic_timeout_factor),10)
            
            elapsed_time = optPlanTime
            optChoiceFlag = "optimal"

            print("optPlanTime :", optPlanTime)
            print("timeout threshold :", timeout_threshold)

            query_gl += 1
        
        else:
            hint_count+=1
            # transform guideline template to guideline
            gl = templateToGuideline(guideline, alias_template_dict)
            # print(gl)
            
            # add guideline to raw_sql
            sql = raw_sql +' /* '+ gl + ' */;'

            try:
                stmt = ibm_db.prepare(ibm_db_conn, sql)
                rc = ibm_db.set_option(stmt, {ibm_db.SQL_ATTR_QUERY_TIMEOUT : timeout_threshold}, 0)
                # value = ibm_db.get_option(stmt, ibm_db.SQL_ATTR_QUERY_TIMEOUT, 0)
                # print("Statement options:\nSQL_ATTR_QUERY_TIMEOUT = {}".format(str(value)), end="\n")
                tic = time.time()
                rc = ibm_db.execute(stmt)
                toc = time.time()
                elapsed_time = toc - tic
                if elapsed_time > timeout_threshold:
                    elapsed_time = 1000
                    timed_out+=1
                # row = ibm_db.fetch_assoc(stmt)
                
                query_gl += 1

            except Exception as inst:
                print("!!!!!!!!!!!!!!!!!!!!", inst)
                date_time = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
                logFile.write(str(date_time) + ": \n")
                logFile.write(sql + "\n")
                logFile.write(str(inst) + "\n")
                elapsed_time = 1000
                timed_out+=1 # treating exceptions like timeouts
            
            # collect plan labels
            if elapsed_time < optPlanTime:
                optChoiceFlag = "alternative"
            elif elapsed_time == 1000:
                optChoiceFlag = "timed_out"
            else:
                optChoiceFlag = "NA"
            
        # store the runtime in the datafrme
        query_hint_gl.loc[idx,'elapsed_time'] = elapsed_time
        query_hint_gl.loc[idx,'optChoiceFlag'] = optChoiceFlag

        optChoiceLabel = ""
        if optChoiceFlag != "":
            optChoiceLabel = " --> " + optChoiceFlag
        date_time = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        print(str(date_time)+": query-guideline : "+str(query_id)+'-'+str(hint_count)+ " : elapsed_time : "+ str(elapsed_time)+ str(optChoiceLabel))
        
        log = str(date_time)+" : Processing query "+str(query_id)+" completed."
        logFile.write(log + "\n")
            
        query_counter+=1
        
        # save the dataframe every n step and then at the end
        if idx%100 == 0:
            query_hint_gl.to_pickle(internalPath+'query_hint_gl_labeled'+encFileID+'.pickle')
        query_hint_gl.to_pickle(internalPath+'query_hint_gl_labeled'+encFileID+'.pickle')
    

                

    print("-----------------Query labeling finished!-----------------")
    print("{} queries were processed.".format(queries_to_process))
    print("{} query/guideline pairs were successfully labeled.".format(str(query_gl)))
    print("{} query/guideline pairs exceeded timeout threshold of {} seconds.".format(str(timed_out),str(timeout_threshold)))
    
    # resultFile.close()
    logFile.close()

    rc = ibm_db.close(ibm_db_conn)
    if rc:
        print('Connection is closed.')
    else:
        print('Closing the connection failed or connection did not exist.')

    labeling_time = time.time() - tic0
    print("Time it took to label queries: ", str(labeling_time))

if __name__ == '__main__':
    labelQueries(max_num_queries = 10, # Specify the max number of queries to label
                def_timeout_threshold = 5, # Cut off time in seconds. Queries taking more than this amount are terminated prematurely.
                dynamic_timeout = True, # determines whether a static timeout threshold determined by timeout_threshold is used, or a dynamic threshold determined by a dynamic_timeout_factor of the optimal plan elapsed time
                dynamic_timeout_factor = 3,
                encFileID = "1-4j_0-2extjp_2-5lp_tmp")
