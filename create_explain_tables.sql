connect to tpcds;

CALL SYSPROC.SYSINSTALLOBJECTS('EXPLAIN', 'C', 
        CAST (NULL AS VARCHAR(128)), CAST (NULL AS VARCHAR(128)));

commit;

connect reset;