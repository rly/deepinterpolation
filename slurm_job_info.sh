#!/bin/bash

last_week=`date -d "last week" +%Y-%m-%d`
sacct --starttime ${last_week} --format=jobid,jobname,nnodes,alloctres%50,state,exitcode,start,elapsed
