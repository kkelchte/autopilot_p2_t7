rm -rf .tmp
END=10
for i in $(seq 1 $END);
do
    #condor_task    #model #lr  #tag
    ./condor_task.sh big 0.001 $i
    sleep 1h #5m
done

#STILL TO BE TESTED
#./condor_task.sh big 0.001 1.0 Adam False frgt05

