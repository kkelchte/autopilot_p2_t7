rm -rf .tmp
#condor_task #model #lr #size #network #ftype
./condor_task.sh big 0.001 10 logits app
./condor_task.sh big 0.1 10 logits app
./condor_task.sh big 0.00001 10 logits app

#STILL TO BE TESTED
#./condor_task.sh big 0.001 1.0 Adam False frgt05

