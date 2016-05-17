rm -rf .tmp
#condor_task    #model #lr #drp #opt #normalized #randomorder
./condor_task.sh big 0.001 1.0 Adam False True
./condor_task.sh big 0.001 1.0 Adam True True
./condor_task.sh big 0.001 0.5 Adam False True
./condor_task.sh big 0.100 1.0 GradientDescent False
./condor_task.sh big 0.001 1.0 Momentum False True
./condor_task.sh big 0.001 1.0 Adam False False

#STILL TO BE TESTED
#./condor_task.sh big 0.001 1.0 Adam False frgt05

