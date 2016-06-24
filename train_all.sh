#Some general training/testing architecture

#See if it is ok to delete the .tmp folder...
#rm -rf .tmp


##STEP 1: pick CNN feats inception / pcnn app/flow/both
#               model net  feat
#./condor_task.sh big inception both
#./condor_task.sh big inception app
#./condor_task.sh big inception flow
#./condor_task.sh big pcnn both
#./condor_task.sh big pcnn app
#./condor_task.sh big pcnn flow
# adapt network and feature type in default values of pilot code

##STEP 2: see how small you can make the prefered model
##              model hsize
#./condor_task.sh big 30
#./condor_task.sh big 20
# ./condor_task.sh big 10
# ./condor_task.sh big 5
# ./condor_task.sh big 3
# ./condor_task.sh big 2
# ./condor_task.sh big 1
## adapt size in default values of pilot code

./condor_task.sh big 0.001
./condor_task.sh dumpster 0.001
./condor_task.sh dumpster 0.01
./condor_task.sh dumpster 0.1
./condor_task.sh dumpster 0.0001
./condor_task.sh dumpster 0.00001

##STEP 3: choose optimizer
##              model optimizir
#./condor_task.sh big Adam
#./condor_task.sh big RMSProp
#./condor_task.sh big GradientDescent
#./condor_task.sh big Momentum
#./condor_task.sh big Adagrad
## adapt size in default values of pilot code





