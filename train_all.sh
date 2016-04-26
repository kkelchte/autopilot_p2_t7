rm -rf ./tmp
./condor_task.sh medium 1 5 2 50 10
./condor_task.sh medium 0.1 5 2 50 10
./condor_task.sh medium 0.01 5 2 50 10
./condor_task.sh medium 0.001 5 2 50 10
./condor_task.sh medium 0.1 5 2 50 -1
./condor_task.sh big 1 5 2 50 -1
./condor_task.sh big 0.1 5 2 50 -1
./condor_task.sh big 0.01 5 2 50 -1

