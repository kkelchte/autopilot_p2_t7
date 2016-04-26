#!/usr/bin/bash
# This scripts sets some parameters for running a tasks,
# creates a condor and shell scripts and launches the stuff on condor.

TASK='pilot_train.py' 	#name of task in matlab to run
#val=0			#id number of this condor task avoid overlap
model="$1"
learning_rate="$2"
max_grad_norm="$3"
num_layers="$4"
hidden_size="$5"
num_steps="$6"

description=""
condor_output_dir='/esat/qayd/kkelchte/tensorflow/condor_output'
#------------------------------------
if [ ! -z $model ]; then 
	TASK="$TASK --model ${model}"
	description="${description}_$model"
fi
if [ ! -z $learning_rate ]; then 
	TASK="$TASK --learning_rate ${learning_rate}"
	description="${description}_lr_$learning_rate"
fi
if [ ! -z $max_grad_norm ]; then 
	TASK="$TASK --max_grad_norm ${max_grad_norm}"
	description="${description}_mxgrad_$max_grad_norm"
fi
if [ ! -z $num_layers ]; then 
	TASK="$TASK --num_layers ${num_layers}"
	description="${description}_layers_$num_layers"
fi
if [ ! -z $hidden_size ]; then 
	TASK="$TASK --hidden_size ${hidden_size}"
	description="${description}_size_$hidden_size"
fi
if [ ! -z $num_steps ]; then 
	TASK="$TASK --num_steps ${num_steps}"
	description="${description}_steps_$num_steps"
fi
echo $TASK
# Delete previous log files if they are there
if [ -d $condor_output_dir ];then
rm -f "$condor_output_dir/condor${description}.log"
rm -f "$condor_output_dir/condor${description}.out"
rm -f "$condor_output_dir/condor${description}.err"
else
mkdir $condor_output_dir
fi
 
temp_dir="/users/visics/kkelchte/tensorflow/examples/pilot/.tmp"
condor_file="${temp_dir}/condor${description}.condor"
shell_file="${temp_dir}/run${description}.sh"

mkdir $temp_dir
#--------------------------------------------------------------------------------------------
echo "Universe         = vanilla" > $condor_file
echo "">> $condor_file
echo "RequestCpus      = 1" >> $condor_file
echo "Request_GPUs	= 1" >> $condor_file
echo "RequestMemory    = 15G" >> $condor_file
#echo "RequestDisk      = 25G" >> $condor_file
#wall time ==> generally assumed a job should take 6hours longest,
#if you want longer or shorter you can set the number of seconds. (max 1h ~ +3600s)
echo "+RequestWalltime = 86400" >> $condor_file
#echo "+RequestWalltime = 10800" >> $condor_file
echo "">> $condor_file
#echo "Requirements = (machineowner == Visics)" >> $condor_file
echo "Requirements = (CUDAGlobalMemoryMb >= 1900) && (CUDACapability >= 3.5)">> $condor_file
#echo "Requirements = (CUDAGlobalMemoryMb > 1900) && (CUDADeviceName == 'GeForce GTX 960' || CUDADeviceName == 'GeForce GTX 980' ) && (machineowner == Visics)" >> $condor_file
#echo "Requirements = ((machine == "vega.esat.kuleuven.be") || (machine == "wasat.esat.kuleuven.be") || (machine == "yildun.esat.kuleuven.be"))" >> $condor_file
echo "Niceuser = true" >> $condor_file
echo "Initialdir   =$temp_dir" >> $condor_file
echo "Executable   = $shell_file" >> $condor_file
echo "Log 	   = $condor_output_dir/condor${description}.log" >> $condor_file
echo "Output       = $condor_output_dir/condor${description}.out" >> $condor_file
echo "Error        = $condor_output_dir/condor${description}.err" >> $condor_file
echo "">> $condor_file
#mail kkelchte on Error or Always
echo "Notification = Error" >> $condor_file
echo "Queue" >> $condor_file

echo "#!/usr/bin/bash" > $shell_file
echo "task='"${TASK}"'">>$shell_file
echo 'echo $task'>>$shell_file
echo "##-------------------------------------------- ">>$shell_file
echo "echo 'run_the_thing has started' ">>$shell_file
echo "# load cuda and cdnn path in load library path">>$shell_file
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-7.5/lib64:/users/visics/kkelchte/local/lib/cudnn-4.0/cuda/lib64:">>$shell_file
echo "# run virtual environment for tensorflow">>$shell_file
echo "source /users/visics/kkelchte/tensorflow/bin/activate">>$shell_file
echo "# set python library path">>$shell_file
echo "export PYTHONPATH=/users/visics/kkelchte/tensorflow/lib/python2.7/site-packages">>$shell_file
echo "cd /users/visics/kkelchte/tensorflow/examples/pilot">>$shell_file
echo "echo 'went to directory ' $PWD">>$shell_file
echo "python $TASK">>$shell_file
echo "echo $TASK has finished. \n \n description: $description.\n \n $con' | mailx -s 'condor' klaas.kelchtermans@esat.kuleuven.be">>$shell_file

chmod 755 $condor_file
chmod 755 $shell_file

condor_submit $condor_file
