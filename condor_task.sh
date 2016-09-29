#!/usr/bin/bash
# This scripts sets some parameters for running a tasks,
# creates a condor and shell scripts and launches the stuff on condor.

#TASK='pilot_CNN.py --chosen_set seq_oa_inc_fc' 	#wall_test
#description="seq_oa_inc_fc_cnn_feats"

#_______________________________________________________
TASK='pilot_train.py' 	#name of task in matlab to run

####   depth_fc_dagger
#model='depth_fc'
#log_tag='seq' #d1
#fine='True'
#mod_dir='depth_fc_d1_fi_hsz_400_net_stijn_depth_fc'

####   inc_fc_dagger
model='inc_fc'
log_tag='seq_d3' 
fine='True'
mod_dir='inc_fc_seq_d2_fi_hsz_400_fc'

#_______________________________________________________
#best to ask for 12G GPU

# # # inc_lstm_dagger
# andromeda='yes'
# model='inc_lstm'
# log_tag='seq_d1'
# fine='True'
# mod_dir='inc_lstm_seq'

# depth_lstm_dagger
#model='depth_lstm'
#log_tag='seq' #d1
# fine='True'
# mod_dir='depth_lstm_net_stijn_depth'

#_______________________________________________________
#Corridor challenge
#model='comb'
#log_tag='corridor'


#Wall challenge !!! Dont forget to preprocess!!!
#model='cwall' #"$1"
#log_tag="d1" #tell python this is not a test so it is not including the testing tag
#wsize="300"
#bsize="2"
#fine='True'
#mod_dir='cwall_batchwise'

#Wall challenge for different windowsizes and according batchsizes
# model='winwall' #"$1"
# wsize="$1"
# bsize="$2"
# log_tag="$3" #tell python this is not a test so it is not including the testing tag

### SELECTED SEQUENTIAL TEST
#model='seldat' #'cwall' #"$1"
#log_tag="selected_30E" #tell python this is not a test so it is not including the testing tag
#feature_type='depth_estimate' #"$5"
#network='stijn' #"$6"
#fc='True'
#hidden_size='400'
#bsize='500'

# RANDOM OA TEST
#model='dagger' #'cwall' #"$1"
#log_tag="big" #tell python this is not a test so it is not including the testing tag
#fc='True'
#bsize='500'
#hidden_size='400'
#feature_type='depth_estimate' #"$5"
#network='stijn' #"$6"


#OA Stijns depth
#model='dagger' #'cwall' #"$1"
#log_tag="big" #tell python this is not a test so it is not including the testing tag
#feature_type='depth_estimate' #"$5"
#network='stijn' #"$6"

### 2 OA FC
#model='dagger'
#log_tag='no_test'

#fine='True'
#mod_dir='dagger_hsz_400_fc'

#Expert session
#model='expert'
#log_tag='_2'
#feature_type='depth_estimate' #"$5"
#network='stijn' #"$6"


#hidden_size='400' #"$2"
#batchwise="$2"
#fc='True' #"$3"
#stpfc="$4"
#feature_type="$5"
#network="$6"
#sample="$9"
#log_tag="${10}" #'no_test'
#mod_dir="${11}" #model dir to finetune
#learning_rate="$2"
#keep_prob="$3"
#optimizer="$4"
#normalized="$5"
#random_order="$2"


condor_output_dir='/esat/qayd/kkelchte/tensorflow/condor_output'
#------------------------------------
if [ ! -z $model ]; then 
	TASK="$TASK --model ${model}"
	description="${description}_$model"
fi
if [ ! -z $batchwise ]; then 
	TASK="$TASK --batchwise_learning ${batchwise}"
	description="${description}_bwise_$batchwise"
fi
if [ ! -z $fc ]; then 
	TASK="$TASK --fc_only ${fc}"
	description="${description}_fc_$fc"
fi
if [ ! -z $stpfc ]; then 
	TASK="$TASK --step_size_fnn ${stpfc}"
	description="${description}_${stpfc}"
fi
if [ ! -z $feature_type ]; then 
	TASK="$TASK --feature_type ${feature_type}"
	description="${description}_$feature_type"
fi
if [ ! -z $network ]; then 
	TASK="$TASK --network ${network}"
	description="${description}_net_$network"
fi
if [ ! -z $wsize ]; then 
	TASK="$TASK --window_size ${wsize}"
	description="${description}_ws_$wsize"
fi
if [ ! -z $bsize ]; then 
	TASK="$TASK --batch_size_fnn ${bsize}"
	description="${description}_bs_$bsize"
fi
if [ ! -z $sample ]; then 
	TASK="$TASK --sample ${sample}"
	description="${description}_sample_${sample}"
fi
if [ ! -z $log_tag ]; then 
	TASK="$TASK --log_tag ${log_tag}"
	description="${description}_${log_tag}"
fi
if [ ! -z $mod_dir ]; then
    TASK="$TASK --init_model_dir ${mod_dir}"
    #description="${description}_${log_tag}"
fi
if [ ! -z $fine ]; then
    TASK="$TASK --finetune ${fine}"
    description="${description}_fine"
fi

if [ ! -z $learning_rate ]; then 
	TASK="$TASK --learning_rate ${learning_rate}"
	description="${description}_lr_$learning_rate"
fi
if [ ! -z $hidden_size ]; then 
	TASK="$TASK --hidden_size ${hidden_size}"
	description="${description}_hsize_$hidden_size"
fi
if [ ! -z $keep_prob ]; then 
	TASK="$TASK --keep_prob ${keep_prob}"
	description="${description}_drop_$keep_prob"
fi
if [ ! -z ${optimizer} ]; then 
	TASK="$TASK --optimizer ${optimizer}"
	description="${description}_opt_${optimizer}"
fi
if [ ! -z $normalized ]; then 
	TASK="$TASK --normalized ${normalized}"
	description="${description}_norm_$normalized"
fi
if [ ! -z $random_order ]; then 
	TASK="$TASK --random_order ${random_order}"
	description="${description}_rand_$random_order"
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

mkdir -p $temp_dir
#--------------------------------------------------------------------------------------------
echo "Universe         = vanilla" > $condor_file
echo "">> $condor_file
echo "RequestCpus      = 8" >> $condor_file
echo "Request_GPUs = 1" >> $condor_file

# ---4Gb option---
if [ -z $andromeda ]; then
echo "RequestMemory = 15900" >> $condor_file
echo "Requirements = (CUDAGlobalMemoryMb >= 1900) && (CUDACapability >= 3.5)">> $condor_file
#echo "Requirements = (CUDAGlobalMemoryMb >= 3900) && (CUDACapability >= 3.5)">> $condor_file
else
echo "RequestMemory = 62000" >> $condor_file
echo "Requirements = (CUDAGlobalMemoryMb >= 3900) && (CUDACapability >= 3.5)">> $condor_file
#echo "Requirements = (CUDAGlobalMemoryMb >= 11000) && (CUDACapability >= 3.5)">> $condor_file
fi
# ---Trash option---
#echo "Requirements = (CUDAGlobalMemoryMb >= 3900) && (CUDACapability >= 3.5) && (CUDADeviceName == \"GeForce GTX 960\" || CUDADeviceName == \"GeForce GTX 980\" )">> $condor_file

#echo "RequestMemory = 16G" >> $condor_file
#echo "Requirements = (CUDAGlobalMemoryMb >= 1900) && (CUDACapability >= 3.5) && (machineowner == \"Visics\") && (machine != \"amethyst.esat.kuleuven.be\" ) && (CUDADeviceName == 'GeForce GTX 960' || CUDADeviceName == 'GeForce GTX 980' )" >> $condor_file
#echo "Requirements = (CUDAGlobalMemoryMb >= 1900) && (CUDACapability >= 3.5)" >> $condor_file

echo "RequestDisk      = 25G" >> $condor_file
#wall time ==> generally assumed a job should take 6hours longest,
#if you want longer or shorter you can set the number of seconds. (max 1h ~ +3600s)
#100 hours means 4 days 
echo "+RequestWalltime = 360000" >> $condor_file 
#echo "+RequestWalltime = 10800" >> $condor_file
echo "">> $condor_file
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
echo "export PYTHONPATH=/users/visics/kkelchte/tensorflow/lib/python2.7/site-packages:/users/visics/kkelchte/tensorflow/examples">>$shell_file
echo "cd /users/visics/kkelchte/tensorflow/examples/pilot">>$shell_file
echo "echo 'went to directory ' $PWD">>$shell_file
echo "python $TASK">>$shell_file
echo "echo '$TASK has finished. \n \n description: $description.\n \n $condor_file' | mailx -s 'condor' klaas.kelchtermans@esat.kuleuven.be">>$shell_file

chmod 755 $condor_file
chmod 755 $shell_file

condor_submit $condor_file
echo $condor_file
