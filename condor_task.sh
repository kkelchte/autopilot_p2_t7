#!/usr/bin/bash
# This scripts sets some parameters for running a tasks,
# creates a condor and shell scripts and launches the stuff on condor.

# TASK="pilot_CNN.py --chosen_set $1" 	#wall_test
# description="_$1_cnn_feats"

# TASK='pilot_data_prepare.py' 	#wall_test
# description="_prepare_normalized_sequential_oa"

TASK='pilot_train.py' 	


#_______________________________________________________
### Inc n-FC reference to see if something is wrong
#model='inc_fc'
#dataset='sequential_oa'
#log_tag='check'
#stpfc='5'

#_______________________________________________________
### Inc FC
# model='inc_fc'
# log_tag='hard_d3'
# dataset='inc_fc_hard_dagger'


# fine='True'
# mod_dir='inc_fc_hard_rec_hsz_400_fc'


#_______________________________________________________
## Inc FC clipped
model='cut_inc_fc'
log_tag='hard_d3_cut'
dataset='inc_fc_hard_dagger'


#_______________________________________________________
### Inc n-FC
# model='inc_nfc'
# dataset='sequential_oa_hard_rec'
# # dataset='inc_fc_5_hard_dagger'
# log_tag='hard_5_rec'
# 
#_______________________________________________________
### Inc LSTM WW-TBPTT
# model='inc_lstm'
# dataset='inc_lstm_hard_dagger'
# log_tag='hard_d3'
# wsize='20'
# bsize='5'




# fine='True'
# mod_dir='inc_lstm_hard_rec_wsize_20'


#_______________________________________________________
### Inc LSTM sliding
#  model='sliding_lstm'
#  #dataset='sequential_oa_hard_rec'
#  dataset='inc_lstm_sliding_hard_dagger' 
#  log_tag='hard_d2'
#  wsize='20'
#  bsize='5'

 #_______________________________________________________
### Inc LSTM sliding with recovery
#  model='sliding_lstm'
#  dataset='sequential_oa_hard_rec'
#  #dataset='inc_lstm_sliding_rec_hard_dagger' 
#  log_tag='hard_rec'
#  wsize='20'
#  bsize='5'


#_______________________________________________________
### Subsampled LSTM
# model='inc_lstm'
# dataset='sequential_oa_rec'
# #dataset='sequential_oa_small'
# log_tag='recovery'
# # log_tag='s8_w3'
# # sample='1' #'8' #'4' #'2''1'
# wsize='20' #'10' #'3'    #'5'  #'10''20'
# bsize='32' #'32' #'64' #'256' #'128' #'64''32'

#_______________________________________________________
### Sequential OA with different training methods
#### sliding
# model='sliding_lstm'
# log_tag='small'
# dataset='sequential_oa_small'
# bsize='10'

#### fully unrolled
#model='fully_unrolled_lstm'
#log_tag='no_test'






#_______________________________________________________
### Sequential OA with adjusted annotations

# #### inc_lstm_dagger_adjusted
# andromeda='yes'
# model='inc_lstm'
# log_tag='seq_adj_dagger'
# dataset='inc_lstm_adj_dagger'

#_______________________________________________________
### Sequential OA with convolutional layers
### LSTM
#andromeda='yes'
#model='seq_end_lstm'
#log_tag='end_huge'

### FC ----> find learning rate
#model='seq_end_fc'
#log_tag='end_huge_5'
#stpfc='5'


# learning_rate="$1"

#_______________________________________________________
### Sequential OA with recovery cameras

####   inc_fc_dagger
# model='inc_fc'
# dataset='inc_fc_dagger'
# log_tag='seq_rec_d2'


# #### inc_lstm_dagger
# andromeda='yes'
# model='inc_lstm'
# dataset='inc_lstm_dagger'
# log_tag='seq_rec_d2'

#_______________________________________________________
### n-FC
# model='inc_fc'
# dataset='sequential_oa'
# log_tag='step5_exp'
#dataset='inc_fc_dagger_seq'
#log_tag='step5_dagger'
# stpfc=5

#_______________________________________________________
### Reference FC
# model='inc_fc'
# log_tag='no_test'

#dataset='inc_fc_dagger_seq'
#fine='True'
#mod_dir='inc_fc_seq_d6_fi_hsz_400_fc'
#_______________________________________________________
### Reference LSTM
#model='inc_lstm'
#log_tag='s8_w3'
#sample='8'
#wsize='3'
#dataset='sequential_oa'
#_______________________________________________________
### Clipped FC
#model='clip_inc_fc'
# log_tag='seq_exp'
# dataset='sequential_oa'
#log_tag='seq_dagger'
#dataset='inc_fc_dagger_seq'
#fine='True'
#mod_dir='inc_fc_seq_d6_fi_hsz_400_fc'

#_______________________________________________________
### Cut FC
# model='cut_inc_fc'
# # log_tag='seq_exp'
# # dataset='sequential_oa'
# log_tag='seq_dagger'
# dataset='inc_fc_dagger_seq'
# fine='True'
# mod_dir='inc_fc_seq_d6_fi_hsz_400_fc'

#_______________________________________________________
### Blue Inception LSTM
# model='inc_lstm'
# log_tag='seq_blue_d2'
# dataset='inc_lstm_blue_dagger'

#_______________________________________________________
### Sequential OA on pure depth images

####   depth_fc
# model='depth_fc'
# log_tag='seq'


# #### depth_lstm
#andromeda='yes'
# model='depth_lstm'
# log_tag='seq'

#_______________________________________________________
### ONE OA 

### LSTM
# andromeda='yes'
# model='one_inc_lstm'
# log_tag='rec_short'
# short_lbls='True'

### FC
# model='one_inc_fc'
# log_tag='rec_short'
# short_lbls='True'

### n-FC
#model='one_inc_fc'
#log_tag='rec5'
#stpfc=5
#_______________________________________________________
### LSTM (trained on better machines)
# andromeda='yes'
# 
# #### inc_lstm_dagger
# model='inc_lstm'
# log_tag='seq_d6'
# fine='True'
# mod_dir='inc_lstm_seq_d5_fi'
# dataset='inc_lstm_dagger'

#### inc_lstm_dagger_nof
# model='inc_lstm'
# log_tag='nof_d4'
# dataset='inc_lstm_nof_dagger'

#### inc_lstm_dagger_one
# model='inc_lstm'
# log_tag='f1_d1_straight'
# fine='True'
# mod_dir='inc_lstm_seq_d1_lr_0-05_fi'
# dataset='inc_lstm_f1_dagger'

#_______________________________________________________
####   depth_fc_dagger
#model='depth_fc'
#log_tag='seq' #d1
#fine='True'
#mod_dir='depth_fc_d1_fi_hsz_400_net_stijn_depth_fc'


# depth_lstm_dagger
#model='depth_lstm'
#log_tag='seq' #d1
# fine='True'
# mod_dir='depth_lstm_net_stijn_depth'

#_______________________________________________________
#Corridor challenge
#model='comb'
#log_tag='corridor'

#hidden_size='400' #"$2"
#batchwise="$2"
#fc='True' #"$3"

#feature_type="$5"
#network="$6"
#sample="$9"
#log_tag="${10}" #'no_test'
#mod_dir="${11}" #model dir to finetune

#keep_prob="$3"
#optimizer="$4"
#normalized="$5"
#random_order="$2"
#dataset=""

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
if [ ! -z $dataset ]; then 
	TASK="$TASK --dataset ${dataset}"
	#description="${description}_rand_$random_order"
fi
if [ ! -z $short_lbls ]; then 
	TASK="$TASK --short_labels ${short_lbls}"
	#description="${description}_rand_$random_order"
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
prescript_file="${temp_dir}/prescript${description}.sh"
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

echo "Requirements = ((machine != \"izar.esat.kuleuven.be\") && (machine != \"oculus.esat.kuleuven.be\")  && (machine != \"emerald.esat.kuleuven.be\"))" >> $condor_file
#echo "Niceuser = true" >> $condor_file

echo "Initialdir   = $temp_dir" >> $condor_file
echo "Executable   = $shell_file" >> $condor_file
#echo "+PreCmd      = \"$prescript_file\"" >> $condor_file
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
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-7.5/lib64:/users/visics/kkelchte/local/lib/cudnn-4.0x/cuda/lib64:">>$shell_file
echo "# run virtual environment for tensorflow">>$shell_file
echo "source /users/visics/kkelchte/tensorflow/bin/activate">>$shell_file
echo "# set python library path">>$shell_file
echo "export PYTHONPATH=/users/visics/kkelchte/tensorflow/lib/python2.7/site-packages:/users/visics/kkelchte/tensorflow/examples">>$shell_file
echo "cd /users/visics/kkelchte/tensorflow/examples/pilot">>$shell_file
echo "echo 'went to directory ' $PWD">>$shell_file
echo "python $TASK">>$shell_file
echo "echo '$TASK has finished. description: $description. $condor_file' | mailx -s 'condor' klaas.kelchtermans@esat.kuleuven.be">>$shell_file
#--------------------------------------------------------------------------------------------
echo "stamp=\$( date +\"%F-%T\" )">>$prescript_file
echo "if [ -e $condor_output_dir/condor${description}.log ] ; then mv $condor_output_dir/condor${description}.log $condor_output_dir/condor${description}_"'$stamp'".log; fi " >>$prescript_file
echo "if [ -e $condor_output_dir/condor${description}.err ] ; then mv $condor_output_dir/condor${description}.err $condor_output_dir/condor${description}_"'$stamp'".err; fi " >>$prescript_file
echo "if [ -e $condor_output_dir/condor${description}.out ] ; then mv $condor_output_dir/condor${description}.out $condor_output_dir/condor${description}_"'$stamp'".out; fi " >>$prescript_file
#--------------------------------------------------------------------------------------------

chmod 755 $condor_file
chmod 755 $shell_file
chmod 755 $prescript_file

condor_submit $condor_file
echo $condor_file
