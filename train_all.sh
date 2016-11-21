./condor_task.sh seq_oa_hard_inc_fc_d3
./condor_task.sh seq_oa_hard_inc_lstm_ww_d3

# ./tmp_scripts/preprocess_dagger.sh seq_oa_hard_inc_fc_5_d3
# ./tmp_scripts/preprocess_dagger.sh seq_oa_hard_inc_fc_d3
# ./tmp_scripts/preprocess_dagger.sh seq_oa_hard_inc_lstm_sliding_d3
# ./tmp_scripts/preprocess_dagger.sh seq_oa_hard_inc_lstm_ww_d3

#Some general training/testing architecture

#See if it is ok to delete the .tmp folder...
#rm -rf .tmp



###Loop over different windowsizes for wall challenge
# for i in $(seq 20 20 200);
# do
#     ./condor_task.sh $i $((500/$i)) r2 #with 500 the max size of the model (ws*bs)
# done

##STEP 1: pick CNN feats inception/pcnn app/flow/both
#                model bwise fc step_size_fnn ftype network wsize bsize sample log_tag

#./condor_task.sh cont False False 1 app inception 0 100 1 no_test
#./condor_task.sh cont False False 1 depht_estimate stijn 0 100 1 no_test

#./condor_task.sh cont True False 1 app inception 0 100 16 no_test
#./condor_task.sh cont False True 1 app inception 0 100 1 no_test
#./condor_task.sh cont False True 2 app inception 0 100 1 no_test
#./condor_task.sh cont False True 4 app inception 0 100 1 no_test

#./condor_task.sh cont True False 1 depht_estimate stijn 0 100 16 no_test
#./condor_task.sh cont False True 1 depht_estimate stijn 0 100 1 no_test
#./condor_task.sh cont False True 2 depht_estimate stijn 0 100 1 no_test
#./condor_task.sh cont False True 4 depht_estimate stijn 0 100 1 no_test

# ./condor_task.sh cwall False False 1 app inception 1 500 1 no_test
# ./condor_task.sh cwall False False 1 app inception 5 100 1 no_test
# ./condor_task.sh cwall False False 1 app inception 10 50 1 no_test
# ./condor_task.sh cwall False False 1 app inception 50 10 1 no_test
# ./condor_task.sh cwall False False 1 app inception 100 5 1 no_test
# ./condor_task.sh cwall False False 1 app inception 500 1 1 no_test
# ./condor_task.sh cwall True False 1 app inception 0 100 1 no_test

#./condor_task.sh big inception app False
#./condor_task.sh big inception flow
#./condor_task.sh big pcnn both
#./condor_task.sh big pcnn app True
#./condor_task.sh big pcnn app False
#./condor_task.sh big pcnn flow
# adapt network and feature type in default values of pilot code

##DAGGER 1
#./condor_task.sh dagger False False 1 app inception 0 100 1 d1 /esat/qayd/kkelchte/tensorflow/lstm_logs/cont
#./condor_task.sh dagger False False 1 depth_estimate stijn 0 100 1 d1 /esat/qayd/kkelchte/tensorflow/lstm_logs/cont_net_stijn_depth
#./condor_task.sh dagger False True 1 depth_estimate stijn 0 100 1 d1 /esat/qayd/kkelchte/tensorflow/lstm_logs/cont_batchwise_net_stijn_depth_fc
#./condor_task.sh dagger False True 1 app inception 0 100 1 d1 /esat/qayd/kkelchte/tensorflow/lstm_logs/cont_batchwise_fc

##STEP 1: choose learning rate
##              model learningrate
# ./condor_task.sh 0.00001
# ./condor_task.sh 0.001
# ./condor_task.sh 0.1

##STEP 2: choose hidden size
# ##              model hidden window batch
# ./condor_task.sh cont 20 100 1
# ./condor_task.sh cont 50 100 1
# ./condor_task.sh cont 100 100 1
# ./condor_task.sh cont 200 100 1


## adapt size in default values of pilot code

# ./condor_task.sh big 0.001
# ./condor_task.sh dumpster 0.001
# ./condor_task.sh dumpster 0.01
# ./condor_task.sh dumpster 0.1
# ./condor_task.sh dumpster 0.0001
# ./condor_task.sh dumpster 0.00001

##STEP 3: choose optimizer
##              model optimizir
#./condor_task.sh big Adam
#./condor_task.sh big RMSProp
#./condor_task.sh big GradientDescent
#./condor_task.sh big Momentum
#./condor_task.sh big Adagrad
## adapt size in default values of pilot code





