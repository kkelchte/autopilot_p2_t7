"""
main script for training the LSTM model on features
Uses pilot_data, pilot_eval, pilot_settings, pilot_model, pilot_states
"""
#from __future__ import division
import pilot_data
import pilot_model
import pilot_settings
import pilot_states
import pilot_eval

import random
import time
import scipy.io as sio

import tensorflow as tf
import numpy as np

import math
import shutil
import sys, os

from sklearn import metrics

logging = tf.logging

FLAGS = tf.app.flags.FLAGS

##If you change default settings please modify the interpret_flags function
##in order to set your logfolder to a proper name...
tf.app.flags.DEFINE_boolean(
    "save_states_train", False,
    "Whether or not the innerstates are saved away during training.")
tf.app.flags.DEFINE_boolean(
    "validate_always", False,
    "Whether or not every epoch should be followed by a validation run.")
tf.app.flags.DEFINE_boolean(
    "random_order", True,
    "Whether or not the batches during 1 epoch are shuffled in random order.")
tf.app.flags.DEFINE_float(
    "learning_rate", "0.0001",
    "Define the learning rate for training.")
## =1/1.15
tf.app.flags.DEFINE_float(
    "lr_decay", "0.86957", 
    "Define the rate at which the learning rate decays.")
tf.app.flags.DEFINE_float(
    "init_scale", "0.1",
    "Define the scale from which random initial weights are chosen.")
tf.app.flags.DEFINE_integer(
    "max_epoch", "50",
    "Number of epochs after which the learning rate starts decaying.")
tf.app.flags.DEFINE_integer(
    "max_max_epoch", "100",
    "Maximum of total number of epochs, after this and a test run the program is finished.")
tf.app.flags.DEFINE_boolean(
    "batchwise_learning", False,
    "Whether the network is trained fully unrolled according to the movie length of the grouped batch.\
    Else it is trained in a windowwize manner unrolling the network these x steps in time (the windowsize) \
    The batchsize is set according to the windowsize and the size of the GPU.")
tf.app.flags.DEFINE_integer(
	"window_size", 0,
	"Define what windowsize is used according to the required memory span of the task.\
	If kept to default value (0) than windowsizes and batchsizes are picked in order to fit on a 2G GPU.")
tf.app.flags.DEFINE_float(
    "scale", "4",
    "NOTUSED In case of windowwize learning, the batchsizes can be scaled according to the size of RAM and GPU.\
    For a 2Gb GPU and 16Gb Ram the scale should be around 4.")
tf.app.flags.DEFINE_boolean(
    "finetune", False,
    "Define wheter there should be a model loaded from init_model_dir to finetune or you want to train from scratch."
    )
tf.app.flags.DEFINE_string(
    "init_model_dir", "/esat/qayd/kkelchte/tensorflow/lstm_logs/remote_set7_sz_100_net_inception",
    "location of initial model on which will be finetuned in case of finetuning."
    )
tf.app.flags.DEFINE_integer(
    "batch_size_fnn", 100,
    "set the amount of data in one mini batch for training the FNN. In one epoch this minibatch is that many times\
    applied so it covers in general all the data available."
    )

def run_batch_fc(session, model, data, targets, eval_op, writer=None, verbose=False):
    ''' Run one time through the data given in a batch. fc stands for fully connected as we re only training the last layers of an FNN.
    Args:
            session: current session in which operations are defined
            model: model object that contains operations and represents network
            data: an np array containing 1 batch of data [batchsize, num_steps, feature_dimension]
            targets: [batch, steps, output_size]
    '''
    #print "train FNN... datashape: ", data.shape, "targets shape: ", targets.shape
    start_time = time.time()
    batch_lengths = []
    feed_dict = {model.inputs: data[:,0:1,:], model.targets: targets[:,0:1,:]}
    
    # call the network: forward and backward pass
    outputs, costs, _ = session.run([model.logits, model.cost, eval_op],feed_dict)
    
    trgts = targets.reshape((outputs.shape[0],outputs.shape[1]))
    total = data.shape[0]
    
    if FLAGS.continuous:
        # get the sum of the 2 norm difference between the targets and the outputs over the different batches and timesteps
        #score = [sum([sum((outputs[i,:]-trgts[i,:])**2) for i in range(outputs.shape[0])])]
        #score_mae = sum([metrics.mean_absolute_error(trgts[i,:], outputs[i,:]) for i in range(outputs.shape[0])])
        #score_mse = sum([metrics.mean_squared_error(trgts[i,:], outputs[i,:]) for i in range(outputs.shape[0])])
        #score_mdae = sum([metrics.median_absolute_error(trgts[i,:], outputs[i,:]) for i in range(outputs.shape[0])])
        #print 'mae: ',score_mae,'. mse: ',score_mse,'. mdae: ',score_mdae
        score = metrics.mean_squared_error(trgts, outputs)*total
        #print 'score ',score
    else:
        score = float(sum((np.argmax(trgts, axis=1)==np.argmax(outputs, axis=1))))
    if verbose:
        if FLAGS.continuous: print "Score: ",score, '. Fps:', total/(time.time() - start_time), '. Loss: ',costs
        else: print "Accuracy: ",float(score/total), '. Fps:', total/(time.time() - start_time)
        
    return score, costs, total
    
def run_batch_unrolled(session, model, data, targets, eval_op, state, writer=None, verbose=False, number_of_samples=500, matfile = ""):
    """
    Args:
        session: current session in which operations are defined
        model: model object that contains operations and represents network
        data: an np array containing 1 batch of data [batchsize, num_steps, feature_dimension]
        targets: [batch, steps, 4]
    """
    
    # run one time through the data given in a batch of all equal length.
    # network is fully unrolled. Used during training
    num_steps=data.shape[1]
    print "Fully unroll... over ", num_steps, " steps."
    start_time = time.time()
    costs = 0.0
    iters = 0.0
    score = 0.0
    #import pdb; pdb.set_trace()
    # prepare data to be fed into the model 
    feed_dict = {model.inputs: data, 
                    model.initial_state: state, 
                    model.targets: targets}
    # call the network: forward and backward pass
    #outputs, states, costs, _, outputs_lstm, loss_w, lgts = session.run([model.logits, model.states, model.cost, eval_op, model.output, model.loss_wild, model.rsh_lgts],feed_dict)
    #import pdb; pdb.set_trace()
    outputs, states, costs, _ = session.run([model.logits, model.states, model.cost, eval_op],feed_dict)
    # reshape outputs, get score
    trgts = targets.reshape((outputs.shape[0],outputs.shape[1]))
    total = data.shape[0]*num_steps
    if FLAGS.continuous:
        # get the sum of the 2 norm difference between the targets and the outputs over the different batches and timesteps
        #score = [sum([sum((outputs[i,:]-trgts[i,:])**2) for i in range(outputs.shape[0])])]
        score = metrics.mean_squared_error(trgts, outputs)*total
    else:
        score = float(sum((np.argmax(trgts, axis=1)==np.argmax(outputs, axis=1))))
    # Keep track of cell states
    if(writer != None) and (matfile != "") and FLAGS.save_states_train:
        #states = [batchsize, num_steps*hidden_size*num_layers*2 (~output;state)]
        d = {'states': states, 'targets':trgts}
        sio.savemat(matfile,d, appendmat=True)
    btime = time.time()
    # Do all the summary writing, both for states as summary of network (histogram of activations)
    if(writer != None) and FLAGS.save_states_train:
        with tf.device('/cpu:0'):
            state_images = pilot_states.get_image(states, targets, num_steps, model.hidden_size, model.num_layers, model.batch_size)
            im_op = tf.image_summary('Innerstates', state_images, max_images=9)
            summary_str = session.run(tf.merge_summary([im_op]))
            writer.add_summary(summary_str)
            print "state image: ", time.time() - btime, " sec"
    # Write the predicted steps away: first movie in case of batch: 
    # output of different movies in batch is concatenated in time direction .
    # should make this dynamically callable as this slows things down a lot.
    # also not interesting for 1 step unrolled learning
    for i in range(num_steps):
        pred = np.argmax(outputs[i,:])
        trgt = np.argmax(trgts[i,:])
        if writer != None:
            pred_op = tf.scalar_summary(model.prefix+"_predictions", pred)
            trgt_op = tf.scalar_summary(model.prefix+"_targets", trgt)
            summary_str = session.run(tf.merge_summary([pred_op, trgt_op]))
            writer.add_summary(summary_str,i)
        #if verbose and (i % int(num_steps/15) == 0):
        #    print("Frame: %d target: %d prediction: %d"%(i, trgt, pred))        
    if verbose:
        if FLAGS.continuous: print "Score: ",score, '. Fps:', total/(time.time() - start_time), '. Loss: ',costs
        else: print "Accuracy: ",float(score/total), '. Fps:', total/(time.time() - start_time)
    return score, costs, total
   
def run_epoch(is_training, session, models, data_list, config, location="", epoch_index=0, frame_writer=None, window_sizes=None, batch_sizes=None, mvalid=None):
    ''' Go 1 epoch through training or validation keeping track of the minimum maximum and average accuracy
    and perplexity. Things are written away in detail with the frame_writer for some epochs and in general with the overview_writer.
    Args:
        is_training: boolean defining whether this is the training part of the epoch
        session: current training session
        models: a list of models unrolled over different number of steps and
        with different batchsizes
        data_list: list with data tuples
        config: configuration containing info about number of outpus and feature dimension
        location: the location to write the general accuracy and perplexity values of this epoch number
        i: the epoch number
        frame_writer: the SummaryWriter object for writing away the predictions and targets at frame rate
        window_indices: a list of indices pointing to the windows filling each batch for each model
        window_sizes: the num_steps the window should contain.
    Return:
    	model_results: return a list of results for each model. The results is the average in case of multiple runs for one model.
    	The results contain [score, cost, total]. With total the number of frames it evaluated in that window-batch.
    '''
    window_indices = None
    all_initial_states = None
    if not FLAGS.batchwise_learning and is_training: # In case of windowwise training
    	# make a nested list of tuples containing batches with randomly picked windows of data
        # the indices are starting indices of the windows in certain data trajectories 
        # used for finding the initial state of the models
        # the outer nested list corresponds to the different data models
        # the inner nested list corresponds to different tuples for that model according to the size of the data
        # the more the data ==> the more index tuples in the inner list
        # the bigger the gpu ==> the more models ==> the larger the outer list
        if FLAGS.sliding_window:
            window_indices = pilot_data.pick_sliding_windows(data_list, window_sizes, batch_sizes)
        else:
            window_indices = pilot_data.pick_random_windows(data_list, window_sizes, batch_sizes)
        print 'finished picking window indices'
        #import pdb; pdb.set_trace()
        # get the initial innerstate of the models according to the windows
        if not FLAGS.fc_only: 
            before_time = time.time()
            #if the total number of initial state calculation is bigger than 200, do it multi threaded
            if sum([len(window_indices[i])*window_indices[i][0].shape[0] for i in range(len(window_indices))]) > 20:
                all_initial_states = pilot_eval.get_initial_states(models, window_indices, mvalid, data_list, session)
            else: #calculate initial states single threaded
                all_initial_states = pilot_eval.get_initial_states_single(models, window_indices, mvalid, data_list, session)
            #print 'old duration', print_time(after_time), ' from ',print_time(before_time)
            #import pdb; pdb.set_trace()
            #if np.amax(all_initial_states[0][0]-all_initial_states_old[0][0]) != 0:
            #    raise IndexError('multithreaded innerstate calculations failed')
            print 'finished getting initial states: ', print_time(before_time)
        #res_state = session.run(trainingmodels[0].initial_state)
        
    #choose 1 model/batch from which the first movie is recorded
    #in case of validation each batch contains only 1 movie
    if window_indices:
        chosen_batch = int(random.uniform(0,len(models))) 
        model_indices = range(len(models))
    else:
        chosen_batch = 0
        model_indices = [0]
    #    chosen_batch = int(random.uniform(0,len(data_list))) 
    #    model_indices = range(len(data_list))
    #pick the models to train first from randomly
    if is_training and FLAGS.random_order: random.shuffle(model_indices)
    #import pdb; pdb.set_trace()
    model_results = []
    result_list=[]
    for model_index in model_indices:
        if model_index == chosen_batch : writer = frame_writer
        else : writer = None
        # In case we are training...
        if is_training:
            # if batchwise training loop over tuples containing batches
            mtrain = models[model_index]
            # learning rate goes down during learning
            lr_decay = FLAGS.lr_decay ** max(epoch_index - FLAGS.max_epoch, 0.0)
            mtrain.assign_lr(session, FLAGS.learning_rate * lr_decay)
            print("Epoch: %d Learning rate: %.5f" % (epoch_index + 1, session.run(mtrain.lr)))
            #The next batch is copied into RAM   
            #windowwise learning aka truncated BPTT 
            if not FLAGS.batchwise_learning and window_indices and window_sizes:
                #import pdb; pdb.set_trace()
                #loop over the index-tuples that relate to the model_index-th model
                for inner_list_index in range(len(window_indices[model_index])):
                    data, targets = pilot_data.copy_windows_from_data(data_list, window_sizes[model_index], window_indices[model_index][inner_list_index])
                    if FLAGS.fc_only: #in case of fully connected only
                        results = run_batch_fc(session, mtrain, data, targets, mtrain.train_op, writer=writer)
                    else: #normal LSTM
                        initial_state = all_initial_states[model_index][inner_list_index]
                        results = run_batch_unrolled(session, 
                                   mtrain, 
                                   data, #data 
                                   targets, #targets
                                   mtrain.train_op, #eval op
                                   initial_state,
                                   writer = writer, #frame writer
                                   verbose = False) #matfile=location+"/trainstates_"+str(i)+".mat")
                    result_list.append(results)
                results=[]
                for r in range(len(result_list[0])):#average over batches that belong to this model
                    results.append(sum([res[r] for res in result_list])/len(result_list))
                #print 'Average loss over window ',model_index,': ', results[1]
                #import pdb;pdb.set_trace()
            else: #batchwise learning aka fully unrolled BPTT
                data = data_list[model_index][0] #take data out of model_index-th batch-tuple (grouped)
                targets = data_list[model_index][1] #take labels out of model_index-th batch-tuple (grouped)
                # initialize network with 0 initial state
                initial_state = session.run(mtrain.initial_state)
                results = run_batch_unrolled(session, 
                        mtrain, 
                        data, #data 
                        targets, #targets
                        mtrain.train_op, #eval op
                        initial_state,
                        writer = writer, #frame writer
                        verbose = False) #matfile=location+"/trainstates_"+str(i)+".mat")
        # In case of validation...
        else:
            #define model
            mvalid = models[0]
            results = pilot_eval.run_one_step_unrolled(session, 
                                                       mvalid, 
                                                       data_list[model_index][0],#one movie is copied into ram... 
                                                       data_list[model_index][1], 
                                                       tf.no_op(), 
                                                       writer = writer, 
                                                       verbose = False) #matfile=location+"/valstates_"+str(epoch_index)+".mat")
            
        model_results.append(results)
    if len(model_results) == 1 and result_list:
        return result_list
    else:
        return model_results
    
def define_batch_sizes_for_stepwise_unrolling(data_list):
    '''creates a list of sizes starting from size 1 
    going till the lenght ofthe largest movie among the data_list
    The size is the number of movies that can be used for training
    these number of unrolled steps
    Args:
        datalist: list of tuples containing 1 movie in 1 tuple with features and labels
    Returns:
        list of sizes of the batches neede for defining the network batchsize
    '''
    # Get a list of all batchsizes depending on the lengths of the movies:
    lengths = [ t[0].shape[1] for t in data_list ]
    sizes = []
    batch_s = len(lengths) #batch size ~ number of movies in one batch
    print lengths
    for i in range(lengths[-1]):
        if i < lengths[0]:
            #first 150(smallest movie) models have a full batch
            sizes.append(batch_s)
        else:
            #cut the first elements having i-size
            while i==lengths[0]:
                lengths = lengths[1:]
                batch_s = batch_s-1
            print batch_s
            sizes.append(batch_s)
    return sizes


def print_time(start_time):
    '''Print the time passed after the start_time defined in hours, minutes and seconds
    Arg:
        start_time: the moment from which you started counting the time.
    Returns:
        string with time message.
    '''
    duration = (time.time()-start_time)
    m, s = divmod(duration, 60)
    h, m = divmod(m, 60)
    return "time: %d:%02d:%02d" % (h, m, s)


def main(_):
    start_time = time.time()
    
    # Get configuration chosen by the model flag ~ reset some predefined flags,
    # it defines the movies on which is trained/tested/validated,
    config = pilot_settings.get_config()
    
    
    #import pdb; pdb.set_trace()
    #extract a good name of the logfolder depending on the flags
    logfolder = pilot_settings.extract_logfolder()
    #print "train: ",config.training_objects
    #print "validate: ",config.validate_objects
    #print "test: ",config.test_objects

    # Delete logfolder if its already used, otherwise it gets huge
    if os.path.isdir(logfolder) and FLAGS.log_tag != 'testing':
        raise NameError( 'Logfolder already exists, overwriting alert: '+ logfolder )
    else :      
        shutil.rmtree(logfolder,ignore_errors=True)
    
    print 'Logfolder: ', logfolder
    os.mkdir(logfolder)
    #save configuration
    pilot_settings.save_config(logfolder)
    
    #import pdb; pdb.set_trace()
    # List of sorted(already done in txt file) tuples (data, labels), each tuple corresponds to one training movie
    # data is an array of shape [batch_of_movies, movie_length, feature_dimension]
    # labels is an array of shape [batch_of_movies, movie_length, output_dimension]
    data_time = time.time()
    training_data_list = pilot_data.prepare_data_general(config.training_objects)
    #else : training_data_list = pilot_data.prepare_data_list(config.training_objects)
    #validation set is always one movie with batchsize one and tested with stepsize 1
    if len(config.validate_objects) != 0:
        validate_data_list = pilot_data.prepare_data_list(config.validate_objects)
    print 'loaded data, ',print_time(data_time)
    
    #set params according to the shape of the obtained data
    config.output=training_data_list[0][1].shape[2]
    config.feature_dimension=training_data_list[0][0].shape[2]
    if FLAGS.fc_only:
        config.feature_dimension=FLAGS.step_size_fnn*training_data_list[0][0].shape[2]
        
    #import pdb; pdb.set_trace()
    # Tell TensorFlow that the model will be built into the default Graph.
    #with tf.Graph().as_default(), tf.Session() as session:
    #with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(log_device_placement=True, log_device_placement=True)) as session, tf.device('/cpu:0'):
    start_time = time.time()
        
    # Define the batchsizes and window_sizes (numsteps) for the different models during training
    if FLAGS.batchwise_learning:
        batch_sizes = [training_tuple[0].shape[0] for training_tuple in training_data_list]
        window_sizes = [training_tuple[0].shape[1] for training_tuple in training_data_list]
    else: # in case of windowwise learning
        #window_sizes = [50, 100, 500] 
        #batch_sizes = [30, 15, 3] #12G
        
        #window_sizes = [50, 100, 500]#4G
        #batch_sizes = [10, 5, 1] 
        window_sizes = [20, 40, 80]#4G
        batch_sizes = [32, 16, 8]
        
        #window_sizes = [300, 100, 30]
        #batch_sizes = [2, 5, 15] 
        
        #window_sizes = [1, 5, 10, 50, 100, 500]
        #batch_sizes = [100, 50, 20, 8, 4, 1] #4G 
        #batch_sizes = [200, 50, 20, 8, 4, 1] #4G 
        #batch_sizes = [600, 150, 60, 24, 12, 3] #12G
        
        if FLAGS.window_size != 0:
            window_sizes = [FLAGS.window_size]
            batch_sizes = [FLAGS.batch_size_fnn]
        if FLAGS.model == "small":
            window_sizes = [5,10] 
            batch_sizes = [10, 5] 
        if FLAGS.fc_only: #in case of training FNN no timesteps are needed
            window_sizes = [1]
            batch_sizes = [FLAGS.batch_size_fnn]
    print 'batch_sizes ',batch_sizes
    print 'window_sizes ',window_sizes
    
    #Train LSTM starting on a each windowsize max_max amount of epochs [could be looped later: ]
    for windex in range(len(window_sizes)):
        window_start_time = time.time()
        print "started window number ",windex," with wsize: ",window_sizes[windex]," and batchsize: ",batch_sizes[windex]
        
        g = tf.Graph()
        with g.as_default():
            
            # Define initializer
            initializer = tf.random_uniform_initializer(-FLAGS.init_scale, FLAGS.init_scale)
            #initializer = tf.constant_initializer(1.0)
            
            # Build the LSTM Graph with 1 model for training, validation
            trainingmodels = []
            with tf.variable_scope("model", reuse=False, initializer=initializer) as model_scope:
                config.batch_size = batch_sizes[windex]
                config.num_steps = window_sizes[windex]
                mtrain = pilot_model.LSTMModel(True, config.output, config.feature_dimension, 
                                            config.batch_size, config.num_steps, 'train')
                trainingmodels.append(mtrain)
            # Reuse the defined weights but initialize with random_uniform_initializer
            with tf.variable_scope(model_scope, reuse=True, initializer=initializer): 
                #if FLAGS.batchwise_learning:
                    #for i in range(1,len(batch_sizes)):
                        #config.batch_size=batch_sizes[i]
                        #config.num_steps=window_sizes[i]
                        #mtrain = pilot_model.LSTMModel(True, config.output, config.feature_dimension, 
                                                #config.batch_size, config.num_steps, 'train')
                        #trainingmodels.append(mtrain)       
                FLAGS.gpu = False
                mvalid = pilot_model.LSTMModel(False, config.output, config.feature_dimension, prefix='eval')
            
            print "Loading models finished... ", print_time(start_time)
            print "Number of models: ", len(trainingmodels)
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
                        
            # Define session and initialize
            session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            #session = tf.Session(config=tf.ConfigProto(allow_soft_placement=False))
            
            # Initialize all variables (weights and biases)
            if FLAGS.finetune:
                print 'restoring session from folder: ', FLAGS.init_model_dir
                saver.restore(session, '/esat/qayd/kkelchte/tensorflow/lstm_logs/'+FLAGS.init_model_dir+"/model.ckpt")
                print "model restored: ",FLAGS.init_model_dir+"/model.ckpt"
            else:
                init = tf.initialize_all_variables()
                session.run(init)
            #import pdb; pdb.set_trace()
            
            # define overview_writer
            # create some tensor op for calculating and keeping the accuracy/loss over different epochs
            writer_overview= tf.train.SummaryWriter(logfolder+'/overview'+str(windex), graph=session.graph) 
            average_loss=tf.placeholder(tf.float32)
            max_loss=tf.placeholder(tf.float32)
            min_loss=tf.placeholder(tf.float32)
            al=tf.scalar_summary("average_loss",average_loss)
            mal=tf.scalar_summary("max_loss",max_loss)
            mil=tf.scalar_summary("min_loss",min_loss)
            average_score=tf.placeholder(tf.float32)
            max_score=tf.placeholder(tf.float32)
            min_score=tf.placeholder(tf.float32)
            a_s=tf.scalar_summary("average_score",average_score)
            ma_s=tf.scalar_summary("max_score",max_score)
            mi_s=tf.scalar_summary("min_score",min_score)
            merge = tf.merge_summary([al, mal, mil, a_s, ma_s, mi_s])
            
            #Train while model gets better with upperbound max_max_epoch as number of runs
            for current_epoch in range(FLAGS.max_max_epoch):
                #every 1/5/10/50/100/500/1000 epochs it should write the current data away
                writer = None
                validate = False #only validate at some points during traing but not every epoch
                # write different epochs away as different logs in a logarithmic range till max_max_epoch
                #if current_epoch in (b*10**exp for exp in range(1+int(math.log(FLAGS.max_max_epoch,10))) for b in [1,2,5]):
                if current_epoch in [1,2,5,10,20,50,100]:
                    if current_epoch!= 1:
                        #delete logdata of previous run except for the first run
                        location = logfolder+"/e"+str(current_epoch)
                        shutil.rmtree(location,ignore_errors=True) #it might delete the wrong folder...		
                    #Dont validate, in an obstacle world that doesn't make sense
                    validate=True
                    #print "save epoch: ", current_epoch+1
                    location = logfolder+"/e"+str(current_epoch)
                    #writer = tf.train.SummaryWriter(location, graph=session.graph)
                #Always validate the last run
                if current_epoch == (FLAGS.max_max_epoch-1):
                    validate=True
                    
                #Train
                train_results = []
                epoch_time=time.time()
                if FLAGS.batchwise_learning:
                    train_results = run_epoch(True, session, trainingmodels, [training_data_list[windex]], config, logfolder, current_epoch, writer, [window_sizes[windex]], [batch_sizes[windex]], mvalid)
                else:
                    train_results = run_epoch(True, session, trainingmodels, training_data_list, config, logfolder, current_epoch, writer, [window_sizes[windex]], [batch_sizes[windex]], mvalid)
                print "Training finished... ", print_time(epoch_time), "."
                scores = [mres[0] for mres in train_results]
                losses = [mres[1] for mres in train_results]
                print("Epoch: %d Average loss over different unrolled models: %f, Max: %f, Min: %f. Average score: %f, Max: %f, Min: %f" % (current_epoch + 1,sum(losses)/len(losses),max(losses),min(losses),sum(scores)/len(scores),max(scores),min(scores)))
                #write final validation away in txt file.
                if current_epoch == (FLAGS.max_max_epoch-1):
                    f = open(logfolder+"/results.txt", 'a')
                    f.write('training: Average loss: %f, Max: %f, Min: %f.\n' % (sum(losses)/len(losses),max(losses),min(losses)))
                    f.close()
                
                # write away results of this epoch
                if writer_overview: 
                    feed_dict = {average_loss: sum(losses)/len(losses), max_loss: max(losses), min_loss: min(losses), average_score: sum(scores)/len(scores), max_score: max(scores), min_score: min(scores)}
                    summary_str = session.run(merge, feed_dict)
                    writer_overview.add_summary(summary_str, current_epoch+windex*FLAGS.max_max_epoch)
                
                #only validate in few cases as it takes a lot of time setting the network on the GPU and evaluating all the movies
                if (validate or FLAGS.validate_always) and len(config.validate_objects)!= 0:
                    val_results = []
                    #Validate
                    epoch_time=time.time()
                    val_results = run_epoch(False, session, [mvalid], validate_data_list, config, logfolder, current_epoch, writer)
                    print "Validation finished... ",print_time(epoch_time)
                    scores = [mres[0] for mres in val_results]
                    losses = [mres[1] for mres in val_results]
                    print("Epoch: %d Average loss over different unrolled models: %f, Max: %f, Min: %f. Average score: %f, Max: %f, Min: %f" % (current_epoch + 1,sum(losses)/len(losses),max(losses),min(losses),sum(scores)/len(scores),max(scores),min(scores)))
                    #write final validation away in txt file.
                    if current_epoch == (FLAGS.max_max_epoch-1):
                        f = open(logfolder+"/results.txt", 'a')
                        f.write('validation: Average loss: %f, Max: %f, Min: %f.\n' % (sum(losses)/len(losses),max(losses),min(losses)))
                        f.close()
                    
                sys.stdout.flush()
                #save session every 5Es 
                #if (current_epoch%5) == 0 or (current_epoch==FLAGS.max_max_epoch-1) or (current_epoch == 2):
                if current_epoch==FLAGS.max_max_epoch-1:
                    if os.path.isfile(logfolder+"/model.ckpt"):
                        shutil.move(logfolder+"/model.ckpt", logfolder+"/model_"+str(current_epoch+windex*FLAGS.max_max_epoch)+".ckpt")
                    # Save the variables to disk.
                    save_path = saver.save(session, logfolder+"/model.ckpt")
                    print("Model %d saved in file: %s" %( current_epoch+windex*FLAGS.max_max_epoch, save_path ))
                    # Use this model for finetuning in case of several windowsizes:
                    head, FLAGS.init_model_dir = os.path.split(logfolder)
                    FLAGS.finetune = True
            # Free all memory after training
            session.close()
            saver = None
            writer_overview = None
            merge = None
            if not FLAGS.batchwise_learning:
                print 'windowsize: ', window_sizes[windex],' is finished after: ',print_time(window_start_time),'.'
        g = None
    # Free some memory
    training_data_list = None
    validate_data_list = None
    #if os.path.isfile(logfolder+"/model.ckpt"):
        #os.remove(logfolder+"/model.ckpt")
    ## Save the variables to disk.
    #save_path = saver.save(session, logfolder+"/model.ckpt")
    #print("Final model saved in file: %s" % save_path)
        
    # Evaluate the model
    FLAGS.model_dir = logfolder
    pilot_eval.evaluate(logfolder=logfolder, config=config, scope="model")
        
    # Give some nice feedback on how long it all took
    print "DONE... ", print_time(start_time)
    
if __name__ == '__main__':
    tf.app.run() 
    
    
