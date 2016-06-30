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
import sys

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
    "learning_rate", "0.001",
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
    "batchwise_learning", True,
    "Whether the network is trained fully unrolled according to the movie length of the grouped batch.\
    Else it is trained in a windowwize manner unrolling the network the x steps (the windowsize) \
    The batche size is set according to the windowsize.")
tf.app.flags.DEFINE_float(
    "scale", "4",
    "In case of windowwize learning, the batchsizes can be scaled according to the size of RAM and GPU.\
    For a 2Gb GPU and 16Gb Ram the scale should be around 4.")


#def run_epoch(session, model, data, targets, eval_op, merge_op=tf.no_op(), writer=None, verbose=False):


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
    outputs, states, costs, _ = session.run([model.logits, model.states, model.cost, eval_op],feed_dict)
    # reshape outputs, get score
    trgts = targets.reshape((outputs.shape[0],outputs.shape[1]))
    score = float(sum((np.argmax(trgts, axis=1)==np.argmax(outputs, axis=1))))
    total = data.shape[0]*num_steps
    
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
        if verbose and (i % int(num_steps/15) == 0):
            print("Frame: %d target: %d prediction: %d"%(i, trgt, pred))        
    if verbose:
        print "Accuracy: ",float(score/total), '. Fps:', total/(time.time() - start_time)
    return score, costs, total
   
def run_epoch(is_training, session, models, data_list, config, location="", i=0, frame_writer=None, window_indices=None, window_sizes=None, all_initial_states=None):
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
    '''
    # define overview_writer
    # create some tensor op for calculating and keeping the accuracy over different epochs
    if location != "":
        accuracy_av = tf.placeholder(tf.float32)
        accuracy_max_min = tf.placeholder(tf.float32)
        perplexity_av = tf.placeholder(tf.float32)
        if is_training: namescope = "overview_training"
        else: namescope = "overview_validation"
        acc_sum = tf.scalar_summary(namescope+"_accuracy_av", accuracy_av)
        acc_max_min_sum = tf.scalar_summary(namescope+"_accuracy_maxmin", accuracy_max_min)
        per_sum = tf.scalar_summary(namescope+"_perplexity", perplexity_av)
        merge = tf.merge_summary([acc_sum, acc_max_min_sum, per_sum])
        location = location+"/overview"
        writer_overview= tf.train.SummaryWriter(location, graph=session.graph) 
    
    acc = 0.0
    acc_max = 0.0
    acc_min = 1.0
    per = 0.0
    winner_index = -1
    lozer_index = -1
    
    #choose 1 model/batch from which the first movie is recorded
    #in case of validation each batch contains only 1 movie
    if window_indices:
        chosen_batch = int(random.uniform(0,len(models))) 
        indices = range(len(models))
    else:
        chosen_batch = int(random.uniform(0,len(data_list))) 
        indices = range(len(data_list))
    #pick the models to train first from randomly
    if is_training and FLAGS.random_order: random.shuffle(indices)
    for j in indices:
        if j == chosen_batch : writer = frame_writer
        else : writer = None
        results = []
        # In case we are training...
        if is_training:
            # if batchwise training loop over tuples containing batches
            #import pdb; pdb.set_trace()
            mtrain = models[j]
            # learning rate goes down during learning
            lr_decay = FLAGS.lr_decay ** max(i - FLAGS.max_epoch, 0.0)
            mtrain.assign_lr(session, FLAGS.learning_rate * lr_decay)
            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(mtrain.lr)))
            #The next batch is copied into RAM
            if not FLAGS.batchwise_learning and window_indices and window_sizes and all_initial_states:
                #import pdb; pdb.set_trace()
                data, targets = pilot_data.copy_windows_from_data(data_list, window_sizes[j], window_indices[j])
                initial_state = all_initial_states[j]
            else:
                data = data_list[j][0] 
                targets = data_list[j][1]
                # initialize network with 0 initial state
                initial_state = session.run(mtrain.initial_state)
                
            results = run_batch_unrolled(session, 
                                         mtrain, 
                                         data, #data 
                                         targets, #targets
                                         mtrain.train_op, #eval op
                                         initial_state,
                                         writer = writer, #frame writer
                                         verbose = False,
                                         matfile=location+"/trainstates_"+str(i)+".mat")
        # In case of validation...
        else:
            #define model
            mvalid = models[0]
            results = pilot_eval.run_one_step_unrolled(session, 
                                                       mvalid, 
                                                       data_list[j][0],#one movie is copied into ram... 
                                                       data_list[j][1], 
                                                       tf.no_op(), 
                                                       writer = writer, 
                                                       verbose = False, 
                                                       matfile=location+"/valstates_"+str(i)+".mat")
        perplexity = np.exp(results[1] / results[2])
        accuracy = results[0]/results[2]+0.000
        #print("Train Accuracy: %.3f" % ( train_accuracy))
        if acc_max < accuracy: 
            acc_max = accuracy
            winner_index = j
        if acc_min > accuracy: 
            acc_min = accuracy
            lozer_index = j
        acc = acc+accuracy/len(indices)
        per = per+perplexity/len(indices)
    
    # write away results of this epoch
    if writer_overview: 
        feed_dict = {accuracy_av : acc, perplexity_av: per, accuracy_max_min: acc_max}
        summary_str = session.run(merge, feed_dict)
        writer_overview.add_summary(summary_str, i)
        #summary_str = session.run(acc_max_min_sum,{accuracy_max_min: acc_min})
        #writer_overview.add_summary(summary_str, i)
        print("Epoch: %d Final Accuracy: %.3f, Max: %.3f at index %d, Min: %.3f at index %d." % (i + 1, acc, acc_max, winner_index, acc_min, lozer_index))
    sys.stdout.flush()

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
    
    #extract a good name of the logfolder depending on the flags
    logfolder = pilot_settings.extract_logfolder()
    
    # Delete logfolder if its already used, otherwise it gets huge
    shutil.rmtree(logfolder,ignore_errors=True)
    print 'Logfolder: ', logfolder
    
    # Get configuration chosen by the model flag ~ reset some predefined flags,
    # it defines the movies on which is trained/tested/validated,
    config = pilot_settings.get_config()
    
    # List of sorted(already done in txt file) tuples (data, labels), each tuple corresponds to one training movie
    # data is an array of shape [batch_of_movies, movie_length, feature_dimension]
    # labels is an array of shape [batch_of_movies, movie_length, output_dimension]
    data_time = time.time()
    training_data_list = pilot_data.prepare_data_grouped(config.training_objects)
    #else : training_data_list = pilot_data.prepare_data_list(config.training_objects)
    validate_data_list = pilot_data.prepare_data_list(config.validate_objects)
    print 'loading data, ',print_time(data_time)
    
    #set params according to the shape of the obtained data
    config.output=training_data_list[0][1].shape[2]
    config.feature_dimension=training_data_list[0][0].shape[2]
    
    # Tell TensorFlow that the model will be built into the default Graph.
    #with tf.Graph().as_default(), tf.Session() as session:
    #with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(log_device_placement=True, log_device_placement=True)) as session, tf.device('/cpu:0'):
    with tf.Graph().as_default():
        
        # Define initializer
        initializer = tf.random_uniform_initializer(-FLAGS.init_scale, FLAGS.init_scale)
            
        start_time = time.time()
        
        # Define the batchsizes and window_sizes (numsteps) for the different models during training
        if FLAGS.batchwise_learning:
            batch_sizes = [training_tuple[0].shape[0] for training_tuple in training_data_list]
            window_sizes = [training_tuple[0].shape[1] for training_tuple in training_data_list]
        else: # in case of windowwise learning
            window_sizes = sorted([b*10**e for b in [1,2,5] for e in range(10) if b*10**e >=5 and b*10**e < training_data_list[-1][0].shape[1]])
            #it would be good if batch_sizes were adaptive to trainingdata... 
            #if there are only x movies of largest length
            #batchsize shouldnt be bigger than x and other batches could be bigger.
            batch_sizes = [100, 50, 25, 10, 5, 2]
            batch_sizes = [int(b*FLAGS.scale) for b in batch_sizes]
            
            if FLAGS.model == "small":
                window_sizes = [5,10,20] #TODO debugging purpose
                batch_sizes = [1, 2, 2] #TODO debugging purpose
 
        print 'batch_sizes ',batch_sizes
        print 'window_sizes ',window_sizes
        
        # Build the LSTM Graph with 1 model for training, validation
        trainingmodels = []
        with tf.variable_scope("model", reuse=False, initializer=initializer) as model_scope:
            config.batch_size = batch_sizes[0]
            config.num_steps = window_sizes[0]
            mtrain = pilot_model.LSTMModel(True, config.output, config.feature_dimension, 
                                           config.batch_size, config.num_steps, 'train')
            trainingmodels.append(mtrain)
        # Reuse the defined weights but initialize with random_uniform_initializer
        with tf.variable_scope(model_scope, reuse=True, initializer=initializer): 
            for i in range(1,len(batch_sizes)):
                config.batch_size=batch_sizes[i]
                config.num_steps=window_sizes[i]
                mtrain = pilot_model.LSTMModel(True, config.output, config.feature_dimension, 
                                           config.batch_size, config.num_steps, 'train')
                trainingmodels.append(mtrain)       
            FLAGS.gpu = 'False'
            mvalid = pilot_model.LSTMModel(False, config.output, config.feature_dimension, prefix='eval')
        
        print "Loading models finished... ", print_time(start_time)
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
        # Initialize all variables (weights and biases)
        init = tf.initialize_all_variables()
        
        # Define session and initialize
        session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        session.run(init)
        #import pdb; pdb.set_trace()
        
        current_epoch = 0
        #Train while model gets better with upperbound max_max_epoch as number of runs
        for i in range(FLAGS.max_max_epoch):
            #every 1/5/10/50/100/500/1000 epochs it should write the current data away
            writer = None
            validate = False #only validate at some points during traing but not every epoch
            # write different epochs away as different logs in a logarithmic range till max_max_epoch
            if i in (b*10**exp for exp in range(1+int(math.log(FLAGS.max_max_epoch,10))) for b in [1,5]):
                if i!= 1:
                    #delete logdata of previous run except for the first run
                    location = logfolder+"/e"+str(current_epoch)
                    shutil.rmtree(location,ignore_errors=True) #it might delete the wrong folder...		
                current_epoch = i
                validate=True
                #print "save epoch: ", i+1
                location = logfolder+"/e"+str(i)
                writer = tf.train.SummaryWriter(location, graph=session.graph)
            
            #Train
            epoch_time=time.time()
            indices = None
            initial_states = None
            if not FLAGS.batchwise_learning: # In case of windowwise training
                # make a list of tuples containing batches with randomly picked windows of data
                # the indices are starting indices of the windows in certain data movies 
                # used for finding the initial state of the models
                indices = pilot_data.pick_random_windows(training_data_list, window_sizes, batch_sizes)
                
                # set the initial innerstate of the models according to the windows
                initial_states = pilot_eval.get_initial_states(trainingmodels, indices, mvalid, training_data_list, session)
                #res_state = session.run(trainingmodels[0].initial_state)
                #print 'resulting initial state: ',res_state
                #import pdb; pdb.set_trace()
                
            run_epoch(True, session, trainingmodels, training_data_list, config, logfolder, i, writer, indices, window_sizes, initial_states)
            print "Training finished... ", print_time(epoch_time)
            
            #only validate in few cases as it takes a lot of time setting the network on the GPU and evaluating all the movies
            if validate or FLAGS.validate_always:
                #Validate
                epoch_time=time.time()
                run_epoch(False, session, [mvalid], validate_data_list, config, logfolder, i, writer)
                print "Validation finished... ",print_time(epoch_time)           
            
        # Free some memory
        training_data_list = None
        validate_data_list = None
        
        # Save the variables to disk.
        save_path = saver.save(session, logfolder+"/model.ckpt")
        print("Model saved in file: %s" % save_path)
        
        # Free all memory after training
        session.close()
        #tf.reset_default_graph()
    # Evaluate the model
    FLAGS.model_dir = logfolder
    pilot_eval.evaluate(logfolder=logfolder, config=config, scope="model")
        
    # Give some nice feedback on how long it all took
    print "DONE... ", print_time(start_time)
    
if __name__ == '__main__':
    tf.app.run() 
    
    
