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

logging = tf.logging

FLAGS = tf.app.flags.FLAGS

##If you change default settings please modify the interpret_flags function
##in order to set your logfolder to a proper name...
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


#def run_epoch(session, model, data, targets, eval_op, merge_op=tf.no_op(), writer=None, verbose=False):
def run_epoch_fully_unroll(session, model, data, targets, eval_op, num_steps=1, writer=None, verbose=False, number_of_samples=500, matfile = ""):
    """
    Args:
        targets: [batch, steps, 4]
    """
    # run one time through the data given in a batch of all equal length.
    # network is fully unrolled. Used during training
    print "Fully unroll... over ", num_steps, " steps."
    start_time = time.time()
    costs = 0.0
    iters = 0.0
    score = 0.0
    state = model.initial_state.eval()
    feed_dict = {model.inputs: data[:,:,:], 
                    model.initial_state: state, 
                    model.targets: targets[:,:,:]}
    outputs, states, costs, _ = session.run([model.logits, model.states, model.cost, eval_op],feed_dict)
    trgts = targets.reshape((outputs.shape[0],outputs.shape[1]))
    score = float(sum((np.argmax(trgts, axis=1)==np.argmax(outputs, axis=1))))
    total = data.shape[0]*num_steps
    
    # Keep track of cell states
    if(writer != None) and (matfile != ""):
        #states = [batchsize, num_steps*hidden_size*num_layers*2 (~output;state)]
        d = {'states': states, 'targets':trgts}
        sio.savemat(matfile,d, appendmat=True)
    btime = time.time()
    if(writer != None):
        with tf.device('/cpu:0'):
            state_images = pilot_states.get_image(states, targets, num_steps, model.hidden_size, model.num_layers, model.batch_size)
            im_op = tf.image_summary('Innerstates', state_images, max_images=9)
            summary_str = session.run(tf.merge_summary([im_op]))
            writer.add_summary(summary_str)
            print "state image: ", time.time() - btime, " sec"
    #write the steps away: first movie in case of batch: output of different movies in batch is concatenated in time direction 
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
    training_data_list = pilot_data.prepare_data_grouped(config.training_objects)
    validate_data_list = pilot_data.prepare_data_list(config.validate_objects)
    
    #set params according to the shape of the obtained data
    config.output=training_data_list[0][1].shape[2]
    config.feature_dimension=training_data_list[0][0].shape[2]
    
    # Tell TensorFlow that the model will be built into the default Graph.
    #with tf.Graph().as_default(), tf.Session() as session:
    #with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(log_device_placement=True, log_device_placement=True)) as session, tf.device('/cpu:0'):
    with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
        
        # Define initializer
        initializer = tf.random_uniform_initializer(-FLAGS.init_scale, FLAGS.init_scale)
            
        start_time = time.time()
        
        # Build the LSTM Graph with 1 model for training, validation and testing
        trainingmodels = []
        with tf.variable_scope("model", reuse=False, initializer=initializer) as model_scope:
            config.batch_size = training_data_list[0][0].shape[0]#take all movies of first group in one batch
            config.num_steps = training_data_list[0][0].shape[1]
            mtrain = pilot_model.LSTMModel(True, config.output, config.feature_dimension, 
                                           config.batch_size, config.num_steps, 'train')
            trainingmodels.append(mtrain)
        # Reuse the defined weights but initialize with random_uniform_initializer
        with tf.variable_scope(model_scope, reuse=True): 
            for training_tuple in training_data_list[1:]:
                config.batch_size = training_tuple[0].shape[0]
                config.num_steps = training_tuple[0].shape[1]
                mtrain = pilot_model.LSTMModel(True, config.output, config.feature_dimension, 
                                           config.batch_size, config.num_steps, 'train')
                trainingmodels.append(mtrain)       
            mvalid = pilot_model.LSTMModel(False, config.output, config.feature_dimension, prefix='val')
        
        # create some tensor op for calculating and keeping the accuracy over different epochs
        accuracy = tf.placeholder(tf.float32, name="accuracy")
        accuracy_max = tf.placeholder(tf.float32, name="accuracy_max")
        accuracy_min = tf.placeholder(tf.float32, name="accuracy_min")
        perplexity = tf.placeholder(tf.float32, name="perplexity")
        with tf.name_scope("training_overview"):
            acc_sum = tf.scalar_summary("training_accuracy_av", accuracy)
            acc_max_sum = tf.scalar_summary("training_accuracy_max", accuracy_max)
            acc_min_sum = tf.scalar_summary("training_accuracy_min", accuracy_min)
            per_sum = tf.scalar_summary("training_perplexity", perplexity)
            merge_t = tf.merge_summary([acc_sum, acc_max_sum, acc_min_sum, per_sum])
        with tf.name_scope("validation_overview"):
            acc_sum = tf.scalar_summary("validation_accuracy", accuracy)
            acc_max_sum = tf.scalar_summary("validation_accuracy_max", accuracy_max)
            acc_min_sum = tf.scalar_summary("validation_accuracy_min", accuracy_min)
            per_sum = tf.scalar_summary("validation_perplexity", perplexity)
            merge_v = tf.merge_summary([acc_sum, acc_max_sum, acc_min_sum, per_sum])
        location = logfolder+"/overview"
        writer_overview= tf.train.SummaryWriter(location, graph=session.graph)    
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
        # Initialize all variables (weights and biases)
        init = tf.initialize_all_variables()
        session.run(init)
        
        current_epoch = 0
        #Train while model gets better with upperbound max_max_epoch as number of runs
        for i in range(FLAGS.max_max_epoch):
            #every 1/5/10/50/100/500/1000 epochs it should write the current data away
            writer = None
            # write different epochs away as different logs in a logarithmic range till max_max_epoch
            if i in (b*10**exp for exp in range(1+int(math.log(FLAGS.max_max_epoch,10))) for b in [1,5]):
                if i!= 1:
                    #delete logdata of previous run except for the first run
                    location = logfolder+"/e"+str(current_epoch)
                    shutil.rmtree(location,ignore_errors=True) #it might delete the wrong folder...		
                current_epoch = i
                #print "save epoch: ", i+1
                location = logfolder+"/e"+str(i)
                writer = tf.train.SummaryWriter(location, graph=session.graph)
            
            ### learning rate goes down during learning
            lr_decay = FLAGS.lr_decay ** max(i - FLAGS.max_epoch, 0.0)
            for mtrain in trainingmodels: mtrain.assign_lr(session, FLAGS.learning_rate * lr_decay)
            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(mtrain.lr)))
            
            ### Train the model
            acc = 0.0
            acc_max = 0.0
            acc_min = 1.0
            per = 0.0
            winner_index = -1
            lozer_index = -1
            chosen_model = int(random.uniform(0,len(trainingmodels))) #choose 1 model from which the first movie is recorded
            indices = range(len(trainingmodels))
            if FLAGS.random_order: random.shuffle(indices)
            for j in indices:
                if j == chosen_model : train_writer = writer
                else : train_writer = None
                results = run_epoch_fully_unroll(session, trainingmodels[j], training_data_list[j][0], training_data_list[j][1], trainingmodels[j].train_op, num_steps = training_data_list[j][0].shape[1], writer = train_writer, verbose = False, matfile=logfolder+"/trainstates_"+str(i)+".mat")
                train_perplexity = np.exp(results[1] / results[2])
                train_accuracy = results[0]/results[2]+0.000
                #print("Train Accuracy: %.3f" % ( train_accuracy))
                if acc_max < train_accuracy: 
                    acc_max = train_accuracy
                    winner_index = j
                if acc_min > train_accuracy: 
                    acc_min = train_accuracy
                    lozer_index = j
                acc = acc+train_accuracy/len(trainingmodels)
                per = per+train_perplexity/len(trainingmodels)
            # write away results of last movie
            if(writer_overview != None): 
                summary_str= session.run(merge_t,{accuracy : acc, perplexity: per, accuracy_max: acc_max, accuracy_min: acc_min})
                writer_overview.add_summary(summary_str, i)
                print("Epoch: %d Final Train Accuracy: %.3f, Max: %.3f at index %d, Min: %.3f at index %d." % (i + 1, acc, acc_max, winner_index, acc_min, lozer_index))
                
            ### Validate the model
            acc = 0.0
            acc_max = 0.0
            acc_min = 1.0
            per = 0.0
            chosen_model = int(random.uniform(0,len(validate_data_list)))
            indices = range(len(validate_data_list))
            if FLAGS.random_order : random.shuffle(indices)
            for j in indices:
                # write away prediction results of last movie
                if j == chosen_model : val_writer = writer
                else : val_writer = None
                #import pdb; pdb.set_trace()
                results = pilot_eval.run_epoch_one_step(session, mvalid, validate_data_list[j][0], validate_data_list[j][1], tf.no_op(), writer = val_writer, verbose = False, matfile=logfolder+"/valstates_"+str(i)+".mat")
                valid_perplexity = np.exp(results[1] / results[2])
                valid_accuracy = results[0]/results[2]
                print("Valid Accuracy: %.3f" % (valid_accuracy))
                if acc_max < valid_accuracy: acc_max = valid_accuracy
                if acc_min > valid_accuracy: acc_min = valid_accuracy
                acc = acc+valid_accuracy/len(validate_data_list)
                per = per+valid_perplexity/len(validate_data_list)
            if(writer_overview != None): 
                summary_str= session.run(merge_v,{accuracy : acc, perplexity: per, accuracy_max: acc_max, accuracy_min: acc_min})
                writer_overview.add_summary(summary_str, i)
                print("Epoch: %d Final Valid Accuracy: %.3f, Max: %.3f, Min: %.3f." % (i + 1, acc, acc_max, acc_min))
        
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
    pilot_eval.evaluate(logfolder=logfolder, config=config, scope="model")
        
    # Give some nice feedback on how long it all took
    duration = (time.time()-start_time)
    m, s = divmod(duration, 60)
    h, m = divmod(m, 60)
    print "DONE... time: %d:%02d:%02d" % (h, m, s)
    
if __name__ == '__main__':
    tf.app.run() 
    
    
