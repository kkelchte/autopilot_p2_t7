#from __future__ import division
import pilot_data
import pilot
import pilot_settings
import pilot_states

import random
import time
import scipy.io as sio

import tensorflow as tf
import numpy as np

import math
import shutil

logging = tf.logging
FLAGS = tf.app.flags.FLAGS
    
#def run_epoch(session, model, data, targets, eval_op, merge_op=tf.no_op(), writer=None, verbose=False):
def run_epoch_fully_unroll(session, model, data, targets, eval_op, num_steps=1, writer=None, verbose=False, number_of_samples=100, matfile = ""):
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
    #import pdb; pdb.set_trace()
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
    
    
def run_epoch_one_step(session, model, data, targets, eval_op, num_steps=1, writer=None, verbose=False, number_of_samples=100, matfile = ""):
    # Validate / test one step at the time going through the data
    # data is only 1 movie. Not a batch of different movies.
    # this function can also be used when you want to unroll over a few time steps but not all
    print "One step at the time... "
    start_time = time.time()
    costs = 0.0
    iters = 0.0
    score = 0.0
    state = model.initial_state.eval()
    
    #In case you loop over frames in steps of size 1
    list_of_states = []
    for f in range(data.shape[1]-num_steps+1):
        feed_dict = {model.inputs: data[:,f:f+num_steps,:], 
                     model.initial_state: state, 
                     model.targets: targets[:,f:f+num_steps,:]}
        write_freq = int((data.shape[1]-num_steps+1)/number_of_samples)
        #if frames - steps > 100 ==> dont do modulo of zero
        #import pdb; pdb.set_trace()
        if ( write_freq != 0) and not (writer == None):
            if (f % write_freq) == 0:
                outputs, state, states, current_loss, _, summary_str= session.run([model.logits, model.state, model.states, model.cost, eval_op, model.merge()],feed_dict)
                writer.add_summary(summary_str, f)
                #print '----------------------------im writing away every write frequency'
        #in case frames - steps < 100 ==> save every frame because there are less than 100 steps
        #the summary writer will only record the last <100 steps
        elif (write_freq == 0) and not (writer == None):
            outputs, state, states, current_loss, _, summary_str= session.run([model.logits, model.state, model.states,model.cost, eval_op, model.merge()],feed_dict)
            writer.add_summary(summary_str, f)
            #print '--------------------------im writing away'
        else:
            # outputs is of shape (num_steps*batchsize, outputsize)
            outputs, state, states, current_loss, _= session.run([model.logits, model.state, model.states,model.cost, eval_op],feed_dict)
            
        iters += data.shape[0]
        list_of_states.append(state)
	
	#maybe add different way of calculating score and prediction when working with num_steps = -1 ~ video length
        for b in range(data.shape[0]):
            if np.argmax(outputs[b*num_steps]) == np.argmax(targets[b,f,:]):
                score += 1.0
            if verbose and (f % int(data.shape[1]/15) == 0):
                print("Frame: %d batch: %d target: %d prediction: %d speed: %.3f fps"\
                    %(f, b, np.argmax(targets[b,f,:]), np.argmax(outputs[b*num_steps]),
                    iters * data.shape[0] / (time.time() - start_time)))    
        costs += current_loss
    
    # Keep track of cell states
    if(writer != None) and (matfile != ""):
        #states = [batchsize, num_steps*hidden_size*num_layers*2 (output;state)]
        states = np.concatenate(list_of_states)
        #import pdb; pdb.set_trace()
        trgts = targets.reshape((-1,outputs.shape[1]))
        d = {'states': states, 'targets': targets}
        sio.savemat(matfile,d, appendmat=True)
        
    if verbose:
        print "Accuracy: ",(score/iters)
    #import pdb; pdb.set_trace()
     
           
    ##Add call for tensor accuracy + write it away
    return score, costs, iters


    
    

def main(_):
    start_time = time.time()
    
    # Get configuration adapted to the flags of user input
    train_config, valid_config, test_config = pilot_settings.get_config()
    
    # Check if folder for logging is already used.
    logfolder = train_config.logfolder
    
    # Delete logfolder if its already used, otherwise it gets huge
    shutil.rmtree(logfolder,ignore_errors=True)
    print 'Logfolder: ', logfolder
    
    # List of sorted(already done in txt file) tuples (data, labels), each tuple corresponds to one training movie
    # data is an array of shape [batch_of_movies, movie_length, feature_dimension]
    # labels is an array of shape [batch_of_movies, movie_length, output_dimension]
    #training_data_list = pilot_data.prepare_data_grouped(train_config.training_objects, train_config)
    training_data_list = pilot_data.prepare_data_list(train_config.training_objects, train_config)
    validate_data_list = pilot_data.prepare_data_list(train_config.validate_objects, train_config)
    test_data_list = pilot_data.prepare_data_list(train_config.test_objects, train_config)
    
    # Feature dimension from data
    if train_config.feature_dimension == -1:
        train_config.feature_dimension = training_data_list[0][0].shape[2]
        valid_config.feature_dimension = training_data_list[0][0].shape[2]
        test_config.feature_dimension = training_data_list[0][0].shape[2]
    if train_config.output_size == -1:
        train_config.output_size = training_data_list[0][1].shape[2]
        valid_config.output_size = training_data_list[0][1].shape[2]
        test_config.output_size = training_data_list[0][1].shape[2]
    
    # if number of steps is -1 it adapts to the length of 1 movie,
    # unrolling the network over the full length of the movie.
    if train_config.num_steps == -1: train_config.num_steps = training_data_list[0][0].shape[1]
    
    # batch_size:
    train_config.batch_size = training_data_list[0][0].shape[0]#take all movies of first group in one batch
    
    
    # Tell TensorFlow that the model will be built into the default Graph.
    #with tf.Graph().as_default(), tf.Session() as session:
    #with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(log_device_placement=True, log_device_placement=True)) as session, tf.device('/cpu:0'):
    with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
        
        # Define initializer
        #with tf.device('/gpu:0'):
        initializer = tf.random_uniform_initializer(-train_config.init_scale, train_config.init_scale)
        #initializer = tf.constant_initializer(1.0)
            
        start_time = time.time()
        trainingmodels = []
        # Get a list of all batchsizes depending on the lengths of the movies:
        #lengths = [ t[0].shape[1] for t in training_data_list ]
        #b_sizes = np.
        #print lengths
        # Build the LSTM Graph with 1 model for training, validation and testing
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            mtrain = pilot.LSTMModel(is_training = True, config=train_config)
            trainingmodels.append(mtrain)
        # Reuse the defined weights but initialize with random_uniform_initializer
        with tf.variable_scope("model", reuse=True): #, initializer=initializer): I dont think they should be randomly initialized if variables are reused anyway
            for training_tuple in training_data_list[1:]:
                print 'make a new model for other training batch'
                train_config.batch_size = training_tuple[0].shape[0]
                train_config.num_steps = training_tuple[0].shape[1]
                mtrain = pilot.LSTMModel(is_training = True, config=train_config)
                trainingmodels.append(mtrain)       
            mvalid = pilot.LSTMModel(is_training=False, config=valid_config)
            mtest = pilot.LSTMModel(is_training=False, config=test_config)
            print "Time for loading models: ", (time.time()-start_time)  
            print "Number of models: ", (len(trainingmodels)+2)
        
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
            acc_max_sum = tf.scalar_summary("training_accuracy_max", accuracy_max)
            acc_min_sum = tf.scalar_summary("training_accuracy_min", accuracy_min)
            per_sum = tf.scalar_summary("validation_perplexity", perplexity)
            merge_v = tf.merge_summary([acc_sum, acc_max_sum, acc_min_sum, per_sum])
        location = logfolder+"/overview"
        writer_overview= tf.train.SummaryWriter(location, graph=session.graph)    
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
        # Initialize all variables (weights and biases)
        init = tf.initialize_all_variables()
        session.run(init)
        
        #import pdb; pdb.set_trace()
        current_epoch = 0
        #Train while model gets better with upperbound max_max_epoch as number of runs
        for i in range(train_config.max_max_epoch+1):
            #every 1/5/10/50/100/500/1000 epochs it should write the current data away
            writer = None
            # write different epochs away as different logs in a logarithmic range till max_max_epoch
            if i in (b*10**exp for exp in range(1+int(math.log(train_config.max_max_epoch,10))) for b in [1,5]):
                #if i!= 1:
                    ##delete logdata of previous run except for the first run
                    #location = logfolder+"/e"+str(current_epoch)
                    #shutil.rmtree(location,ignore_errors=True) #it might delete the wrong folder...		
                current_epoch = i
                #print "save epoch: ", i+1
                location = logfolder+"/e"+str(i)
                writer = tf.train.SummaryWriter(location, graph=session.graph)
        
            ### learning rate goes down during learning
            lr_decay = train_config.lr_decay ** max(i - train_config.max_epoch, 0.0)
            mtrain.assign_lr(session, train_config.learning_rate * lr_decay)
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
            if train_config.random_order: random.shuffle(indices)
            for j in indices:
                if j == chosen_model : train_writer = writer
                else : train_writer = None
                results = run_epoch_fully_unroll(session, trainingmodels[j], training_data_list[j][0], training_data_list[j][1], trainingmodels[j].train_op, num_steps = training_data_list[j][0].shape[1], writer = train_writer, verbose = False, matfile=logfolder+"/trainstates_"+str(i)+".mat")
                train_perplexity = np.exp(results[1] / results[2])
                train_accuracy = results[0]/results[2]+0.000
                print("Train Accuracy: %.3f" % ( train_accuracy))
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
                
            #### Validate the model
            acc = 0.0
            acc_max = 0.0
            acc_min = 1.0
            per = 0.0
            chosen_model = int(random.uniform(0,len(validate_data_list)))
            indices = range(len(validate_data_list))
            if train_config.random_order : random.shuffle(indices)
            for j in indices:
                # write away prediction results of last movie
                if j == chosen_model : val_writer = writer
                else : val_writer = None
                #import pdb; pdb.set_trace()
                results = run_epoch_one_step(session, mvalid, validate_data_list[j][0], validate_data_list[j][1], tf.no_op(), writer = val_writer, number_of_samples = valid_config.number_of_samples, verbose = False, matfile=logfolder+"/valstates_"+str(i)+".mat")
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
        
        # Test the model
        location = logfolder+"/test"
        writer = tf.train.SummaryWriter(location, graph=session.graph)
        for j in range(len(test_data_list)):
            # write away results of last movie
            if j == len(test_data_list)-1 : test_writer = writer
            else : test_writer = None
            results = run_epoch_one_step(session, mtest, test_data_list[j][0], test_data_list[j][1], tf.no_op(), writer = test_writer, verbose=True, number_of_samples = test_config.number_of_samples)
            test_perplexity = np.exp(results[1] / results[2])
            print("Test perplexity: %.3f" % test_perplexity)
        
        # Save the variables to disk.
        save_path = saver.save(session, logfolder+"/model.ckpt")
        print("Model saved in file: %s" % save_path)
    
    # Give some nice feedback on how long it all took
    duration = (time.time()-start_time)
    duration_message = ""
    if duration > 60:
        duration_message = duration_message+str(int(duration%60))+"sec."
        duration = int(duration/60) #number of minutes
        if duration > 60:
            duration_message = str(int(duration%60))+"min "+duration_message
            duration = int(duration/60) #number of hours
            if duration > 24:
                duration_message = str(int(duration%60))+"hours "+duration_message
                duration = int(duration/24) #number of days
                duration_message = str(duration)+"days "+duration_message
            else:
                duration_message = str(duration)+"hours "+duration_message
        else:
            duration_message = str(duration)+"min "+duration_message
    else:
        duration_message = str(duration)+"sec. "
    print "DONE time: "+duration_message
    
if __name__ == '__main__':
    tf.app.run() 
