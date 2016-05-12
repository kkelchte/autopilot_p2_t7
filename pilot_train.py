#from __future__ import division
import pilot_data
import pilot as pilot
import random
import time

import tensorflow as tf
import numpy as np

import math
import shutil

flags = tf.flags
logging = tf.logging

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, big and dumpster.")
flags.DEFINE_string(
    "log_directory", "/esat/qayd/kkelchte/tensorflow/lstm_logs/",
    "this is the directory for log files that are visualized by tensorboard")
flags.DEFINE_float(
    "learning_rate", "0",
    "Define the learning rate for training.")
flags.DEFINE_integer(
    "num_layers", "0",
    "Define the num_layers for training.")
flags.DEFINE_integer(
    "hidden_size", "0",
    "Define the hidden_size for training.")
flags.DEFINE_float(
    "keep_prob", "0",
    "Define the hidden_size for training.")
flags.DEFINE_string(
    "optimizer", "",
    "Define the wanted optimizer for training.")



# the class with general empty class variables that are set by the code
# and general parameters of which i'm not playing around with yet.
class Configuration:
    gpu= True
    batch_size=1
    prefix = ""
    feature_dimension = -1
    output_size = -1
    number_of_samples = 500 #number of times the output is memorized over the full test and validation set (13 to 15 movies), now only used for test configuration to speed things up
    sample = 16 #the training data will be downsampled in time with i%sample==0
    init_scale = 0.1
    learning_rate = 0.01
#    max_grad_norm = 5 #According to search space odyssey hurts clipping performance
    num_layers = 2
    num_steps = -1
    hidden_size = 50 #dimensionality of cell state and output
    max_epoch = 50 #100
    max_max_epoch = 100
    keep_prob = 1.0 #1.0 #dropout
    lr_decay = 1 / 1.15
    optimizer = 'Adam'
    dataset = 'data'
    feature_type = 'both' #flow or app or both
    training_objects = ['dumpster', 'box']
    validate_objects = ['box']
    test_objects = ['wooden_case']
    

class SmallConfig(Configuration):
    """small config, for testing."""
    sample = 20
    num_layers = 1
    hidden_size = 1 #dimensionality of cell state and output
    max_epoch = 2
    max_max_epoch = 10
    dataset = 'generated'
    training_objects = ['modeldaa','modelcba','modelfee']
    validate_objects = ['modeldaa']
    test_objects = ['modelaaa']
    feature_type = 'app' #flow or app or both
    
class GPUConfig(Configuration):
    num_steps = -1
    sample = 1
    hidden_size = 1
    max_epoch = 3
    max_max_epoch = 6
    feature_type='app'
    dataset = 'generated'
    training_objects = ['modeldde']#, 'modelffc']
    validate_objects = ['modeldde']
    test_objects = ['modeldde']
    
class MediumConfig(Configuration):
    """small config, for testing."""
    sample = 16
    num_layers = 2
    hidden_size = 10 #dimensionality of cell state and output
    max_epoch = 50
    max_max_epoch = 100
    dataset = 'generated'
    training_objects = ['modeldaa','modelbae','modelacc','modelbca','modelafa', 'modelaaa']
    validate_objects = ['modeldea','modelabe']
    test_objects = ['modelaaa', 'modelccc']
    feature_type = 'app' #flow or app or both
        
class BigConfig(Configuration):
    """Big config, for real training.
        List of objects is obtained from pilot_data
    """
    dataset='generated'
    training_objects, validate_objects, test_objects = pilot_data.get_objects(dataset)
    
#def run_epoch(session, model, data, targets, eval_op, merge_op=tf.no_op(), writer=None, verbose=False):
def run_epoch(session, model, data, targets, eval_op, num_steps=1, writer=None, verbose=False, number_of_samples=100):
    # In one epoch, the model trains/validates/tests in different minibatch runs
    # The size of the minibatch is defined by the num_steps the LSTM is unrolled in time
    print "num_steps: ",num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0.0
    score = 0.0
    state = model.initial_state.eval()
    #import pdb; pdb.set_trace()
        
    # In case you go through the data in one track
    if num_steps == data.shape[1]:
        feed_dict = {model.inputs: data[:,:,:], 
                     model.initial_state: state, 
                     model.targets: targets[:,:,:]}
        outputs, state, costs, _ = session.run([model.logits, model.state, model.cost, eval_op],feed_dict)
        trgts = targets.reshape(outputs.shape[0],outputs.shape[1])
        score = float(sum((np.argmax(trgts, axis=1)==np.argmax(outputs, axis=1))))
        total = data.shape[0]*num_steps
        #import pdb; pdb.set_trace()
        #write the first steps away
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
    
    #loop over frames in steps of size 1
    for f in range(data.shape[1]-num_steps+1):
        feed_dict = {model.inputs: data[:,f:f+num_steps,:], 
                     model.initial_state: state, 
                     model.targets: targets[:,f:f+num_steps,:]}
        write_freq = int((data.shape[1]-num_steps+1)/number_of_samples)
        #if frames - steps > 100 ==> dont do modulo of zero
        #import pdb; pdb.set_trace()
        if ( write_freq != 0) and not (writer == None):
            if (f % write_freq) == 0:
                outputs, state, current_loss, _, summary_str= session.run([model.logits, model.state, model.cost, eval_op, model.merge()],feed_dict)
                writer.add_summary(summary_str, f)
                #print '----------------------------im writing away every write frequency'
        #in case frames - steps < 100 ==> save every frame because there are less than 100 steps
        #the summary writer will only record the last <100 steps
        elif (write_freq == 0) and not (writer == None):
            outputs, state, current_loss, _, summary_str= session.run([model.logits, model.state, model.cost, eval_op, model.merge()],feed_dict)
            writer.add_summary(summary_str, f)
            #print '--------------------------im writing away'
        else:
            # outputs is of shape (num_steps*batchsize, outputsize)
            outputs, state, current_loss, _= session.run([model.logits, model.state, model.cost, eval_op],feed_dict)
        iters += data.shape[0]
        
	#maybe add different way of calculating score and prediction when working with num_steps = -1 ~ video length
        for b in range(data.shape[0]):
            if np.argmax(outputs[b*num_steps]) == np.argmax(targets[b,f,:]):
                score += 1.0
            if verbose and (f % int(data.shape[1]/15) == 0):
                print("Frame: %d batch: %d target: %d prediction: %d speed: %.3f fps"\
                    %(f, b, np.argmax(targets[b,f,:]), np.argmax(outputs[b*num_steps]),
                    iters * data.shape[0] / (time.time() - start_time)))    
        costs += current_loss
    
    if verbose:
        print "Accuracy: ",(score/iters)
    #import pdb; pdb.set_trace()
     
           
    ##Add call for tensor accuracy + write it away
    return score, costs, iters

def get_config():
    if FLAGS.model == "big":
        return BigConfig()
    if FLAGS.model == "small":
        return SmallConfig()
    if FLAGS.model == "GPU":
        return GPUConfig()
    if FLAGS.model == "medium":
        return MediumConfig()
    else:
        return Configuration()

#def interpret_flags(FLAGS, config):
    
    

def main(_):
    start_time = time.time()
    
    # Get chosen configuration
    config = get_config()
    
    # Specify the configurations for training/validation/testing
    valid_config = get_config()
    config.prefix = "training"
    valid_config.prefix = "validation"
    test_config = get_config()
    test_config.prefix = "test"
    
    # Check if folder for logging is already used.
    logfolder = FLAGS.log_directory+FLAGS.model
    #import pdb; pdb.set_trace()
    # Read the flags from the input
    if FLAGS.learning_rate != 0: 
        config.learning_rate = FLAGS.learning_rate
        lr = 0
        if FLAGS.learning_rate < 1:
            lr = str(FLAGS.learning_rate)[2:]
            lr = "0" + lr
        else:
            lr = str(FLAGS.learning_rate)
            lr = lr[:lr.find('.')]
        #import pdb; pdb.set_trace()
        logfolder = logfolder + "_lr_"+ lr
    if FLAGS.num_layers != 0:
        config.num_layers = FLAGS.num_layers
        valid_config.num_layers = FLAGS.num_layers
        test_config.num_layers = FLAGS.num_layers
        logfolder = logfolder + "_layers_"+ str(FLAGS.num_layers)
    if FLAGS.hidden_size != 0: 
        config.hidden_size = FLAGS.hidden_size
        valid_config.hidden_size = FLAGS.hidden_size
        test_config.hidden_size = FLAGS.hidden_size
        logfolder = logfolder + "_size_"+ str(FLAGS.hidden_size)
    if FLAGS.keep_prob != 0: 
        config.keep_prob = FLAGS.keep_prob
        kp = str(FLAGS.keep_prob)[2:]
        kp = "0" + kp
        logfolder = logfolder + "_drop_"+ str(kp)
    if FLAGS.optimizer != "": 
        config.optimizer = FLAGS.optimizer
        logfolder = logfolder + "_opt_"+ str(FLAGS.optimizer)
    
    #delete logfolder if its already used otherwise it gets huge
    shutil.rmtree(logfolder,ignore_errors=True)
    print 'make logfolder: ', logfolder
    # Trainingdata is a list of tuples (data, labels), each tuple corresponds to one training group
    # data is an array of shape [batch_of_movies, movie_length, feature_dimension]
    # labels is an array of shape [batch_of_movies, movie_length, output_dimension]
    training_data_list = pilot_data.prepare_data_grouped(config.training_objects, config.sample, config.feature_type, config.dataset)
    validate_data_list = pilot_data.prepare_data_list(config.validate_objects, config.sample, config.feature_type, config.dataset)
    test_data_list = pilot_data.prepare_data_list(config.test_objects, config.sample, config.feature_type, config.dataset)
    
    #import pdb; pdb.set_trace()
    
    # Feature dimension from data
    if config.feature_dimension == -1:
        config.feature_dimension = training_data_list[0][0].shape[2]
        valid_config.feature_dimension = training_data_list[0][0].shape[2]
        test_config.feature_dimension = training_data_list[0][0].shape[2]
    if config.output_size == -1:
        config.output_size = training_data_list[0][1].shape[2]
        valid_config.output_size = training_data_list[0][1].shape[2]
        test_config.output_size = training_data_list[0][1].shape[2]
    # Num_steps:
    valid_config.num_steps = 1
    test_config.num_steps = 1
    # if number of steps is -1 it adapts to the length of 1 movie,
    # unrolling the network over the full length of the movie.
    if config.num_steps == -1: config.num_steps = training_data_list[0][0].shape[1]
    
    # batch_size:
    config.batch_size = training_data_list[0][0].shape[0]#take all movies in one batch
    valid_config.batch_size = 1
    test_config.batch_size = 1
    # adapt test data and labels according to batchsize 1 ~ make 1 movie by concatenating the different testmovies
    #test_data = np.reshape(test_data,(1, -1, test_data.shape[2]))      
    #test_labels = np.reshape(test_labels,(1, -1, test_labels.shape[2]))
    
    
    # Tell TensorFlow that the model will be built into the default Graph.
    #with tf.Graph().as_default(), tf.Session() as session:
    #with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(log_device_placement=True, log_device_placement=True)) as session, tf.device('/cpu:0'):
    with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
        
        # Define initializer
        #with tf.device('/gpu:0'):
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
        #initializer = tf.constant_initializer(1.0)
            
        start_time = time.time()
        trainingmodels = []
        # Build the LSTM Graph with 1 model for training, validation and testing
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            mtrain = pilot.LSTMModel(is_training = True, config=config)
            trainingmodels.append(mtrain)
        # Reuse the defined weights but initialize with random_uniform_initializer
        with tf.variable_scope("model", reuse=True): #, initializer=initializer): I dont think they should be randomly initialized if variables are reused anyway
            for training_tuple in training_data_list[1:]:
                print 'make a new model'
                config.batch_size = training_tuple[0].shape[0]
                config.num_steps = training_tuple[0].shape[1]
                mtrain = pilot.LSTMModel(is_training = True, config=config)
                trainingmodels.append(mtrain)       
            mvalid = pilot.LSTMModel(is_training=False, config=valid_config)
            mtest = pilot.LSTMModel(is_training=False, config=test_config)
        print "Time for loading models: ", (time.time()-start_time)  
        print "Number of models: ", (len(trainingmodels)+2)
        
        # create some tensor op for calculating and keeping the accuracy over different epochs
        accuracy = tf.placeholder(tf.float32, name="accuracy")
        perplexity = tf.placeholder(tf.float32, name="perplexity")
        with tf.name_scope("training_overview"):
            acc_sum = tf.scalar_summary("training_accuracy", accuracy)
            per_sum = tf.scalar_summary("training_perplexity", perplexity)
            merge_t = tf.merge_summary([acc_sum, per_sum])
        with tf.name_scope("validation_overview"):
            acc_sum = tf.scalar_summary("validation_accuracy", accuracy)
            per_sum = tf.scalar_summary("validation_perplexity", perplexity)
            merge_v = tf.merge_summary([acc_sum, per_sum])
        location = logfolder+"/overview"
        writer_overview= tf.train.SummaryWriter(location, session.graph_def)    
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
        # Initialize all variables (weights and biases)
        init = tf.initialize_all_variables()
        session.run(init)
        
        #import pdb; pdb.set_trace()
        current_epoch = 0
        #Train while model gets better with upperbound max_max_epoch as number of runs
        for i in range(config.max_max_epoch+1):
            #every 1/5/10/50/100/500/1000 epochs it should write the current data away
            writer = None
            # write different epochs away as different logs in a logarithmic range till max_max_epoch
            if i in (b*10**exp for exp in range(1+int(math.log(config.max_max_epoch,10))) for b in [1,5]):
                if i!= 1:
                    #delete logdata of previous run except for the first run
                    location = logfolder+"/e"+str(current_epoch)
                    shutil.rmtree(location,ignore_errors=True) #it might delete the wrong folder...		
                current_epoch = i
                #print "save epoch: ", i+1
                location = logfolder+"/e"+str(i)
                writer = tf.train.SummaryWriter(location, session.graph_def)
        
            ### learning rate goes down during learning
            lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
            mtrain.assign_lr(session, config.learning_rate * lr_decay)
            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(mtrain.lr)))
            
            ### Train the model
            minacc = 1.0
            maxper = 0.0
            chosen_model = int(random.uniform(0,len(trainingmodels))) #choose 1 model from which the first movie is recorded
            indices = range(len(trainingmodels))
            random.shuffle(indices)
            for j in indices:
                if j == chosen_model : train_writer = writer
                else : train_writer = None
                results = run_epoch(session, trainingmodels[j], training_data_list[j][0], training_data_list[j][1], trainingmodels[j].train_op, num_steps = training_data_list[j][0].shape[1], writer = train_writer, verbose = False)
                train_perplexity = np.exp(results[1] / results[2])
                train_accuracy = results[0]/results[2]+0.000
                print("Epoch: %d Train Accuracy: %.3f" % (i + 1, train_accuracy))
                if minacc > train_accuracy: minacc = train_accuracy
                if maxper < train_perplexity: maxper = train_perplexity
            # write away results of last movie
            if(writer_overview != None): 
                summary_str= session.run(merge_t,{accuracy : minacc, perplexity: maxper})
                writer_overview.add_summary(summary_str, i)
                print("Epoch: %d Final Accuracy: %.3f" % (i + 1, minacc))
            
            ### Validate the model
            minacc = 100.0
            maxper = 0.0
            chosen_model = int(random.uniform(0,len(validate_data_list)))
            indices = range(len(validate_data_list))
            random.shuffle(indices)
            for j in indices:
                # write away prediction results of last movie
                if j == chosen_model : val_writer = writer
                else : val_writer = None
                #import pdb; pdb.set_trace()
                results = run_epoch(session, mvalid, validate_data_list[j][0], validate_data_list[j][1], tf.no_op(), writer = val_writer, number_of_samples = valid_config.number_of_samples, verbose = False)
                valid_perplexity = np.exp(results[1] / results[2])
                valid_accuracy = results[0]/results[2]
                print("Epoch: %d Valid Accuracy: %.3f" % (i + 1, valid_accuracy))
                if minacc > valid_accuracy: minacc = valid_accuracy
                if maxper < valid_perplexity: maxper = valid_perplexity
            if(writer_overview != None): 
                #summary_str= session.run(merge_v,{score : results[0], cost: results[1], total: results[2]})
                summary_str= session.run(merge_v,{accuracy : minacc, perplexity: maxper})
                writer_overview.add_summary(summary_str, i)
                print("Epoch: %d Final Accuracy: %.3f" % (i + 1, minacc))
        
        # Test the model
        location = logfolder+"/test"
        writer = tf.train.SummaryWriter(location, session.graph_def)
        for j in range(len(test_data_list)):
            # write away results of last movie
            if j == len(test_data_list)-1 : test_writer = writer
            else : test_writer = None
            results = run_epoch(session, mtest, test_data_list[j][0], test_data_list[j][1], tf.no_op(), writer = test_writer, verbose=True, number_of_samples = test_config.number_of_samples)
            test_perplexity = np.exp(results[1] / results[2])
            print("Test Perplexity: %.3f" % test_perplexity)
        
        # Save the variables to disk.
        save_path = saver.save(session, logfolder+"/model.ckpt")
        print("Model saved in file: %s" % save_path)
        
    print "DONE: ", (time.time()-start_time)/60, " minutes"
        #import pdb; pdb.set_trace()
            

if __name__ == '__main__':
    tf.app.run() 
