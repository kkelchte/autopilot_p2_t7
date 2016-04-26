import pilot_data
import pilot as pilot

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
flags.DEFINE_float(
    "max_grad_norm", "0",
    "Define the max_grad_norm for training.")
flags.DEFINE_integer(
    "num_layers", "0",
    "Define the num_layers for training.")
flags.DEFINE_integer(
    "hidden_size", "0",
    "Define the hidden_size for training.")
flags.DEFINE_integer(
    "num_steps", "0",
    "Define the hidden_size for training.")


#flags.DEFINE_integer("batch_size", 2, "The number of movies processed in one full batch.")

# the class with general empty class variables that are set by the code
# and general parameters of which i'm not playing around with yet.
class Configuration:
    gpu=True
    batch_size=1.0
    prefix = ""
    feature_dimension = -1
    output_size = -1
    sample = 1 #the training data will be downsampled in time with i%sample==0
    init_scale = 1.0 #0.1
    learning_rate = 1.0
    max_grad_norm = 100 #5 According to search space odyssey hurts clipping performance
    num_layers = 2
    num_steps = -1
    hidden_size = 10 #dimensionality of cell state and output
    max_epoch = 100
    max_max_epoch = 1000
    keep_prob = 0.5 #1.0 
    lr_decay = 1 / 1.15
    dataset = 'data'
    feature_type = 'both' #flow or app or both
    training_objects = ['dumpster', 'box']
    validate_objects = ['box']
    test_objects = ['wooden_case']
    

class SmallConfig(Configuration):
    """small config, for testing."""
    num_steps = -1
    sample = 100
    num_layers = 1
    hidden_size = 1 #dimensionality of cell state and output
    max_epoch = 5
    max_max_epoch = 10
    dataset='generated'
    training_objects = ['modelaaa']
    validate_objects = ['modelaaa']
    test_objects = ['modelaaa']
    feature_type = 'app' #flow or app or both
    
class GPUConfig(Configuration):
    """train only flying around a dumpster and see if error gets really low"""
    num_steps = -1
    sample = 1
    hidden_size=10
    dataset = 'generated'
    feature_type='both'
    training_objects = ['modelffe']
    validate_objects = ['modelffe']
    test_objects = ['modelffc']
    
class MediumConfig(Configuration):
    """Medium config, for train small."""
    num_steps = -1
    hidden_size = 100 #dimensionality of cell state and output
    dataset='generated'
    training_objects = ['modelaaa']
    validate_objects = ['modelaaa']
    test_objects = ['modelaaa']


        
class BigConfig(Configuration):
    """Big config, for real training.
        List of objects is obtained from pilot_data
    """
    dataset='generated'
    training_objects, validate_objects, test_objects = pilot_data.get_objects(dataset)
    

#def run_epoch(session, model, data, targets, eval_op, merge_op=tf.no_op(), writer=None, verbose=False):
def run_epoch(session, model, data, targets, eval_op, num_steps=1, writer=None, verbose=True):
    # In one epoch, the model trains/validates/tests in different minibatch runs
    # The size of the minibatch is defined by the num_steps the LSTM is unrolled in time
    print "num_steps: ",num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0.0
    score = 0
    state = model.initial_state.eval()
    #import pdb; pdb.set_trace()
        
    # In case you go through the data in one track
    if num_steps >= data.shape[1]:
        feed_dict = {model.inputs: data[:,:,:], 
                     model.initial_state: state, 
                     model.targets: targets[:,:,:]}
        outputs, state, costs, _ = session.run([model.logits, model.state, model.cost, eval_op],feed_dict)
        trgts = targets.reshape(outputs.shape[0],outputs.shape[1])
        score = sum((np.argmax(trgts, axis=1)==np.argmax(outputs, axis=1)))
        total=data.shape[0]*num_steps
        print score, costs, num_steps
        return score, costs, data.shape[0]*num_steps
    
    for f in range(data.shape[1]-num_steps+1):#loop over frames
        feed_dict = {model.inputs: data[:,f:f+num_steps,:], 
                     model.initial_state: state, 
                     model.targets: targets[:,f:f+num_steps,:]}
        write_freq = int((data.shape[1]-num_steps+1)/100)
        
        #if frames - steps > 100 ==> dont do modulo of zero
        if ( write_freq != 0) and not (writer == None):
            if (f % write_freq) == 0:
                outputs, state, current_loss, _, summary_str= session.run([model.logits, model.state, model.cost, eval_op, model.merge()],feed_dict)
                writer.add_summary(summary_str, f)
        #in case frames - steps < 100 ==> save every frame because there are less than 100 steps
        #the summary writer will only record the last <100 steps
        elif (write_freq == 0) and not (writer == None):
            outputs, state, current_loss, _, summary_str= session.run([model.logits, model.state, model.cost, eval_op, model.merge()],feed_dict)
            writer.add_summary(summary_str, f)
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
        
    
        
    #import pdb; pdb.set_trace()
    #if verbose:
    print "Accuracy: ",(score/iters)
    print "FPS: ", data.shape[1]/(time.time()-start_time) 
    print "time: ",(time.time()-start_time) 
     
           
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

def main(_):
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
        logfolder = logfolder + "_lr_"+ str(FLAGS.learning_rate)
    if FLAGS.max_grad_norm != 0: 
        config.max_grad_norm = FLAGS.max_grad_norm
        logfolder = logfolder + "_mxgrad_"+ str(FLAGS.max_grad_norm)
    if FLAGS.num_layers != 0:
        print "i do get in this loop"
        config.num_layers = FLAGS.num_layers
        valid_config.num_layers = FLAGS.num_layers
        test_config.num_layers = FLAGS.num_layers
        logfolder = logfolder + "_layers_"+ str(FLAGS.num_layers)
    if FLAGS.hidden_size != 0: 
        config.hidden_size = FLAGS.hidden_size
        valid_config.hidden_size = FLAGS.hidden_size
        test_config.hidden_size = FLAGS.hidden_size
        logfolder = logfolder + "_size_"+ str(FLAGS.hidden_size)
    if FLAGS.num_steps != 0: 
        config.num_steps = FLAGS.num_steps
        logfolder = logfolder + "_stps_"+ str(FLAGS.num_steps)
    
    #delete logfolder if its already used otherwise it gets huge
    shutil.rmtree(logfolder,ignore_errors=True)
    
    # Obtain data in shape [movies, movie_length, feature_dimension]
    # Obtain targets in shape [movies, movie_length, output_dimension]
    training_data, training_labels = pilot_data.prepare_data(config.training_objects, config.sample, config.feature_type, config.dataset)
    #validate_data, validate_labels = pilot_data.prepare_data(config.validate_objects, config.sample, config.feature_type, config.dataset)
    #test_data, test_labels = pilot_data.prepare_data(config.test_objects, config.sample, config.feature_type, config.dataset)
    
    #import pdb; pdb.set_trace()
    
    # Feature dimension from data
    if config.feature_dimension == -1:
        config.feature_dimension = training_data.shape[2]
        valid_config.feature_dimension = training_data.shape[2]
        test_config.feature_dimension = training_data.shape[2]
    if config.output_size == -1:
        config.output_size = training_labels.shape[2]
        valid_config.output_size = training_labels.shape[2]
        test_config.output_size = training_labels.shape[2]
    # Num_steps:
    #valid_config.num_steps = 1
    #test_config.num_steps = 1
    # feature_dimension
    # if number of steps is -1 it adapts to the length of 1 movie,
    # unrolling the network over the full length of the movie.
    if config.num_steps == -1 or config.num_steps > training_data.shape[1]: 
        config.num_steps = training_data.shape[1]#config.batch_size = 1
    
    
    # batch_size:
    config.batch_size = training_data.shape[0]#take all movies in one batch
    # The validation set performs in batches according to the length of the movie
    #valid_config.batch_size = validate_data.shape[0]
    # The test set predicts every step reshapes the data so it is all in 1 long movie
    #test_config.batch_size = 1
    # adapt test data and labels according to batchsize 1 ~ make 1 movie by concatenating the different testmovies
    #test_data = np.reshape(test_data,(1, -1, test_data.shape[2]))      
    #test_labels = np.reshape(test_labels,(1, -1, test_labels.shape[2]))
    
    
    # Tell TensorFlow that the model will be built into the default Graph.
    #with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session, tf.device('/cpu:0'):
    with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(log_device_placement=True,  allow_soft_placement=True)) as session:
        # Define initializer
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
        #initializer = tf.constant_initializer(1.0)
        start_time = time.time()
        # Build the LSTM Graph with 1 model for training, validation and testing  , tf.device('/gpu:0')
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            mtrain = pilot.LSTMModel(is_training = True, config=config)
            
        print "Loading model: ", (time.time()-start_time)
        # Reuse the defined weights but initialize with random_uniform_initializer
        #with tf.variable_scope("model", reuse=True, initializer=initializer):
        #    mvalid = pilot.LSTMModel(is_training=False, config=valid_config)
        #    mtest = pilot.LSTMModel(is_training=False, config=test_config)
            
        # create some tensor + op for calculating + keeping the accuracy over different epochs
        score = tf.placeholder(tf.float32, name="score")
        cost = tf.placeholder(tf.float32, name="cost")
        total = tf.placeholder(tf.float32, name="total")
        with tf.name_scope("training_overview"):
            accuracy = tf.div(score,total,name="accuracy")
            acc_sum = tf.scalar_summary(accuracy.op.name, accuracy)
            perplexity = tf.exp(tf.div(cost,total),name="perplexity")
            per_sum = tf.scalar_summary(perplexity.op.name, perplexity)
            merge_t = tf.merge_summary([acc_sum, per_sum])
        #with tf.name_scope("validation_overview"):
        #    accuracy = tf.div(score,total,name="accuracy")
        #    acc_sum = tf.scalar_summary(accuracy.op.name, accuracy)
        #    perplexity = tf.exp(tf.div(cost,total),name="perplexity")
        #    per_sum = tf.scalar_summary(perplexity.op.name, perplexity)
        #    merge_v = tf.merge_summary([acc_sum, per_sum])
            
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
        # Initialize all variables (weights and biases)
        init = tf.initialize_all_variables()
        session.run(init)
        
        location = logfolder+"/overview"
        writer_overview= tf.train.SummaryWriter(location, session.graph_def)    
        current_epoch = 0
        #Train while model gets better with upperbound max_max_epoch as number of runs
        for i in range(config.max_max_epoch+1):
            #every 10/100/1000 epochs it should write the current data away
            writer = None
            # write different epochs away as different logs in a logarithmic range till max_max_epoch
            if i in (b*10**exp for exp in range(1+int(math.log(config.max_max_epoch,10))) for b in [1,5]):
		if i!= 1:
			#delete logdata of previous run except for the first run
			location = logfolder+"/e"+str(current_epoch)
			shutil.rmtree(location,ignore_errors=True)		
		current_epoch = i
                #print "save epoch: ", i+1
                location = logfolder+"/e"+str(i)
                writer = tf.train.SummaryWriter(location, session.graph_def)
        
            # learning rate goes time during learning
            lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
            mtrain.assign_lr(session, config.learning_rate * lr_decay)
            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(mtrain.lr)))
            # Train the model
            results = run_epoch(session, mtrain, training_data, training_labels, mtrain.train_op, num_steps = config.num_steps, writer = writer, verbose = True)
            #import pdb; pdb.set_trace()
            train_perplexity = np.exp(results[1] / results[2])
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            summary_str= session.run(merge_t,{score : results[0], cost: results[1], total: results[2]})
            if(writer_overview != None): writer_overview.add_summary(summary_str, i)
            
            #import pdb; pdb.set_trace()
            # Validate the model
            #results = run_epoch(session, mvalid, validate_data, validate_labels, tf.no_op(), writer = writer)
            #valid_perplexity = np.exp(results[1] / results[2])
            #print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
            #summary_str= session.run(merge_v,{score : results[0], cost: results[1], total: results[2]})
            #if(writer_overview != None): writer_overview.add_summary(summary_str, i)
            
        # Test the model
        #location = logfolder+"/test"
        #writer = tf.train.SummaryWriter(location, session.graph_def)
        #results = run_epoch(session, mtest, test_data, test_labels, tf.no_op(), writer = writer, verbose=True)
        #test_perplexity = np.exp(results[1] / results[2])
        #print("Test Perplexity: %.3f" % test_perplexity)
        
        # Save the variables to disk.
        save_path = saver.save(session, logfolder+"/model.ckpt")
        print("Model saved in file: %s" % save_path)
    print "DONE"
        #import pdb; pdb.set_trace()
            

if __name__ == '__main__':
    tf.app.run() 
