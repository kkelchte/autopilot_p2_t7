"""
This class is used for groupings all the hyperparameters of the networks trained.
It defines a configuration class and uses the flags obtained from the test/train script in order to set the parameters.
"""
import pilot_data
import pilot
import pilot_settings

import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean(
    "normalized", None,
    "Whether or not the input data is normalized (mean substraction and divided by the variance) of the training data.")
tf.app.flags.DEFINE_boolean(
    "random_order", None,
    "Whether or not the batches during 1 epoch are shuffled in random order.")
tf.app.flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, big and dumpster.")
tf.app.flags.DEFINE_string(
    "log_directory", "/esat/qayd/kkelchte/tensorflow/lstm_logs/",
    "this is the directory for log files that are visualized by tensorboard")
tf.app.flags.DEFINE_string(
    "log_tag", "",
    "an extra tag to give some information about the job")
tf.app.flags.DEFINE_float(
    "learning_rate", "0",
    "Define the learning rate for training.")
tf.app.flags.DEFINE_integer(
    "num_layers", "0",
    "Define the num_layers for training.")
tf.app.flags.DEFINE_integer(
    "hidden_size", "0",
    "Define the hidden_size for training.")
tf.app.flags.DEFINE_float(
    "keep_prob", "0",
    "Define the keep probability for training.")
tf.app.flags.DEFINE_string(
    "optimizer", "",
    "Define the wanted optimizer for training.")
tf.app.flags.DEFINE_string(
    "network", "",
    "Define from which CNN network the features come: pcnn or inception or logits_clean or logits_noisy.")
tf.app.flags.DEFINE_string(
    "feature_type", "",
    "app or flow or both.")

# the class with general empty class variables that are set by the code
# and DEFAULT parameters of which i'm not playing around with yet.
class Configuration:
    network = 'pcnn'
    gpu= True
    random_order=True
    batch_size=1
    prefix = ""
    feature_dimension = -1
    output_size = -1
    number_of_samples = 500 
    sample = 16 #the training data will be downsampled in time with i%sample==0
    init_scale = 0.1
    learning_rate = 0.001
#   max_grad_norm = 5 #According to search space odyssey hurts clipping performance
    num_layers = 2
    num_steps = -1
    hidden_size = 50 #dimensionality of cell state and output
    max_epoch = 50 #100
    max_max_epoch = 100
    keep_prob = 1.0 #1.0 #dropout
    lr_decay = 1 / 1.15
    optimizer = 'Adam'
    normalized = False
    dataset = 'generated'
    feature_type = 'both' #flow or app or both
    training_objects = ['dumpster', 'box']
    validate_objects = ['box']
    test_objects = ['wooden_case']
    logfolder = "/esat/qayd/kkelchte/tensorflow/lstm_logs/"

    def interpret_flags(self):
        """
        Read in the settings from FLAGS and fill in the configuration accordingly
        """
        logfolder = FLAGS.log_directory+FLAGS.model
        # Read the flags from the input
        if FLAGS.learning_rate != 0: 
            self.learning_rate = FLAGS.learning_rate
            lr = 0
            if FLAGS.learning_rate < 1:
                lr = str(FLAGS.learning_rate)[2:]
                lr = "0" + lr
            else:
                lr = str(FLAGS.learning_rate)
                lr = lr[:lr.find('.')]
            logfolder = logfolder + "_lr_"+ lr
        if FLAGS.num_layers != 0:
            self.num_layers = FLAGS.num_layers
            logfolder = logfolder + "_layers_"+ str(FLAGS.num_layers)
        if FLAGS.hidden_size != 0: 
            self.hidden_size = FLAGS.hidden_size
            logfolder = logfolder + "_size_"+ str(FLAGS.hidden_size)
        if FLAGS.keep_prob != 0: 
            self.keep_prob = FLAGS.keep_prob
            kp = str(FLAGS.keep_prob)[2:]
            kp = "0" + kp
            logfolder = logfolder + "_drop_"+ str(kp)
        if FLAGS.optimizer != "": 
            self.optimizer = FLAGS.optimizer
            logfolder = logfolder + "_opt_"+ str(FLAGS.optimizer)
        if FLAGS.network != "": 
            self.network = FLAGS.network
            logfolder = logfolder + "_net_"+ str(FLAGS.network)
        if FLAGS.normalized != None: 
            self.normalized = FLAGS.normalized
            logfolder = logfolder + "_norm_"+ str(FLAGS.normalized)
        if FLAGS.random_order != None: 
            self.random_order = FLAGS.random_order
            logfolder = logfolder + "_random_order_"+ str(FLAGS.random_order)
        if FLAGS.feature_type != "": 
            self.feature_type = FLAGS.feature_type
            logfolder = logfolder + "_"+ str(FLAGS.feature_type)
        if FLAGS.log_tag != "": 
            logfolder = logfolder + "_"+ str(FLAGS.log_tag)
        self.logfolder = logfolder

class SmallConfig(Configuration):
    """small config, for testing."""
    network='inception'
    sample = 20
    num_layers = 1
    hidden_size = 10 #dimensionality of cell state and output
    max_epoch = 2
    max_max_epoch = 10
    training_objects = ['modelaaa','modelcba','modelfee']
    validate_objects = ['modelaaa']
    test_objects = ['modelaaa']
    feature_type = 'both' #flow or app or both
    
class MediumConfig(Configuration):
    """small config, for testing."""
    sample = 16
    num_layers = 2
    hidden_size = 10 #dimensionality of cell state and output
    max_epoch = 50
    max_max_epoch = 100
    training_objects = ['modeldaa','modelbae','modelacc','modelbca','modelafa', 'modelaaa']
    validate_objects = ['modeldea','modelabe']
    test_objects = ['modelaaa', 'modelccc']
    feature_type = 'app' #flow or app or both
        
class BigConfig(Configuration):
    """Big config, for real training on generated data set.
        List of objects is obtained from pilot_data
    """
    training_objects, validate_objects, test_objects = pilot_data.get_objects(Configuration.dataset)

def get_config():
    """deduce the configuration for training, validation and testing according to the flags set by the user
    Args:
        FLAGS: set by the user when calling python pilot_train.py --flag_1 value_1
    Returs:
        train_config: configuration for training
        valid_config: configuration for validating
        test_config: configuration for testing
    """
    # Get general configuration
    if FLAGS.model == "big":
        train_config = BigConfig()
        valid_config = BigConfig()
        test_config = BigConfig()
    elif FLAGS.model == "small":
        train_config = SmallConfig()
        valid_config = SmallConfig()
        test_config = SmallConfig()
    elif FLAGS.model == "medium":
        train_config = MediumConfig()
        valid_config = MediumConfig()
        test_config = MediumConfig()
    else:
        train_config = Configuration()
        valid_config = Configuration()
        test_config = Configuration()
    
    # Specify prefix for naming
    train_config.prefix = "training"
    valid_config.prefix = "validation"
    test_config.prefix = "test"
    
    # Interpret FLAGS for each configuration
    train_config.interpret_flags()
    valid_config.interpret_flags()
    test_config.interpret_flags()
    
    # Num_steps:
    valid_config.num_steps = 1
    test_config.num_steps = 1
    
    # Batch sizes:
    valid_config.batch_size = 1
    test_config.batch_size = 1
    
    # adapt test data and labels according to batchsize 1 ~ make 1 movie by concatenating the different testmovies
    #test_data = np.reshape(test_data,(1, -1, test_data.shape[2]))      
    #test_labels = np.reshape(test_labels,(1, -1, test_labels.shape[2]))
    

    return train_config, valid_config, test_config
