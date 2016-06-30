"""
This class is used for groupings all the hyperparameters of the networks trained.
It defines a configuration class and uses the flags obtained from the test/train script in order to set the parameters.
"""
import pilot_data
import pilot_model

import tensorflow as tf


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, big and dumpster. Setting the model to any model different than big overwrites some default AND user defined FLAGS relevant or fixed for that model.")
tf.app.flags.DEFINE_string(
    "log_directory", "/esat/qayd/kkelchte/tensorflow/lstm_logs/",
    "This is the directory for log files that are visualized by tensorboard")
tf.app.flags.DEFINE_string(
    "log_tag", "testing",
    "an extra tag to give some information about the job")


def extract_logfolder():
    """
    Read in the settings from FLAGS and see if they differ from the default value
    if that is the case, name the logfolder according to special settings
    """
    logfolder = FLAGS.log_directory+FLAGS.model
    # Read the flags from the input
    # Training flags
    if FLAGS.log_tag == "testing": 
        logfolder = logfolder + "_"+ str(FLAGS.log_tag)
    elif FLAGS.log_tag != "no_test": # not testing
        logfolder = logfolder + "_"+ str(FLAGS.log_tag)
    try:#only runs this when these values are actually set
        if not FLAGS.random_order: 
            logfolder = logfolder + "_rndmordr_"+ str(FLAGS.random_order)
        if FLAGS.learning_rate != 0.001:
            if FLAGS.learning_rate < 1:
                lr = str(FLAGS.learning_rate)[2:]
                lr = "0" + lr
            else:
                lr = str(FLAGS.learning_rate)
                lr = lr[:lr.find('.')]
            logfolder = logfolder + "_lr_"+ lr
        # Model flags
        if FLAGS.optimizer != "Adam": 
            logfolder = logfolder + "_opt_"+ str(FLAGS.optimizer)
        if not FLAGS.batchwise_learning:
            logfolder = logfolder + "_windowwise"
    except AttributeError:
        #in case you are not training... these values wont be set
        print "some flags were not found...probably in pilot_train."
    if FLAGS.num_layers != 2:
        logfolder = logfolder + "_nlayers_"+ str(FLAGS.num_layers)
    if FLAGS.hidden_size != 50:
        logfolder = logfolder + "_sz_"+ str(FLAGS.hidden_size)
    if FLAGS.keep_prob != 1.0: 
        kp = str(FLAGS.keep_prob)[2:]
        kp = "0" + kp
        logfolder = logfolder + "_drop_"+ str(kp)
    # Data flags
    if FLAGS.network != "pcnn": 
        logfolder = logfolder + "_net_"+ str(FLAGS.network)
    if FLAGS.normalized != False: 
        logfolder = logfolder + "_norm_"+ str(FLAGS.normalized)
    if FLAGS.feature_type != "app": 
        logfolder = logfolder + "_"+ str(FLAGS.feature_type)
    return logfolder

# the class with general empty class variables that are set by the code
# and DEFAULT parameters of which i'm not playing around with yet.
class Configuration:
    # these values are set by the data object
    batch_size = -1
    feature_dimension = -1
    output_size = -1
    num_steps = -1
    training_objects = ['modelaaa']
    validate_objects = ['modelaaa']
    test_objects = ['modelaaa']

class SmallConfig(Configuration):
    """small config, for testing."""
    training_objects = ['modelbaa','modelaba','modelbee']#,'modelfee']
    #training_objects = ['modelaaa']
    validate_objects = ['modelaaa']
    test_objects = ['modelaaa']
    
    def __init__(self):
        FLAGS.sample = 16
        FLAGS.num_layers = 1
        FLAGS.hidden_size = 2 #dimensionality of cell state and output
        FLAGS.max_epoch = 5
        FLAGS.max_max_epoch = 1 #5
        FLAGS.feature_type = 'app' #flow or app or both
        FLAGS.network='inception'
        
class LogitsConfig(Configuration):
    """train only on logit data ==> requires a bit a different network"""
    #training_objects = ['modeldaa','modelbae','modelacc','modelbca','modelafa', 'modelaaa']
    #validate_objects = ['modeldea','modelabe']
    #test_objects = ['modelaaa', 'modelccc']
    training_objects, validate_objects, test_objects = pilot_data.get_objects()
    
    def __init__(self):
        #Overwrite the FLAGS from the user according to the flags
        #defined in this configuration
        FLAGS.feature_type = 'app' #flow or app or both
        FLAGS.network = 'logits' #or logits_noisy
        FLAGS.learning_rate = 0.01
        
class DumpsterConfig(Configuration):
    """train only on logit data ==> requires a bit a different network"""
    #training_objects = ['modeldaa','modelbae','modelacc','modelbca','modelafa', 'modelaaa']
    #validate_objects = ['modeldea','modelabe']
    #test_objects = ['modelaaa', 'modelccc']
    training_objects = ['dumpster','ragdoll','polaris_ranger', 'box']
    validate_objects = ['wooden_case','cafe_table']
    test_objects = ['dumpster','ragdoll','wooden_case','polaris_ranger','box', 'cafe_table']
    def __init__(self):
        #Overwrite the FLAGS from the user according to the flags
        #defined in this configuration
        FLAGS.feature_type = 'app' #flow or app or both
        FLAGS.dataset = 'data'
        
class TestConfig(Configuration):
    """evaluate on real data"""
    training_objects=[]
    validate_objects=[]
    test_objects=['modelbee','modelbee_bluewall',
                  'modelbee_bricks', 'modelbee_gray',
                  'modelbee_cylinder','modelbee_cylinder_blue',
                  'modelbee_light','modelbee_noisy'] #['bag_narrow'] < real data
    def __init__(self):
        """reset some flags"""
        FLAGS.dataset='testset' #'generated_world'

class BigConfig(Configuration):
    """Big config, for real training on generated data set.
        List of objects is obtained from pilot_data
        This configuration just uses all the general big data settings
        The configuration doesnt overwrite any flags so the user can still define these.
    """
    training_objects, validate_objects, test_objects = pilot_data.get_objects()

def get_config():
    """deduce the configuration for training, validation and testing according to the flags set by the user
    Args:
        FLAGS: set by the user when calling python pilot_train.py --flag_1 value_1
    Returs:
        config: configuration for training
    """
    # Get general configuration
    if FLAGS.model == "big":
        config = BigConfig()
    elif FLAGS.model == "small":
        config = SmallConfig()
    elif FLAGS.model == "logits":
        config = LogitsConfig()
    elif FLAGS.model == "dumpster":
        config = DumpsterConfig()
    elif FLAGS.model == "test":
        config = TestConfig()
    else:
        config = Configuration()
    
    return config
if __name__ == '__main__':
    print 'None'
