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
        if FLAGS.learning_rate != 0.0001:
            if FLAGS.learning_rate < 1:
                lr = str(FLAGS.learning_rate)[2:]
                lr = "0" + lr
            else:
                lr = str(FLAGS.learning_rate)
                lr = lr[:lr.find('.')]
            logfolder = logfolder + "_lr_"+ str(lr)
        # Model flags
        if FLAGS.optimizer != "Adam": 
            logfolder = logfolder + "_opt_"+ str(FLAGS.optimizer)
        if FLAGS.batchwise_learning:
            logfolder = logfolder + "_batchwise"
        if FLAGS.window_size != 0:
            logfolder = logfolder + "_wsize_"+str(FLAGS.window_size)
    except AttributeError:
        #in case you are not training... these values wont be set
        print "some flags were not found...probably in pilot_train."
    if FLAGS.num_layers != 2:
        logfolder = logfolder + "_nlayers_"+ str(FLAGS.num_layers)
    if FLAGS.hidden_size != 100: ###change this back to default when hiddensize is chosen
        logfolder = logfolder + "_hsz_"+ str(FLAGS.hidden_size)
    if FLAGS.keep_prob != 1.0: 
        kp = str(FLAGS.keep_prob)[2:]
        kp = "0" + kp
        logfolder = logfolder + "_drop_"+ str(kp)
    # Data flags
    if FLAGS.network != "inception": 
        logfolder = logfolder + "_net_"+ str(FLAGS.network)
    if FLAGS.normalized != False: 
        logfolder = logfolder + "_norm_"+ str(FLAGS.normalized)
    if FLAGS.feature_type != "app": 
        logfolder = logfolder + "_"+ str(FLAGS.feature_type[:5])
    if FLAGS.fc_only: 
        logfolder = logfolder + "_fc"
    if FLAGS.step_size_fnn != 1:
        logfolder = logfolder + "_stp_"+str(FLAGS.step_size_fnn)
        
    return logfolder

# the class with general empty class variables that are set by the code
# and DEFAULT parameters of which i'm not playing around with yet.
class Configuration:
    # these values are set by the data object
    batch_size = -1
    feature_dimension = -1
    output_size = -1
    num_steps = -1
    training_objects = ['modelaaa_one_cw']
    validate_objects = ['modelaaa_one_cw']
    test_objects = ['modelaaa_one_cw']

class SmallConfig(Configuration):
    """small config, for testing."""
    training_objects = ['0000','0010']#['0035'],'0025',#,'modelfee']
    validate_objects = ['0000']
    test_objects = ['0000']
    
    def __init__(self):
    	FLAGS.sample = 10
        FLAGS.num_layers = 1
        FLAGS.hidden_size = 10 #dimensionality of cell state and output
        FLAGS.max_epoch = 5
        FLAGS.max_max_epoch = 10 #5
        #FLAGS.feature_type = 'depth_estimate' #'depth' #flow or app or both
        #FLAGS.network='stijn' #'inception'
        FLAGS.dataset='../../../emerald/tmp/remote_images/tiny_set'
        
        
class DiscreteConfig(Configuration):
    """discrete labels for the OA challenge."""
    training_objects = ['0000'] #['set_7', 'set_7_1', 'set_7_2', 'set_7_3', 'set_7_4']
    validate_objects = ['0001']
    test_objects = ['0002']
    
    def __init__(self):
        FLAGS.max_epoch = 15 #50
        FLAGS.max_max_epoch = 30 #100
        #FLAGS.hidden_size = 10 #dimensionality of cell state and output
        #FLAGS.max_epoch = 2
        #FLAGS.max_max_epoch = 5
        FLAGS.continuous= False
        FLAGS.dataset='../../../emerald/tmp/remote_images/discrete_expert_2'
        self.training_objects, self.validate_objects, self.test_objects = pilot_data.get_objects()
        
class ContinueConfig(Configuration):
    """continue config, for images from laptop with oa."""
    training_objects = ['0000']
    validate_objects = ['0002']
    test_objects = ['0003']
        
    def __init__(self):
        FLAGS.max_epoch = 50 #50 #15 # let the learning rate decay faster
        FLAGS.max_max_epoch = 100 # 100 #30
        FLAGS.dataset='../../../emerald/tmp/remote_images/continuous_expert'
        self.training_objects, self.validate_objects, self.test_objects = pilot_data.get_objects()

class DaggerConfig(Configuration):
    """dagger config points to the huge dataset"""
    training_objects = ['0000']
    validate_objects = ['0001']
    test_objects = ['0002']
        
    def __init__(self):
        FLAGS.max_epoch = 15
        FLAGS.max_max_epoch = 100 #30
        FLAGS.dataset='../../../emerald/tmp/remote_images/dagger_total'
        self.training_objects, self.validate_objects, self.test_objects = pilot_data.get_objects()
        #FLAGS.finetune = True
        #FLAGS.init_model_dir= '/esat/qayd/kkelchte/tensorflow/lstm_logs/???'

class WallConfig(Configuration):
    """wall config, small set for testing memory stretching"""
    training_objects = ['0000_1']
    validate_objects = ['0001_1']
    test_objects = ['0002_1']
        
    def __init__(self):
        FLAGS.max_epoch = 50 #15
        FLAGS.max_max_epoch =  200 #30
        FLAGS.window_size = 100
        FLAGS.batch_size_fnn = 32
        FLAGS.dataset='../../../emerald/tmp/remote_images/wall_expert'
        self.training_objects, self.validate_objects, self.test_objects = pilot_data.get_objects()
        #FLAGS.finetune= True
        #FLAGS.init_model_dir= '/esat/qayd/kkelchte/tensorflow/lstm_logs/cwall_testing_wsize_100'
        
class LogitsConfig(Configuration):
    """train only on logit data ==> requires a bit a different network"""
    #training_objects = ['modeldaa','modelbae','modelacc','modelbca','modelafa', 'modelaaa']
    #validate_objects = ['modeldea','modelabe']
    #test_objects = ['modelaaa', 'modelccc']
    #
    def __init__(self):
        #Overwrite the FLAGS from the user according to the flags
        #defined in this configuration
        FLAGS.feature_type = 'app' #flow or app or both
        FLAGS.network = 'logits' #or logits_noisy
        FLAGS.learning_rate = 0.01
        training_objects, validate_objects, test_objects = pilot_data.get_objects()
    
        
class DumpsterConfig(Configuration):
    """train only on logit data ==> requires a bit a different network"""
    #training_objects = ['modeldaa','modelbae','modelacc','modelbca','modelafa', 'modelaaa']
    #validate_objects = ['modeldea','modelabe']
    #test_objects = ['modelaaa', 'modelccc']
    training_objects = ['dumpster_one_cw','ragdoll_one_cw','polaris_ranger_one_cw', 'box_one_cw']
    validate_objects = ['wooden_case_one_cw','cafe_table_one_cw']
    test_objects = ['dumpster_one_cw','ragdoll_one_cw','wooden_case_one_cw','polaris_ranger_one_cw','box_one_cw', 'cafe_table_one_cw']
    def __init__(self):
        #Overwrite the FLAGS from the user according to the flags
        #defined in this configuration
        FLAGS.feature_type = 'app' #flow or app or both
        FLAGS.dataset = 'data'
        
class TestConfig(Configuration):
    """evaluate on real data"""
    training_objects=[]
    validate_objects=[]
    test_objects=['modelbee_one_cw','modelbee_bluewall_one_cw',
                  'modelbee_bricks_one_cw', 'modelbee_gray_one_cw',
                  'modelbee_cylinder_one_cw','modelbee_cylinder_blue_one_cw',
                  'modelbee_light_one_cw','modelbee_noisy_one_cw'] #['bag_narrow'] < real data
    def __init__(self):
        """reset some flags"""
        FLAGS.dataset='testset' #'generated_world'

class BigConfig(Configuration):
    """Big config, for real training on generated data set.
        List of objects is obtained from pilot_data
        This configuration just uses all the general big data settings
        The configuration doesnt overwrite any flags so the user can still define these.
    """
    def __init__(self):
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
    elif FLAGS.model == "dis":
        config = DiscreteConfig()
    elif FLAGS.model == "cont":
        config = ContinueConfig()
    elif FLAGS.model == "cwall":
        config = WallConfig()
    elif FLAGS.model == "dagger":
        config = DaggerConfig()
    else:
        config = Configuration()
    # Some FLAGS have priority on others in case of training the fully connected final layers for a FNN
    # this means FLAGS.fc_only is true and the default values change
    if FLAGS.fc_only: #data is extracted in the same way as normal time-window-batches but with window size 1.
        FLAGS.batchwise_learning = False
        
    return config
if __name__ == '__main__':
    print 'None'
