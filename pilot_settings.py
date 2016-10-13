"""
This class is used for groupings all the hyperparameters of the networks trained.
It defines a configuration class and uses the flags obtained from the test/train script in order to set the parameters.
"""
import pilot_data
import pilot_model
#import xml.etree.cElementTree as ET
from lxml import etree as ET
import tensorflow as tf
import os.path


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
        if FLAGS.finetune:
            logfolder = logfolder + "_fi"
        if FLAGS.skip_initial_state:
            logfolder = logfolder + "_no_init_state"
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

def save_config(logfolder, file_name = "configuration"):
    """
    save all the FLAG values in a config file / xml file
    """
    print "Save configuration in xml."
    root = ET.Element("conf")
    flg = ET.SubElement(root, "flags")
    
    flags_dict = FLAGS.__dict__['__flags']
    for f in flags_dict:
        #print f, flags_dict[f]
        ET.SubElement(flg, f, name=f).text = str(flags_dict[f])
    tree = ET.ElementTree(root)
    tree.write(os.path.join(logfolder,file_name+".xml"), encoding="us-ascii", xml_declaration=True, method="xml", pretty_print=True)

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
    training_objects = ['0000','0010','0035','0025']
    validate_objects = ['0000']
    test_objects = ['0000']
    
    def __init__(self):
        FLAGS.max_num_windows = 1000 
        FLAGS.sample = 1
        FLAGS.num_layers = 1
        FLAGS.hidden_size = 10 #dimensionality of cell state and output
        FLAGS.max_epoch = 5
        FLAGS.max_max_epoch = 10
        #FLAGS.feature_type = 'depth_estimate' #'depth' #flow or app or both
        #FLAGS.network='stijn' #'inception'
        FLAGS.dataset='tiny_set'
        
class DepthFCConfig(Configuration):
    """depth fully connected layers trained on sequential OA dataset"""
    training_objects = ['sequential_oa_0012_0.5_2']
    validate_objects = ['sequential_oa_0004_0.5_1']
    test_objects = ['sequential_oa_0008_0_1']
        
    def __init__(self):
        FLAGS.max_epoch = 50
        FLAGS.max_max_epoch = 100
        FLAGS.batch_size_fnn = 100
        FLAGS.fc_only = True
        FLAGS.hidden_size = 400
        FLAGS.network = 'stijn'
        FLAGS.feature_type = 'depth_estimate'
        FLAGS.dataset='sequential_oa'
        self.training_objects, self.validate_objects, self.test_objects = pilot_data.get_objects()
        
class DepthLSTMConfig(Configuration):
    """depth LSTM layers trained on sequential OA dataset"""
    training_objects = ['sequential_oa_0012_0.5_2']
    validate_objects = ['sequential_oa_0004_0.5_1']
    test_objects = ['sequential_oa_0008_0_1']
        
    def __init__(self):
        FLAGS.max_epoch = 15
        FLAGS.max_max_epoch = 30
        FLAGS.max_num_threads = 20 #for 4G gpu should be ok?
        FLAGS.network = 'stijn'
        FLAGS.feature_type = 'depth_estimate'
        FLAGS.dataset='sequential_oa'
        self.training_objects, self.validate_objects, self.test_objects = pilot_data.get_objects()
        
class IncFCConfig(Configuration):
    """inception fully connected layers trained on sequential OA dataset"""
    training_objects = ['sequential_oa_0012_0.5_2']
    validate_objects = ['sequential_oa_0004_0.5_1']
    test_objects = ['sequential_oa_0008_0_1']
        
    def __init__(self):
        FLAGS.max_epoch = 50 #leave 50 instead of 10 in order to make sure it convergers to a minimum
        FLAGS.max_max_epoch = 150
        FLAGS.batch_size_fnn = 100
        FLAGS.fc_only = True
        FLAGS.hidden_size = 400
        FLAGS.dataset='inc_fc_dagger'
        self.training_objects, self.validate_objects, self.test_objects = pilot_data.get_objects()
        
class IncLSTMConfig(Configuration):
    """depth LSTM layers trained on sequential OA dataset"""
    training_objects = ['sequential_oa_0012_0.5_2']
    validate_objects = ['sequential_oa_0004_0.5_1']
    test_objects = ['sequential_oa_0008_0_1']
        
    def __init__(self):
        FLAGS.max_epoch = 30 #15
        FLAGS.max_max_epoch = 60 #30
        FLAGS.max_num_threads = 20 #for 4G gpu should be ok?        
        self.training_objects, self.validate_objects, self.test_objects = pilot_data.get_objects()       

class OneIncLSTMConfig(Configuration):
    """Inc LSTM layers trained on one room OA dataset"""
    training_objects = ['one_oa_0000']
    validate_objects = ['one_oa_0130']
    test_objects = ['one_oa_0140']
        
    def __init__(self):
        FLAGS.max_epoch = 15
        FLAGS.max_max_epoch = 30
        FLAGS.max_num_threads = 20 #for 4G gpu should be ok? 
        FLAGS.dataset='one_oa'
        self.training_objects, self.validate_objects, self.test_objects = pilot_data.get_objects()    

class OneIncFCConfig(Configuration):
    """Inc LSTM layers trained on one room OA dataset"""
    training_objects = ['one_oa_0000']
    validate_objects = ['one_oa_0130']
    test_objects = ['one_oa_0140']
        
    def __init__(self):
        FLAGS.max_epoch = 50
        FLAGS.max_max_epoch = 100
        FLAGS.batch_size_fnn = 100 
        FLAGS.fc_only = True
        FLAGS.hidden_size = 400
        FLAGS.dataset='one_oa'
        self.training_objects, self.validate_objects, self.test_objects = pilot_data.get_objects() 

class CombConfig(Configuration):
    """comb config points to the huge dataset and goes over it with different window sizes"""
    training_objects = ['expert','expert1','expert2','expert3','expert4']
    validate_objects = []
    test_objects = ['expert']
        
    def __init__(self):
        # Each epoch is repeated for each window size
        FLAGS.max_epoch = 15 # let the learning rate decay faster
        FLAGS.max_max_epoch = 30
        FLAGS.max_num_windows = 500 #Number of initial state calculations each epoch with windowwise learning 
        FLAGS.feature_type = 'depth_estimate' #'depth' #flow or app or both
        FLAGS.network='stijn' #'inception'
        if FLAGS.fc_only:
            FLAGS.hidden_size = 400
        FLAGS.dataset='dagger_esat'
        
        
class SelectedConfig(Configuration):
    """continue config, for images from laptop with oa."""
    training_objects = ['0011']
    validate_objects = ['0011']
    test_objects = ['0011']
        
    def __init__(self):
        FLAGS.max_epoch = 15 # let the learning rate decay faster
        FLAGS.max_max_epoch = 30
        FLAGS.dataset='selected'
        if FLAGS.fc_only:
            FLAGS.hidden_size = 400
        
        self.training_objects, self.validate_objects, self.test_objects = pilot_data.get_objects()
        
        
class WallConfig(Configuration):
    """wall config, small set for testing memory stretching"""
    training_objects = ['0000_1']
    validate_objects = ['0001_1']
    test_objects = ['0002_1']
        
    def __init__(self):
        FLAGS.max_epoch = 15
        FLAGS.max_max_epoch = 30
        FLAGS.dataset='../../../emerald/tmp/remote_images/wall_expert'
        self.training_objects, self.validate_objects, self.test_objects = pilot_data.get_objects()
        FLAGS.batchwise_learning = True
        FLAGS.finetune = True
        FLAGS.init_model_dir= '/esat/qayd/kkelchte/tensorflow/lstm_logs/cwall_batchwise'


class WallWindowConfig(Configuration):
    """wall config, small set for testing memory stretching"""
    training_objects = ['0010_1','0010_2','0010_3','0010_4']
    validate_objects = ['0011_1']
    test_objects = ['0012_1']
        
    def __init__(self):
        FLAGS.max_epoch = 20 #15
        FLAGS.max_max_epoch =  50 #30
        FLAGS.dataset='../../../emerald/tmp/remote_images/wall_expert'
        
class ExpertConfig(Configuration):
    """Expert on random wall challenges."""
    training_objects = ['0000'] 
    validate_objects = ['0001']
    test_objects = ['0002']
    
    def __init__(self):
        FLAGS.max_epoch = 15 #50
        FLAGS.max_max_epoch = 30
        FLAGS.dataset='../../../emerald/tmp/remote_images/continuous_expert'
        self.training_objects, self.validate_objects, self.test_objects = pilot_data.get_objects()
        FLAGS.window_size = 300
        FLAGS.batch_size_fnn = 2
        FLAGS.max_num_threads = 5
        #FLAGS.sliding_window = True
        
#### Old configurations        
class DiscreteConfig(Configuration):
    """discrete labels for the OA challenge."""
    training_objects = ['0000'] #['set_7', 'set_7_1', 'set_7_2', 'set_7_3', 'set_7_4']
    validate_objects = ['0001']
    test_objects = ['0002']
    
    def __init__(self):
        FLAGS.max_epoch = 15 #50
        FLAGS.max_max_epoch = 100
        #FLAGS.hidden_size = 10 #dimensionality of cell state and output
        #FLAGS.max_epoch = 2
        #FLAGS.max_max_epoch = 5
        FLAGS.continuous= False
        FLAGS.dataset='../../../emerald/tmp/remote_images/discrete_expert_2'
        self.training_objects, self.validate_objects, self.test_objects = pilot_data.get_objects()
        FLAGS.finetune = True
        FLAGS.init_model_dir= '/esat/qayd/kkelchte/tensorflow/lstm_logs/dis_wsize_300'

        
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
    elif FLAGS.model == "comb":
        config = CombConfig()
    elif FLAGS.model == "logits":
        config = LogitsConfig()
    elif FLAGS.model == "dumpster":
        config = DumpsterConfig()
    elif FLAGS.model == "test":
        config = TestConfig()
    elif FLAGS.model == "expert":
        config = ExpertConfig()
    elif FLAGS.model == "seldat":
        config = SelectedConfig()
    elif FLAGS.model == "cwall":
        config = WallConfig()
    elif FLAGS.model == "winwall":
        config = WallWindowConfig()
    elif FLAGS.model == "depth_fc":
        config = DepthFCConfig()
    elif FLAGS.model == "depth_lstm":
        config = DepthLSTMConfig()
    elif FLAGS.model == "inc_fc":
        config = IncFCConfig()
    elif FLAGS.model == "inc_lstm":
        config = IncLSTMConfig()
    elif FLAGS.model == "one_inc_lstm":
        config = OneIncLSTMConfig()
    elif FLAGS.model == "one_inc_fc":
        config = OneIncFCConfig()
    else:
        config = Configuration()
    # Some FLAGS have priority on others in case of training the fully connected final layers for a FNN
    # this means FLAGS.fc_only is true and the default values change
    if FLAGS.fc_only: #data is extracted in the same way as normal time-window-batches but with window size 1.
        FLAGS.batchwise_learning = False
    if not os.path.isdir(config.training_objects[0]):
        print 'concatenate root and dataset to training objects as full path is not given.'
        config.training_objects = [os.path.join(FLAGS.data_root,FLAGS.dataset,o) for o in config.training_objects]
        config.validate_objects = [os.path.join(FLAGS.data_root,FLAGS.dataset,o) for o in config.validate_objects]
        config.test_objects = [os.path.join(FLAGS.data_root,FLAGS.dataset,o) for o in config.test_objects]
    return config
if __name__ == '__main__':
    print 'None'
