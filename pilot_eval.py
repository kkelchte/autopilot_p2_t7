#This code groups the functions for evaluating a model by unrolling it one step
#and predicting.
#

import numpy as np
import scipy.io as sio

import pilot_model
import pilot_data
import pilot_settings
import convert_control_output

import time

import sys
import pyinotify

import os, shutil, os.path
import re
import skimage
import skimage.transform
from skimage import io
 
#import matplotlib.pyplot as plt
import copy
from lxml import etree as ET

import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    "model_dir","/esat/qayd/kkelchte/tensorflow/lstm_logs/big_net_inception", #features_log/big_app",
    "define the folder from where the model is restored. It doesnt have to be the same as the log folder.")
tf.app.flags.DEFINE_string(
    "machine","qayd",
    "define the machine on which the LSTM is loaded and from which features are read.")
tf.app.flags.DEFINE_boolean(
    "save_states_eval", False,
    "Whether or not the innerstates are saved away during evaluation.")
tf.app.flags.DEFINE_boolean(
    "online", True,
    "Choose online evaluation when script waits for features and outputs control")
tf.app.flags.DEFINE_boolean(
    "ssh", True,
    "In case I ssh to jade the source folder of features should be changed and the time delay should be added.")
tf.app.flags.DEFINE_string(
    "model_name", "remote_set7d3_sz_100_net_inception",
    "pick which control model is used for steering the online control.")
tf.app.flags.DEFINE_boolean(
    "skip_initial_state", False,
    "In case the model is trained on time-window-batches from zero initial state.")
tf.app.flags.DEFINE_boolean(
    "fly_straight", False,
    "Used for changing the control output to go straight.")
tf.app.flags.DEFINE_integer(
    "max_num_threads", 10,
    "Set the maximum number of threads according to the size of the GPU. (20 is ok for 4G)")
tf.app.flags.DEFINE_string(
    "checkpoint_name", "model.ckpt",
    "pick which checkpoint you want to restore.")



#global variable feature bucket is a buffer used by the eventhandler to collect new arrived features as a list of paths
current_feature = "" # feature now in queue ready to be processed
last_feature = "" # feature just processed
frame = 0
local_state = None
mtest = None
delay=0
feature_queue = []
mean=0
variance=1

def run_one_step_unrolled(session, model, data, targets, eval_op, writer=None, verbose=False, number_of_samples=500, matfile = ""):
    """Unroll the model 1 step and evaluate the batches of data 1step at the time and 1 batch at the time.
    Args:
        session: current session in which operations are defined
        model: model object that contains operations and represents network
        data: an np array containing 1 batch of data [batchsize, num_steps, feature_dimension]
        targets: [batch, steps, 4]
    """
    # Validate / test one step at the time going through the data
    # data is only 1 movie. Not a batch of different movies.
    # this function can also be used when you want to unroll over a few time steps but not all
    print "One step at the time... "
    start_time = time.time()
    costs = 0.0
    iters = 0.0
    score = 0.0
    if not FLAGS.fc_only: state = session.run(model.initial_state)
    
    #In case you loop over frames in steps of size 1
    list_of_states = []
    
    #define what frames you want to write
    frames_to_write=[]
    if(writer != None):
        write_freq = int((data.shape[1])/number_of_samples)
        if write_freq != 0: frames_to_write = [f for f in range(data.shape[1]) if (f % write_freq) == 0]
        #if frames - steps > 100 ==> dont do modulo of zero
        #in case frames - steps < 100 ==> save every frame because there are less than 100 steps
        #the summary writer will only record the last <100 steps
        else: frames_to_write = range(data.shape[1])
    
    for f in range(data.shape[1]-FLAGS.step_size_fnn):
        if not FLAGS.fc_only:
            feed_dict = {model.inputs: data[:,f:f+1,:], 
                        model.initial_state: state, 
                        model.targets: targets[:,f:f+1,:]}
            if f in frames_to_write: #write data away from time to time
                outputs, state, current_loss, _, summary_str= session.run([model.logits, model.state, model.cost, eval_op, model.merge()],feed_dict)
                writer.add_summary(summary_str, f)
            else:
                # outputs is of shape (1*batchsize, outputsize)
                outputs, state, current_loss, _= session.run([model.logits, model.state, model.cost, eval_op],feed_dict)
            iters += data.shape[0] #add these batches as they all where handled
            list_of_states.append(state)
        else:#FLAGS.fc_only
            #concatenate multiple features if wanted
            frame = np.zeros((1,1,FLAGS.step_size_fnn*data.shape[2]))
            for i in range(FLAGS.step_size_fnn):
                frame[0,0,i*data.shape[2]:(i+1)*data.shape[2]]=data[0,f+i,:]
            #import pdb; pdb.set_trace()
            feed_dict = {model.inputs: frame, 
                        model.targets: targets[:,f:f+1,:]}
            if f in frames_to_write: #write data away from time to time
                outputs, current_loss, _, summary_str= session.run([model.logits, model.cost, eval_op, model.merge()],feed_dict)
                writer.add_summary(summary_str, f)
            else:
                # outputs is of shape (1*batchsize, outputsize)
                outputs, current_loss, _= session.run([model.logits, model.cost, eval_op],feed_dict)
            iters += data.shape[0] #add these batches as they all where handled
	
	for m in range(data.shape[0]): #loop over the movies of the batch (in general will be one)
            if np.argmax(outputs[m]) == np.argmax(targets[m,f,:]):#see if this frame is calculated correctly
                score += 1.0 #if your score is correct count it up
            if verbose and (f % int(data.shape[1]/15) == 0):
                print("Frame: %d batch: %d target: %d prediction: %d speed: %.3f fps"\
                    %(f, m, np.argmax(targets[m,f,:]), np.argmax(outputs[m]),
                    iters * data.shape[0] / (time.time() - start_time)))    
        costs += current_loss
    
    # Keep track of cell states saved in the list_of_states
    if(writer != None) and (matfile != "") and FLAGS.save_states_eval and not FLAGS.fc_only:
        #states = [batchsize, num_steps*hidden_size*num_layers*2 (output;state)]
        states = np.concatenate(list_of_states)
        trgts = targets.reshape((-1,outputs.shape[1]))
        d = {'states': states, 'targets': targets}
        sio.savemat(matfile,d, appendmat=True)
        
    if verbose:
        print "Average loss: ",(costs/iters)," average score: ",(score/iters)
    ##Add call for tensor accuracy + write it away
    #loss is an average in case of numstep unrolled but not yet in case of one step unrolled.
    return score, (costs/iters), iters

def get_initial_states(models, indices, meval, data_list, sess):
    '''Get the needed initial states for the models to be trained according to the beginning parts of the movies for windowwise training. 
    Use a seperate session in order to keep the memory at low level.
    Args:
        models: training models of which model.initial_state has to be set.
        indices: the tuple/mov/start indices indicating the training window for that batch.
        meval: the evaluation (validation) model used.
        data_list: the original list of grouped tuples according to movie lengths
    Return:
        resulting states listed up in nested lists: the outer list corresponding to the number of models
        the inner list corresponding to the number of times a window batch is applied in one epoch.
    '''
    #### new implementation
    #create pop up list with all index_tuples of indices from which an initial states needs to be found
    all_movies=[ (oi, ii) for oi in range(len(indices)) for ii in range(len(indices[oi]))]
    total_indices = copy.deepcopy(indices)
    
        
    #start some threads and a coordinator
    #each thread should do:
    #   pop index
    #   extract initial state by looping over frames
    #   save initial states in initial_states_all
    #--> should be a 3D array instead of 2 nested lists of 1d states
    
    # Thread body: loop until the coordinator indicates a stop was requested.
    # If some condition becomes true, ask the coordinator to stop.
    def MyLoop(coord,all_movies):#meval, data_list
        while not coord.should_stop():
            try:
                oi, ii = all_movies.pop()
                print 'im evaluating: outer index ', oi,' inner index ',ii
                # Get initial state
                model = models[oi]
                zerostate = sess.run(model.initial_state)
                initial_states = np.zeros(zerostate.shape)
                #print 'initial_states: ',initial_states
                for ib in range(indices[oi][ii].shape[0]):#index of movie in batch
                    #import pdb; pdb.set_trace()
                    [tup_ind, mov_ind, start_ind] = indices[oi][ii][ib] #get indices needed for finding data
                    state = sess.run(meval.initial_state) #initialize eval network with zero state
                    if not FLAGS.skip_initial_state:
                        for f in range(start_ind): #step through data sequence up until the start of the trainingswindow
                            feed_dict = {meval.inputs: np.array([[data_list[tup_ind][0][mov_ind,f,:]]]), meval.initial_state: state}
                            outputs, state = sess.run([meval.logits, meval.state], feed_dict)
                            #state = sess.run([meval.state], feed_dict)
                        #import pdb; pdb.set_trace()
                    initial_states[ib]=state
                total_indices[oi][ii]=initial_states
            except IndexError:
                print 'Innerstate evaluation finished. Wait for threads to stop.'
                coord.request_stop()
        if len(all_movies)==0 :
            coord.request_stop()
    try:
        # Main code: create a coordinator.
        coord = tf.train.Coordinator()
        # Create 10 threads that run 'MyLoop()'
        #num_threads=min(10,len(all_movies)/2)
        # each batch for a certain model can be prepared with another thread
        num_threads=min([len(i) for i in indices])
        num_threads=min(FLAGS.max_num_threads, num_threads) #avoid having way too many threads
    
        threads = [tf.train.threading.Thread(target=MyLoop, args=(coord,all_movies)) for i in xrange(num_threads)]
        print 'number of threads: ',num_threads
        # Start the threads and wait for all of them to stop.
        for t in threads: t.start()
        coord.join(threads, stop_grace_period_secs=240) #wait max 4minutes to stop threads before raising exception
    except RuntimeError:
        print "Thread took more than 4 minutes to sleep so we sleep for an extra 4 minutes..."
        time.sleep(240)
    except Exception as e:
        print "Thread is still not ready so something is probably wrong? Or this is raised by another type of exception.", e.value
        
    return total_indices
    
def get_initial_states_single(models, indices, meval, data_list, sess):

    #### old implementation ###
    initial_states_all = []
    total_num_of_states = len(models)*len(indices[0])*indices[0][0].shape[0]
    count = 0
    for im in range(len(models)):#index for model
        model = models[im]
        zerostate = sess.run(model.initial_state)
        #print 'zerostate: ',zerostate
        initial_states_local = [] #list for each time a batch-window is applied in one epoch
        for il in range(len(indices[im])):
            if FLAGS.skip_initial_state:
                initial_states_local.append(zerostate)
            else:
                initial_states = np.zeros(zerostate.shape)
                #print 'initial_states: ',initial_states
                for ib in range(indices[im][il].shape[0]):#index of movie in batch
                    print 'init_state: ', count, ' of ', total_num_of_states
                    count+=1
                    #import pdb; pdb.set_trace()
                    [tup_ind, mov_ind, start_ind] = indices[im][il][ib] #get indices needed for finding data
                    #state = get_state(meval, start_ind, tup_ind, mov_ind, data_list)
                    state = sess.run(meval.initial_state) #initialize eval network with zero state
                    for f in range(start_ind): #step through data sequence up until the start of the trainingswindow
                        feed_dict = {meval.inputs: np.array([[data_list[tup_ind][0][mov_ind,f,:]]]),
                            meval.initial_state: state}
                        outputs, state = sess.run([meval.logits, meval.state], feed_dict)
                    #state = sess.run([meval.state], feed_dict)
                    initial_states[ib]=state
                initial_states_local.append(initial_states)
        initial_states_all.append(initial_states_local)
    return initial_states_all
    
def evaluate(logfolder, config, scope="model"):
    """Get some evaluation data and test the performance
    of a restored model. The model is chosen according to the flags set by the user.
    """
    with tf.Graph().as_default():
        test_data_list = pilot_data.prepare_data_list(config.test_objects)
        config.output=test_data_list[0][1].shape[2]
        config.feature_dimension=test_data_list[0][0].shape[2]*FLAGS.step_size_fnn
        #set params according to the shape of the obtained data
        with tf.variable_scope(scope):
            mtest = pilot_model.LSTMModel(False, config.output, config.feature_dimension, prefix='eval')
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
            saver.restore(session, FLAGS.model_dir+"/model.ckpt")
            print "model restored", FLAGS.model_dir+"/model.ckpt"
            location = logfolder+"/eval"
            writer = tf.train.SummaryWriter(location, graph=session.graph)
            #decide of what movies you want to keep track.
            movies_to_write_away = [0]
            #acc = 0.0
            #per = 0.0
            #acc_max = 0.0
            #acc_min = 1.0
            test_results = []
            for j in range(len(test_data_list)):
                if j in movies_to_write_away: wrtr = writer
                else: wrtr = None
                #get the shape of the inner state of the model
                #prestate = session.run(mtest.zero_state)
                #initial_state = np.zeros(prestate.shape)
                #assign zeros to the initial state of the model
                #session.run(tf.assign(mtest.initial_state, initial_state))
                results = run_one_step_unrolled(session, 
                                            mtest, 
                                            test_data_list[j][0], 
                                            test_data_list[j][1], 
                                            tf.no_op(),
                                            writer=wrtr,
                                            verbose=False)
                test_results.append(results)
                #perplexity = np.exp(results[1] / results[2])
                #accuracy = results[0]/results[2]
                #print("Accuracy: %.3f" % (accuracy))
                #if acc_max < accuracy: acc_max = accuracy
                #if acc_min > accuracy: acc_min = accuracy
                #acc = acc+accuracy/len(test_data_list)
                #per = per+perplexity/len(test_data_list)
                #print("Test perplexity: %.3f" %perplexity)
            losses = [mres[1] for mres in test_results]
            if not FLAGS.continuous:
                accuracies = [mres[0]/mres[2] for mres in test_results]
                print("Test: Average Accuarcy over different unrolled models: %f, Max: %f, Min: %f, Loss: %f." % (sum(accuracies)/len(accuracies), max(accuracies), min(accuracies), sum(losses)/len(losses)))
            else:#continuous case
                print("Test: Average loss over different unrolled models: %f, Max: %f, Min: %f." % (sum(losses)/len(losses),max(losses),min(losses)))
            f = open(logfolder+"/results.txt", 'a')
            f.write('Test: Average loss: %f, Max: %f, Min: %f.\n' % (sum(losses)/len(losses),max(losses),min(losses)))
            f.close()
            #print 'average accuracy: ',acc
            #print 'max accuracy: ',acc_max
            #print 'min accuracy: ',acc_min
            #print 'average perplexity: ',per
    print 'evaluation...finished!'
    
def load_image(path_to_image):
    """Read in image from file and export as proper feature."""
    scale=0.2
    max_shape=int(360*640*3*scale*scale)
    if not os.path.isfile(path_to_image):
        raise IOError("path_to_image does not exist.")
    success=False
    while not success:
        try:
            #import pdb; pdb.set_trace()
            im_array = io.imread(path_to_image)
            im_array = skimage.transform.rescale(im_array, 0.2)
            success = True
        except:
            print 'failed to load',path_to_image
            time.sleep(0.005)
    return np.reshape(im_array, [1,1,max_shape])

def evaluate_online(logfolder, config, scope="model"):
    """This function is called from the main thread and coordinates the one step evaluation.
    Args:
        logfolder: the log folder is extracted from all the different user configurations
        config: the configurations are grouped in a configuration object    
    """
    global current_feature
    global last_feature
    global frame
    global local_state
    global mtest
    global delay
    global session
    global feature_queue
    global mean
    global variance
    
    if FLAGS.network == 'no_cnn':
        source_dir = '/esat/emerald/tmp/remote_images/set_online/RGB/'
        des_dir = '/esat/emerald/tmp/control_output/'
    elif FLAGS.ssh:
        source_dir = '/esat/'+FLAGS.machine+'/tmp/remote_features/'
        des_dir = '/esat/'+FLAGS.machine+'/tmp/control_output/'
    else:
        source_dir = '/esat/qayd/kkelchte/simulation/remote_features/'
        des_dir = '/esat/qayd/tmp/control_output/'
    # Restore a saved model fitting this configuration < model_dir
    g = tf.Graph()
    with g.as_default():
        if FLAGS.continuous:
            if FLAGS.short_labels:
                config.output = 2
            else:
                config.output = 6 #9 #4 ### TODO make this dynamic from model or input flags?
        else: #continuous case
            config.output = 41
        if FLAGS.network == 'stijn':
            config.feature_dimension = 4070 
        elif FLAGS.network == 'inception':
            config.feature_dimension = 2048
        elif FLAGS.network == 'no_cnn':
            config.feature_dimension = 360*640*3*0.2*0.2
        if FLAGS.normalized:
            #load normalization matrices:
            data = sio.loadmat('mean_variance_'+FLAGS.feature_type+"_"+FLAGS.network+'.mat')
            mean = data['mean']
            variance = data['variance']
            data = None
        if FLAGS.step_size_fnn > 1:
            print 'adjust feature feature_dimension'
            config.feature_dimension=FLAGS.step_size_fnn*config.feature_dimension
        device_name='/gpu:0'
        with tf.variable_scope(scope), tf.device(device_name):
            mtest = pilot_model.LSTMModel(False, config.output, config.feature_dimension, prefix='eval')
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        # Start session
        session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        saver.restore(session, os.path.join(FLAGS.model_dir, FLAGS.checkpoint_name))
        print "model restored", FLAGS.model_dir+"/"+FLAGS.checkpoint_name
        #time.sleep(25)
        location = logfolder+"/eval"
        #writer = tf.train.SummaryWriter(location, graph=session.graph)
        #get the shape of the inner state of the model
        if not FLAGS.fc_only:
            local_state = session.run(mtest.initial_state)
        
        #initial_state = np.zeros(prestate.shape)
        #assign zeros to the initial state of the model
        #session.run(tf.assign(mtest.initial_state, initial_state))
        frame=0
        ##initialize notifier
        # watch manager
        wm = pyinotify.WatchManager()
        wm.add_watch(source_dir, pyinotify.IN_CREATE)
        # event handler
        eh = MyEventHandler()
        # notifier working in the same thread
        notifier = pyinotify.Notifier(wm, eh, timeout=10)
        def on_loop(notifier):
            global current_feature
            global last_feature
            global frame
            global local_state
            global mtest
            global delay
            global session
            global g
            global feature_queue
            global mean
            global variance
            #import pdb; pdb.set_trace()
            if current_feature == last_feature:
                pass
                #print 'wait'
                #return
            else:
                print 'run through new feature: ',os.path.basename(current_feature),' last: ',os.path.basename(last_feature)
                try:
                    # data needs to be np array [batchsize 1, num_steps 1, feature_dimension]
                    if FLAGS.ssh: time.sleep(0.005) #wait till file is fully arrived on local harddrive when using ssh
                    if current_feature == source_dir+'change_network':
                        session.close()
                        f = open(current_feature)
                        FLAGS.model_name = f.read()[:-1]#skip the \n
                        FLAGS.model_dir = '/esat/qayd/kkelchte/tensorflow/lstm_logs/'+FLAGS.model_name
                        if FLAGS.model_name == 'dagger_big_net_stijn_depth' :
                            FLAGS.network = 'stijn'
                            config.feature_dimension = 4070
                        else :
                            config.feature_dimension = 2048
                        if FLAGS.model_name == 'dagger_hsz_400_fc':
                            FLAGS.fc_only = True
                            FLAGS.hidden_size=400
                        else :
                            FLAGS.fc_only = False
                            FLAGS.hidden_size=100
                        if FLAGS.model_name == 'straight' :
                            FLAGS.fly_straight = True
                            FLAGS.model_name = 'dagger_4G_wsize_300'
                            FLAGS.model_dir = '/esat/qayd/kkelchte/tensorflow/lstm_logs/'+FLAGS.model_name
                        else:
                            FLAGS.fly_straight = False
                        if FLAGS.model_name=='finished':
                            sys.exit(0)
                        g = None
                        mtest = None
                        g = tf.Graph()
                        saver = None
                        with g.as_default():
                            with tf.variable_scope(scope), tf.device('/gpu:0'):
                                mtest = pilot_model.LSTMModel(False, config.output, config.feature_dimension, prefix='eval')
                            saver = tf.train.Saver()
                            session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
                            saver.restore(session, FLAGS.model_dir+"/model.ckpt")
                            local_state = session.run(mtest.initial_state)
                            last_feature='%s' % current_feature
                            raise Exception('loaded new network: '+FLAGS.model_name)
                            #import pdb; pdb.set_trace()
                    # check if current features is a clear_memory message:
                    if current_feature == source_dir+'clear_memory' and not FLAGS.fc_only:
                        local_state = session.run(mtest.initial_state)
                        last_feature='%s' % current_feature
                        raise Exception('cleared inner state')
                    if current_feature == source_dir+'clear_memory' and FLAGS.fc_only:
                        last_feature='%s' % current_feature
                        feature_queue=[]
                        raise Exception('trajectory finished')
                    if FLAGS.network == "inception":
                        data=sio.loadmat(current_feature)
                        feature=data['features']
                    elif FLAGS.network == "stijn":
                        data=sio.loadmat(current_feature)
                        data=data['gazebo_sim_dataset']
                        feature=data[0,0]['labels']
                        feature = np.reshape(feature,(1,1,4070))
                    elif FLAGS.network == 'no_cnn':
                        feature=load_image(current_feature)
                    if FLAGS.normalized:
                        #print 'normalize feature: ',feature
                        epsilon = 0.001
                        feature=(feature-mean[0])/(np.sqrt(variance[0])+epsilon)
                        #print 'normalized feature: ',feature
                        
                    #if we have n-steps FC 
                    if FLAGS.step_size_fnn > 1:
                        #print 'append to feature queue'
                        feature_queue.append(feature)
                        if (len(feature_queue) < FLAGS.step_size_fnn):
                            print 'skip calculation because length is too small: ', len(feature_queue)
                            return;
                        else:
                            #print 'concat feature queue to feature'
                            feature = np.array([[np.concatenate(np.squeeze(feature_queue))]])
                    #import pdb; pdb.set_trace()
                    # Define evaluate 1step unrolled
                    if FLAGS.fc_only:
                        feed_dict = {mtest.inputs: feature}
                        outputs= session.run([mtest.logits],feed_dict)
                    else:
                        feed_dict = {mtest.inputs: feature, 
                            mtest.initial_state: local_state}
                        outputs, local_state= session.run([mtest.logits, mtest.state],feed_dict)
                    last_feature='%s' % current_feature # copy string object
                    #writer.add_summary(summary_str, f)
                    if FLAGS.continuous:
                        if FLAGS.short_labels:
                            #print 'prim output ',outputs[0]
                            if FLAGS.fc_only: output_str="0.8 0 "+str(outputs[0][0][0])+" 0 0 "+str(outputs[0][0][1])
                            else: output_str="0.8 0 "+str(outputs[0][0])+" 0 0 "+str(outputs[0][1])
                            #print 'output: ',output_str
                        else:
                            output_str = str(outputs[0])[1:-1] #get the brackets off
                            if FLAGS.fc_only:
                                output_str=re.sub('\n','',output_str)
                                output_str=output_str[2:-1]#get the brackets off
                    else:
                        #output_str = str(np.argmax(outputs))
                        if outputs.shape[1]%2 == 0:
                            raise SyntaxError('disc_factor is wrong: '+str((outputs.shape[1]+1)/2))
                        disc_factor=int((outputs.shape[1]+1)/2)
                        max_index = np.argmax(outputs)
                        #print 'argmax output: ', max_index,' disc_factor: ',disc_factor
                        output_str=convert_control_output.translate_index(max_index, disc_factor)
                    if FLAGS.step_size_fnn > 1:
                        feature_queue = feature_queue[1:]
                        #print 'remove first queue element: ', len(feature_queue)
                    #if FLAGS.fly_straight: 
                    #    output_str = "0.8 0 0 0 0 0"
                    print 'output: ', output_str
                    # Write output to control_output directory
                    fid = open(des_dir+'.tmp', 'w')
                    fid.write(output_str)
                    fid.close()
                    if FLAGS.ssh: time.sleep(0.001) # some time for saving the file
                    os.rename(des_dir+'.tmp', des_dir+str(frame)+'.txt')
                    frame=frame+1
                    #print "delay due to this feature: ", time.time()-delay
                except Exception as e:
                    print "[eval] skip feature due to complication: ",e
        notifier.loop(callback=on_loop)
    

class MyEventHandler(pyinotify.ProcessEvent):
    global current_feature
    global last_feature
    global delay
    #Object that handles the events posted by the notifier
    def process_IN_CREATE(self, event):
    #def process_IN_ACCESS(self, event):
        global current_feature
        global last_feature
        global delay
        current_feature = event.pathname
        print "received feature: ", current_feature
        #print "last feature: ", last_feature
        print "total delay from previous feature: ", time.time()-delay
        delay=time.time()    
        
def empty_folder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)
            
#########################################
def main(unused_argv=None):
    #FLAGS.model = 'small'
    #logfolder = pilot_settings.extract_logfolder()
    #print 'logfolder',logfolder
    #config = pilot_settings.get_config()
    
    #if FLAGS.model == "small":
        #window_sizes = [5] 
        #batch_sizes = [10]
        
    ##data_list
    #training_data_list = pilot_data.prepare_data_general(config.training_objects)
    
    #config.output=training_data_list[0][1].shape[2]
    #config.feature_dimension=training_data_list[0][0].shape[2]
    
    ##models
    #initializer = tf.random_uniform_initializer(-0.01, 0.01)
    #config.batch_size = batch_sizes[0]
    #config.num_steps = window_sizes[0]
    #trainingmodels = []
    #with tf.variable_scope("model", reuse=False, initializer=initializer) as model_scope:
        #mtrain = pilot_model.LSTMModel(True, config.output, config.feature_dimension, 
                                #config.batch_size, config.num_steps, 'train')
    #trainingmodels.append(mtrain)
    
    ##windowindices
    #window_indices = pilot_data.pick_random_windows(training_data_list, window_sizes, batch_sizes)
    
    ##mvalid
    #with tf.variable_scope(model_scope, reuse=True, initializer=initializer):
        #mvalid = pilot_model.LSTMModel(False, config.output, config.feature_dimension, prefix='eval')
    
    ##session
    #session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    #init = tf.initialize_all_variables()
    #session.run(init)
    #new_states = get_initial_states(trainingmodels, window_indices, mvalid, training_data_list, session)
    #old_states = get_initial_states_old(trainingmodels, window_indices, mvalid, training_data_list, session)
    #print str(old_states[0][0]-new_states[0][0])
    #import pdb; pdb.set_trace()
    #print 'finished!'
    #return

    FLAGS.model = 'remote'
    logfolder = pilot_settings.extract_logfolder()
    #print 'logfolder',logfolder
    config = pilot_settings.get_config()
    FLAGS.model_dir = '/esat/qayd/kkelchte/tensorflow/lstm_logs/'+FLAGS.model_name
    if os.path.isfile(os.path.join(FLAGS.model_dir, 'configuration.xml')):
        print 'parsing configuration file in ',FLAGS.model_dir
        tree=ET.parse(os.path.join(FLAGS.model_dir, 'configuration.xml'))
        root=tree.getroot()
        flgs=root.find('flags')
        for flg in flgs:
            if flg.tag != 'model_dir' and flg.tag != 'model_name' and flg.tag != 'online':
                value = ''
                try:
                    value = int(flg.text)
                except ValueError:
                    try:
                        value = float(flg.text)
                    except ValueError:
                        if flg.text == 'False': value = False
                        elif flg.text == 'True': value = True
                        else :  value = flg.text
                FLAGS.__dict__['__flags'][flg.tag] = value
                #print flg.tag, ": ",value," of type ",type(value)
    else:
        if FLAGS.fc_only:
            FLAGS.hidden_size = 400
        if FLAGS.network == 'stijn':
            FLAGS.remote_features = 'depth_estimate'
    if FLAGS.online:
        source_dir = '/esat/'+FLAGS.machine+'/tmp/remote_features/'
        empty_folder(source_dir)
        FLAGS.ssh = True
        evaluate_online(logfolder, config)
    else:
        evaluate(logfolder, config)
    print 'done'

if __name__ == '__main__':
    tf.app.run()
    
