#This code groups the functions for evaluating a model by unrolling it one step
#and predicting.
#
import pilot_model
import pilot_data
import pilot_settings
import convert_control_output

import time
import tensorflow as tf
import numpy as np
import scipy.io as sio

import sys
import pyinotify

import os, shutil
 
import matplotlib.pyplot as plt


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


#global variable feature bucket is a buffer used by the eventhandler to collect new arrived features as a list of paths
current_feature = "" # feature now in queue ready to be processed
last_feature = "" # feature just processed
frame = 0
local_state = None
mtest = None
delay=0

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
        if not FLAGS.continuous: print "Accuracy: ",(score/iters)
        else: print "Average Loss: ",(costs/iters)   
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
    initial_states_all = []
    for im in range(len(models)):#index for model
        model = models[im]
        zerostate = sess.run(model.initial_state)
        #print 'zerostate: ',zerostate
        initial_states_local = [] #list for each time a batch-window is applied in one epoch
        for il in range(len(indices[im])):
        	initial_states = np.zeros(zerostate.shape)
        	#print 'initial_states: ',initial_states
        	for ib in range(indices[im][il].shape[0]):#index of movie in batch
        		#import pdb; pdb.set_trace()
        	    [tup_ind, mov_ind, start_ind] = indices[im][il][ib] #get indices needed for finding data
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
        with tf.Session() as session:
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
            if not FLAGS.continuous:
                accuracies = [mres[0] for mres in test_results]
                losses = [mres[1] for mres in test_results]
                print("Test: Average Accuarcy over different unrolled models: %f, Max: %f, Min: %f, Loss: %f." % (sum(accuracies)/len(accuracies), max(accuracies), min(accuracies), sum(losses)/len(losses)))
            else:#continuous case
                losses = [mres[1] for mres in test_results]
                print("Test: Average loss over different unrolled models: %f, Max: %f, Min: %f." % (sum(losses)/len(losses),max(losses),min(losses)))
                
            #print 'average accuracy: ',acc
            #print 'max accuracy: ',acc_max
            #print 'min accuracy: ',acc_min
            #print 'average perplexity: ',per
    print 'evaluation...finished!'

def evaluate_online(logfolder, config, scope="model"):
    global current_feature
    global last_feature
    global frame
    global local_state
    global mtest
    global delay
    
    if FLAGS.ssh:
        source_dir = '/esat/'+FLAGS.machine+'/tmp/remote_features/'
        #des_dir = '/esat/'+FLAGS.machine+'/tmp/control_output/'
        des_dir = '/esat/'+FLAGS.machine+'/tmp/control_output/'
    else:
        source_dir = '/esat/qayd/kkelchte/simulation/remote_features/'
        des_dir = '/esat/qayd/tmp/control_output/'
    """This function is called from the main thread and coordinates the one step evaluation.
    Args:
        logfolder: the log folder is extracted from all the different user configurations
        config: the configurations are grouped in a configuration object    
    """
    # Restore a saved model fitting this configuration < model_dir
    with tf.Graph().as_default():
        if FLAGS.continuous:
            config.output = 6 #9 #4 ### TODO make this dynamic from model or input flags?
        if FLAGS.network == 'stijn':
            config.feature_dimension = 4070 
        else:
            config.feature_dimension = 2048
        device_name='/gpu:0'
        with tf.variable_scope(scope), tf.device(device_name):
            mtest = pilot_model.LSTMModel(False, config.output, config.feature_dimension, prefix='eval')
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        # Start session
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session, tf.device(device_name):
            saver.restore(session, FLAGS.model_dir+"/model.ckpt")
            print "model restored", FLAGS.model_dir+"/model.ckpt"
            #time.sleep(25)
            location = logfolder+"/eval"
            writer = tf.train.SummaryWriter(location, graph=session.graph)
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
                #import pdb; pdb.set_trace()
                if current_feature == last_feature:
                    print 'wait'
                    #return
                else:
                    print 'run through new feature: ',frame
                    try:
                        # data needs to be np array [batchsize 1, num_steps 1, feature_dimension]
                        if FLAGS.ssh: time.sleep(0.005) #wait till file is fully arrived on local harddrive when using ssh
                        data=sio.loadmat(current_feature)
                        if FLAGS.network == "inception":
                            feature=data['features']
                        elif FLAGS.network == "stijn":
                            data=data['gazebo_sim_dataset']
                            feature=data[0,0]['labels']
                            feature = np.reshape(feature,(1,1,4070))
                        #import pdb;pdb.set_trace()
                        #plt.matshow(np.transpose(np.reshape(feature, (74,55))))
                        #plt.show()
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
                        #import pdb; pdb.set_trace()
                        if FLAGS.continuous:
                            output_str = str(outputs[0])[1:-1] #get the brackets off
                            if FLAGS.fc_only:#Still needs to be tested!!
                                output_str=re.sub('\n','',output_str)
                                output_str=output_str[1:-1]
                        else:
                            #output_str = str(np.argmax(outputs))
                            if outputs.shape[0]%2 == 0:
                                raise SyntaxError('disc_factor is wrong: '+str((outputs.shape[0]+1)/2))
                            disc_factor=int((outputs.shape[0]+1)/2)
                            max_index = np.argmax(outputs)
                            output_str=convert_control_output.translate_index(max_index, disc_factor)
                        print 'output: ', output_str
                        #import pdb;pdb.set_trace()
                        
                        # Write output to control_output directory
                        #fid = open('/esat/qayd/kkelchte/tensorflow/control_output/'+str(frame)+'.txt', 'w')
                        fid = open(des_dir+str(frame)+'.txt', 'w')
                        fid.write(output_str)
                        fid.close()
                        if FLAGS.ssh: time.sleep(0.002) # some time for saving the file
                        frame=frame+1
                        print "delay due to this feature: ", time.time()-delay
                    except Exception as e:
                        print "[eval] skip feature due to error: ",e
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
        print "last feature: ", last_feature
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
    FLAGS.model = 'remote'
    logfolder = pilot_settings.extract_logfolder()
    print 'logfolder',logfolder
    config = pilot_settings.get_config()
    
    if FLAGS.online:
        source_dir = '/esat/'+FLAGS.machine+'/tmp/remote_features/'
        empty_folder(source_dir)
        FLAGS.continuous = True
        FLAGS.ssh = True
        #FLAGS.model_dir = '/esat/qayd/kkelchte/tensorflow/lstm_logs/remote_set5_windowwise_sz_100_net_inception'
        #FLAGS.model_dir = '/esat/qayd/kkelchte/tensorflow/lstm_logs/remote_set7_sz_100_net_inception'
        FLAGS.model_dir = '/esat/qayd/kkelchte/tensorflow/lstm_logs/'+FLAGS.model_name
        evaluate_online(logfolder, config)
    else:
        evaluate(logfolder, config)
    print 'done'

if __name__ == '__main__':
    tf.app.run()
    
