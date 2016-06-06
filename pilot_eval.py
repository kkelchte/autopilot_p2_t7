"""
This code groups the functions for evaluating a model by unrolling it one step
and predicting.
"""

import pilot_model
import pilot_data
import pilot_settings

import time
import tensorflow as tf
import numpy as np
import scipy.io as sio
 
FLAGS = tf.app.flags.FLAGS
 
def run_epoch_one_step(session, model, data, targets, eval_op, num_steps=1, writer=None, verbose=False, number_of_samples=500, matfile = ""):
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
    
    #define what frames you want to write
    frames_to_write=[]
    if(writer != None):
        write_freq = int((data.shape[1]-num_steps+1)/number_of_samples)
        if write_freq != 0: frames_to_write = [f for f in range(data.shape[1]-num_steps+1) if (f % write_freq) == 0]
        #if frames - steps > 100 ==> dont do modulo of zero
        #in case frames - steps < 100 ==> save every frame because there are less than 100 steps
        #the summary writer will only record the last <100 steps
        else: frames_to_write = range(data.shape[1]-num_steps+1)
    
    for f in range(data.shape[1]-num_steps+1):
        feed_dict = {model.inputs: data[:,f:f+num_steps,:], 
                     model.initial_state: state, 
                     model.targets: targets[:,f:f+num_steps,:]}
        if f in frames_to_write: #write data away from time to time
            outputs, state, current_loss, _, summary_str= session.run([model.logits, model.state, model.cost, eval_op, model.merge()],feed_dict)
            writer.add_summary(summary_str, f)
        else:
            # outputs is of shape (num_steps*batchsize, outputsize)
            outputs, state, current_loss, _= session.run([model.logits, model.state, model.cost, eval_op],feed_dict)
        iters += data.shape[0] #add these batches as they all where handled
        list_of_states.append(state)
        
	for m in range(data.shape[0]): #loop over the movies of the batch (in general will be one)
            if np.argmax(outputs[m*num_steps]) == np.argmax(targets[m,f,:]):#see if this frame is calculated correctly
                score += 1.0 #if your score is correct count it up
            if verbose and (f % int(data.shape[1]/15) == 0):
                print("Frame: %d batch: %d target: %d prediction: %d speed: %.3f fps"\
                    %(f, m, np.argmax(targets[m,f,:]), np.argmax(outputs[m*num_steps]),
                    iters * data.shape[0] / (time.time() - start_time)))    
        costs += current_loss
    
    # Keep track of cell states saved in the list_of_states
    if(writer != None) and (matfile != ""):
        #states = [batchsize, num_steps*hidden_size*num_layers*2 (output;state)]
        states = np.concatenate(list_of_states)
        trgts = targets.reshape((-1,outputs.shape[1]))
        d = {'states': states, 'targets': targets}
        sio.savemat(matfile,d, appendmat=True)
        
    if verbose:
        print "Accuracy: ",(score/iters)
        print "Average Loss: ",(costs/iters)   
    ##Add call for tensor accuracy + write it away
    return score, costs, iters

def evaluate(logfolder, config, scope="model"):
    """Get some evaluation data and test the performance
    of a restored model. The model is chosen according to the flags set by the user.
    """
    with tf.Graph().as_default():
        test_data_list = pilot_data.prepare_data_list(config.test_objects)
        config.output=test_data_list[0][1].shape[2]
        config.feature_dimension=test_data_list[0][0].shape[2]
        #set params according to the shape of the obtained data
        with tf.variable_scope(scope):
            mtest = pilot_model.LSTMModel(False, config.output, config.feature_dimension, prefix='eval')
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        with tf.Session() as session:
            saver.restore(session, logfolder+"/model.ckpt")
            print "model restored", logfolder+"/model.ckpt"
            location = logfolder+"/eval"
            writer = tf.train.SummaryWriter(location, graph=session.graph)
            #decide of what movies you want to keep track.
            movies_to_write_away = [0,1,2,3]
            acc = 0.0
            per = 0.0
            acc_max = 0.0
            acc_min = 1.0
            for j in range(len(test_data_list)):
                if j in movies_to_write_away: wrtr = writer
                else: wrtr = None
                results = run_epoch_one_step(session, 
                                            mtest, 
                                            test_data_list[j][0], 
                                            test_data_list[j][1], 
                                            tf.no_op(),
                                            writer=wrtr,
                                            verbose=True)
                perplexity = np.exp(results[1] / results[2])
                accuracy = results[0]/results[2]
                print("Accuracy: %.3f" % (accuracy))
                if acc_max < accuracy: acc_max = accuracy
                if acc_min > accuracy: acc_min = accuracy
                acc = acc+accuracy/len(test_data_list)
                per = per+perplexity/len(test_data_list)
                print("Test perplexity: %.3f" %perplexity)
            print 'average accuracy: ',acc
            print 'max accuracy: ',acc_max
            print 'min accuracy: ',acc_min
            print 'average perplexity: ',per
    print 'evaluation...finished!'

#########################################
if __name__ == '__main__':
    
    logfolder = pilot_settings.extract_logfolder()
    print 'logfolder ',logfolder
    config = pilot_settings.get_config()
    scope = "model"
    evaluate(logfolder, config, scope)
