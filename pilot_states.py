import numpy.random as nr
import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
# avoid images to appear...
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import scipy.io as sio
import tensorflow as tf
import time

def fig2buf(fig):
    """
    Convert a plt fig to a numpy buffer
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = (h, w, 4)
    
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    buf_rgb = buf[:,:,0:3]
    return buf_rgb
    

def get_image(states, trgets, num_steps, hsize, layers, batch_size):
    """
    Get the current states at this step and according to these targets
    and fit them in a nice image with good labels.
    Args:
        step: current step of which states are represented
        states: np matrix [batchsize, num_steps*hidden_size*num_layers*2 (~output;state)]
        targets: labels
    Returns:
        image: 4D tensor
    """
    #states = tf.placeholder(tf.float32, [batch_size, num_steps*hsize*layers*2])
    #targets = tf.placeholder(tf.float32, [batch_size, num_steps*hsize*layers*2])
    
    buf_rgb_list = []
    
    states = states[0] #use only first movie of batch
    targets = np.argmax(trgets[0], axis=1) #get those labels 
    width = 0.9/layers 
    
    #Arrange output and states in vector with following dimensions: [layers, hsize, num_steps]
    data_states = np.zeros((layers, hsize, num_steps))
    data_outputs = np.zeros((layers, hsize, num_steps))
    #loop over steps to rearrange the data
    for step in range(num_steps):
        #data related to 1 step
        step_data = states[step*layers*hsize*2 : (step+1)*layers*hsize*2]
        for layer in range(layers):
            #data of this step related to current layer
            layer_data = step_data[layer*hsize*2:(layer+1)*hsize*2]
            state = layer_data[0:hsize]
            output = layer_data[hsize:]
            data_states[layer,:,step] = state
            data_outputs[layer,:,step] = output
    
    #list of colors for different layers
    clrs = ['r','y','b','g','c','m','k']
        
    # Loop over different time steps: make 15 images spread over full length
    # start with first frame
    step = int(num_steps/9) 
    if step == 0: step = 1

    #plt.bar(np.arange(hsize), data_states[l,:,i], width, color=clrs[l])
    #plt.show()
    for i in range(0,num_steps, step):
        ##subplot for state and output
        fig, ax = plt.subplots(2, sharex=True, sharey=False)
        title='Time: '+str(i)+'. GT: '+str(targets[i])
        ax[0].set_title(title)
        title='Up: innerstates. Down: Outputs. Layers: '
        for l in range(layers): title = title+' "'+clrs[l]+'"'
        ax[1].set_title(title)
        for l in range(layers):
            ax[0].bar(np.arange(hsize)+l*width, data_states[l,:,i], width, color=clrs[l])
            ax[1].bar(np.arange(hsize)+l*width, data_outputs[l,:,i], width, color=clrs[l])
        #plt.show()
        buf_rgb = fig2buf(fig)
        buf_rgb_list.append(buf_rgb)
        fig.clf()
    buf_rgb_list = np.asarray(buf_rgb_list)
    
    return buf_rgb_list
    
##################################################################3    
if __name__ == '__main__':
    print "-----------------------------------"
    print "Local run of pilot_states for testing"
    print "-----------------------------------"
    
    logfolder='/esat/qayd/kkelchte/tensorflow/lstm_logs/states'
    location = logfolder+"/overview"
    
    with tf.Graph().as_default(), tf.Session() as sess:
        batch_size = 2
        num_steps = 1000
        hidden_size = 50
        num_layers = 2
        
        states = nr.rand(batch_size, num_steps*hidden_size*num_layers*2)*100-50
        targets = nr.rand(batch_size, num_steps, 4)
        writer = tf.train.SummaryWriter(location, graph=sess.graph)         
        
        state_images = get_image(states, targets, num_steps, hidden_size, num_layers, batch_size)
        #import pdb; pdb.set_trace()
        
        im_op = tf.image_summary('Innerstates', state_images, max_images=15)
        summary_str = sess.run(tf.merge_summary([im_op]))
        writer.add_summary(summary_str)
