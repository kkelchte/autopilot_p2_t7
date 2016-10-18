"""
Build the pilot LSTM network
"""

import tensorflow as tf
import numpy as np
import math

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    "optimizer", "Adam",
    "Define the wanted optimizer for training.")
tf.app.flags.DEFINE_integer(
    "num_layers", "2",
    "Define the num_layers of the model.")
tf.app.flags.DEFINE_integer(
    "hidden_size", "100",
    "Define the hidden_size of the model.")
tf.app.flags.DEFINE_float(
    "keep_prob", "1.0",
    "Define the keep probability for training.")
tf.app.flags.DEFINE_float(
    "forget_bias", "0.9",
    "Reducing the forget bias makes the model remember longer.")
tf.app.flags.DEFINE_boolean(
    "gpu", True,
    "Define whether you want to work with GPU or not.")
tf.app.flags.DEFINE_boolean(
    "fc_only", False,
    "Define whether the network contains only a fc layer or also an LSTM")
tf.app.flags.DEFINE_integer(
    "step_size_fnn", 1,
    "set the amount frames are concatenated to form a feature of a frame."
    )
tf.app.flags.DEFINE_boolean(
    "conv_layers", False,
    "Include convolutional layers before FC or LSTM.")


class ModelError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class LSTMModel(object):

    def __init__(self, is_training, output_size, input_dimension, batch_size=1, num_steps=1, prefix=''):
        '''initialize the model
        Args:
            is_training = boolean that adds optimizer operation
            output_size, input_dimension, batch_size, num_steps = defines what kind of input the model can get_variable
            prefix = train/validate/test
        '''
        self._is_training=is_training
        
        # Define in what type input data arrives
        self._batch_size = batch_size
        if not FLAGS.fc_only:
            self._num_steps = num_steps
        else :#work with 1step in fully connected feedforward network
            self._num_steps = 1
            
        self._prefix = prefix
        
        # Define model
        self._hidden_size = FLAGS.hidden_size
        self._num_layers = FLAGS.num_layers
        
        # Build LSTM graph
        self._logits = self.inference(FLAGS.num_layers, FLAGS.hidden_size, output_size, input_dimension, FLAGS.gpu, FLAGS.keep_prob)
        self._cost = self.loss(output_size, FLAGS.gpu)
        
        if self._is_training:
            self.training(FLAGS.gpu, optimizer_type=FLAGS.optimizer)
    
    
    def inference(self, num_layers, hidden_size, output_size, input_dimension, gpu, keep_prob=1.0):
        """Build the LSTM model up to where it may be used for inference.

        Args:
            input_ph: Feature/image placeholder, from inputs().
            num_layers: Number of LSTM layers.
            hidden_size: Number of hidden units
            output_size: Dimension of the output tensor. (4 states)
            
        Returns:
            softmax_linear: Output tensor with the computed logits.
        """
        print "inference ",self.prefix,": batch ", self._batch_size," hiddensize ",hidden_size," keep_prob ",keep_prob," layers ",num_layers," ",num_layers
        
        feature_dimension=1024
        
        device_name='/cpu:0'
        if gpu:
            device_name='/gpu:0'
        #print "device name: ", device_name
        # Placeholder for input 
        with tf.device(device_name):
            self._inputs = tf.placeholder(tf.float32, [self._batch_size, self._num_steps, input_dimension], name=self._prefix+"_input_ph")
        if not FLAGS.conv_layers:
            self._feature_inputs=self.inputs
            feature_dimension=input_dimension
        else:
            if input_dimension != 72*128*3:
                raise ValueError("[pilot_model] input dimension to convolutional network is not 72*128*3: "+input_dimension)
            # A few definitions in for readability
            def weight_variable(shape):
                initial = tf.truncated_normal(shape, stddev=0.1)
                return tf.Variable(initial)
            def bias_variable(shape):
                initial = tf.constant(0.1, shape=shape)
                return tf.Variable(initial)
            # Basic padding so size stays the same
            def conv2d(x, W):
                return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
            # Basic 2x2 max pooling
            def max_pool_2x2(x):
                return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
            
            """Add convolutional layers in order to lower the high dimensional input. 
            Extracting features before feeding them to the FC layers or the LSTM layer.
            """
            #x = tf.placeholder(tf.float32, shape=[None, 27648]) # Define shape of placeholder,
            # First layer: calculate 12 features of each 5x5 patch
            W_conv1 = weight_variable([5, 5, 3, 12]) # initialize the weights
            b_conv1 = bias_variable([12])
            #reshape image x to 4d tensor
            x_image = tf.reshape(self._inputs, [-1,72,128,3])
            #convolve 
            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)

            # Second layer: calculate 24 features of each 5x5 patch
            W_conv2 = weight_variable([5, 5, 12, 24]) # initialize the weights
            b_conv2 = bias_variable([24])
            #convolve 
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = max_pool_2x2(h_conv2)

            # Third layer: calculate 24 features of each 5x5 patch
            W_conv3 = weight_variable([5, 5, 24, 48]) # initialize the weights
            b_conv3 = bias_variable([48])
            #convolve 
            h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
            h_pool3 = max_pool_2x2(h_conv3)

            # Final layer: calculate 24 features of each 5x5 patch
            W_conv4 = weight_variable([3, 3, 48, 64]) # initialize the weights
            b_conv4 = bias_variable([64])
            #convolve 
            h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
            h_pool4 = max_pool_2x2(h_conv4)

            # Fully connected layers...
            W_fc1 = weight_variable([5 * 8 * 64, feature_dimension])
            b_fc1 = bias_variable([feature_dimension])
            h_pool4_flat = tf.reshape(h_pool4, [-1, 5 * 8 * 64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)
            
            # some deforming to ensure feature_input is in correct shape
            self._feature_inputs=tf.reshape(h_fc1, [self._batch_size, self._num_steps, feature_dimension])
        
        if not FLAGS.fc_only:
            with tf.variable_scope("LSTM"):
                with tf.device(device_name):
                    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=FLAGS.forget_bias) 
                    #'good' results: 0.9. Init: 0. add forget_bias in order to reduce scale of forgetting 
                    #==> remember longer initially: f = f'(0.1) + f_b(0.9) = 1.0 ==> keep all? ==> c = f * c_-1 . Default value is 1.
                    if self.is_training and keep_prob < 1:
                        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
                    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * int(num_layers))
                    #[batch_size, 1] with value 0
                    # the zero state is a tensor that calls the cell to represent a zero state vector of length [batchsize, num_layers*hidden_size*2]
                    self._initial_state = cell.zero_state(self._batch_size, tf.float32)
                    
                    #self._zero_state = cell.zero_state(self._batch_size, tf.float32)
                    #print 'self._batch_size: ',self._batch_size
                    
                    # the initial_state should be a variable that can be set
                    # during stepwise unrolling in order to contain the necessary
                    # inner state
                    #self._initial_state = tf.placeholder(tf.float32, [self._batch_size,hidden_size*num_layers*2], name=self._prfix +"_initial_state")
                    
                    #self._initial_state = tf.get_variable(self._prefix+"_initial_state",[self._batch_size,hidden_size*num_layers*2], trainable=False)
                    
                    #if self.is_training and keep_prob < 1:
                        #self._feature_inputs = tf.nn.dropout(self.inputs, keep_prob)
                    
                    #unfolled a greater network over num_steps timesteps concatenating the output
                    outputs = []
                    state = self.initial_state
                    states = []
                    for time_step in range(self._num_steps):
                        if time_step > 0: tf.get_variable_scope().reuse_variables()
                        if time_step == 1: self._state = state # the next run will need the state after the first step to continue
                        (cell_output, state) = cell(self._feature_inputs[:, time_step, :], state)
                        # what is the size of this state? [batchsize * layers * hidden_size *2 (for output and cell state)]
                        outputs.append(cell_output)
                        states.append(state)
                    if self._num_steps == 1: self._state = state # if only 1 step is taken, the state is not saved yet.
                    # make it a nice tensor of shape [batch_size, num_steps*hidden_size]
                    # reshape so each row correspond to 1 output of hidden_size length
                    # [b0t0, b0t1, ..., b0tn, b1t0, b1,t1, ... b1tn, ... bst0, ...] with n num steps and s batchsize
                    self._output = tf.reshape(tf.concat(1, outputs), [-1, hidden_size])
                    self._states = tf.concat(1,states)
                    #import pdb; pdb.set_trace()
                
                #collect the data
                self.cell_output_hist = tf.histogram_summary(self._prefix+"_cell_output", cell_output)
                self.state_hist = tf.histogram_summary(self._prefix+"_state", state)
            
        else:
            with tf.variable_scope("hidden1"):
                with tf.device(device_name):
                    weights = tf.get_variable("weights", [feature_dimension, hidden_size])
                    biases = tf.get_variable('biases', [hidden_size])
                    hidden1 = tf.nn.relu(tf.matmul(self._feature_inputs[:,0,:], weights) + biases)
            with tf.variable_scope("hidden2"):
                with tf.device(device_name):
                    weights = tf.get_variable("weights", [hidden_size, hidden_size])
                    biases = tf.get_variable('biases', [hidden_size])
                    self._output = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
        # Add a softmax layer mapping the output to 4 logits defining the network
        with tf.name_scope(self._prefix+'_softmax_linear'):
            with tf.device(device_name):
                weights = tf.get_variable("weights",[hidden_size, output_size])        
                biases = tf.get_variable('biases', [output_size])
                # [?, ouput_size]=[?, hidden_size]*[hidden_size, output_size]
                # with ? = num_steps * batch_size (b0t0, b0t1, ..., b1t0, b1t1, ...)
                logits = tf.matmul(self._output, weights) + biases
            
            # Add summary ops to collect data
            self.w_hist = tf.histogram_summary(self._prefix+"_weights", weights)
            self.b_hist = tf.histogram_summary(self._prefix+"_biases", biases) 
            self.logits_hist = tf.histogram_summary(self._prefix+"_logits", logits)
            
            #write away the last prediction of a series of steps
            preds = tf.argmax(logits,1)
            self.preds_val = [] #list of predictions of each batch
            for i in xrange(self._num_steps-1, self._batch_size*self._num_steps, self._num_steps):
                # for each num_steps period you should keep the last prediction
                self.preds_val.append(tf.scalar_summary(self._prefix+"_predictions_b"+str((i-self._num_steps+1)/self._num_steps), preds[i]))
        return logits
    
    
    def loss(self, output_size, gpu):
        """ Ops required to generate loss.
        """
        device_name='/cpu:0'
        if gpu:
            device_name='/gpu:0'
        
        # Necessary placeholders for input
        with tf.device(device_name):
            self._targets = tf.placeholder(tf.float32, [self._batch_size, self._num_steps, output_size], name=self._prefix+"_targets_ph")
        # Loss learning layer
        with tf.name_scope(self._prefix+'loss'):
            with tf.device(device_name):
                targs = tf.argmax(tf.reshape(self.targets,[self.batch_size*self._num_steps, output_size]), 1)
                if not FLAGS.continuous:
                    #cross_entropy = -SUM t(x)*ln(l(x))
                    if FLAGS.fc_only:
                        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.logits, tf.reshape(self.targets,[self.batch_size*self._num_steps, output_size]), name=self._prefix+'_xentropy')
                        loss = tf.reduce_mean(cross_entropy, name=self._prefix+'_xentropy_mean')
                    else:
                        loss = tf.nn.seq2seq.sequence_loss_by_example(
                            [self.logits],
                            [targs],
                            [tf.ones([self.batch_size*self.num_steps])]) #weights
                        loss = tf.reduce_sum(loss)/self.batch_size
                else: #continuous
                	#needs to be debugged!
                    #loss = tf.reduce_sum(tf.square(tf.sub(tf.reshape(self.targets,[self.batch_size*self._num_steps, output_size]), self.logits)))
                    #self._rsh_lgts = tf.reshape(self.logits, [self._batch_size, self._num_steps, output_size])
                    #self._loss_wild = tf.square(tf.sub(self.targets, self.rsh_lgts))
                    #self._loss_wild = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(self.targets, tf.reshape(self.logits, [self._batch_size, self._num_steps, output_size]))),reduction_indices=2))
                    loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.sub(self.targets, tf.reshape(self.logits, [self._batch_size, self._num_steps, output_size]))),reduction_indices=2)))
            
            
            #add scalar for collecting data
            self.loss_val = tf.scalar_summary(self._prefix+"_loss", loss)
            #targs = tf.argmax(tf.reshape(self.targets,[batch_size*self._num_steps, output_size]), 1)
            self.targs_val=[]
            for i in xrange(self._num_steps-1, self.batch_size*self._num_steps,self._num_steps):
                self.targs_val.append(tf.scalar_summary(self._prefix+"_targets_b"+str((i-self._num_steps+1)/self._num_steps), targs[i]))
        return loss
    
    def training(self, gpu, optimizer_type='GradientDescent'):
        """
        ops required to compute and apply gradients
        """
        device_name='/cpu:0'
        if gpu:
            device_name='/gpu:0'
            
        with tf.device(device_name):
            self._lr = tf.Variable(0.0, trainable=False, name=self._prefix+"_learning_rate")
        
        with tf.name_scope(self._prefix+'_training'):
            with tf.device(device_name):
                tvars = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars), 5)# max_grad_norm = 5
                # optimizer will apply calculated gradients at a certain training rate
                if optimizer_type == 'RMSProp': 
                    optimizer = tf.train.RMSPropOptimizer(self._lr, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False, name='RMSProp')
                elif optimizer_type == 'GradientDescent':
                    optimizer = tf.train.GradientDescentOptimizer(self._lr)
                elif optimizer_type == 'Momentum':
                    optimizer = tf.train.MomentumOptimizer(self._lr, momentum=0.3, use_locking=False, name='Momentum')
                elif optimizer_type == 'Adam':
                    #beta1 and beta2 = exponential decay rates for the moment estimates mean and variance of the gradients
                    #higher decay rate means slower adaption of new gradients
                    optimizer = tf.train.AdamOptimizer(learning_rate=self._lr, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
                elif optimizer_type == 'Adagrad':
                    optimizer = tf.train.AdagradOptimizer(self._lr, initial_accumulator_value=0.1, use_locking=False, name='Adagrad')
                else:
                    raise ModelError("This type of optimizer is not defined: "+optimizer_type)
                    return -1
                self._train_op = optimizer.apply_gradients(zip(grads, tvars))
                
        
    def merge(self):
        """ Specify which values are written down in the logfile and visualized on the tensorboard
        for the case the network is unrolled for just a number of steps (return the last values)
        """
        if FLAGS.fc_only:
            list_of_tensors = self.preds_val+self.targs_val
        else:
            list_of_tensors = [ self.cell_output_hist, 
                            self.state_hist, 
                            self.w_hist,
                            self.b_hist, 
                            self.logits_hist]+self.preds_val+self.targs_val 
        return tf.merge_summary(list_of_tensors)

    
    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    @property
    def lr(self):
        return self._lr
    
    @property
    def prefix(self):
        return self._prefix
    
    @property
    def is_training(self):
        return self._is_training
    
    @property
    def hidden_size(self):
        return self._hidden_size
    
    @property
    def num_layers(self):
        return self._num_layers
        
    @property
    def logits(self):
        return self._logits
    
    @property
    def inputs(self):
        return self._inputs
    
    @property
    def feature_inputs(self):
        return self._feature_inputs
    
    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state
    
    @property
    def zero_state(self):
        return self._zero_state
    
    @property
    def cost(self):
        return self._cost
        
    @property
    def state(self):
        return self._state
    
    @property
    def states(self):
        return self._states
        
    @property
    def output(self):
        return self._output
    @property
    def loss_wild(self):
        return self._loss_wild
    @property
    def rsh_lgts(self):
        return self._rsh_lgts
    @property
    def train_op(self):
        return self._train_op
    
    #number of steps the network unrolls
    @property
    def num_steps(self):
        return self._num_steps
    #the number of movies the model can evaluate in 1 time
    #movies have to be the same length
    @property
    def batch_size(self):
        return self._batch_size
    


