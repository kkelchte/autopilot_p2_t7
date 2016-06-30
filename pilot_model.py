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
    "hidden_size", "50",
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

class ModelError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class LSTMModel(object):

    def __init__(self, is_training, output_size, feature_dimension, batch_size=1, num_steps=1, prefix=''):
        '''initialize the model
        Args:
            is_training = boolean that adds optimizer operation
            output_size, feature_dimension, batch_size, num_steps = defines what kind of input the model can get_variable
            prefix = train/validate/test
        '''
        self._is_training=is_training
        
        # Define in what type input data arrives
        self._batch_size = batch_size
        self._num_steps = num_steps
        self._prefix = prefix
        
        # Define model
        self._hidden_size = FLAGS.hidden_size
        self._num_layers = FLAGS.num_layers
        
        # Build LSTM graph
        self._logits = self.inference(FLAGS.num_layers, FLAGS.hidden_size, output_size, feature_dimension, FLAGS.gpu, FLAGS.keep_prob)
        self._cost = self.loss(output_size, FLAGS.gpu)
        
        if self._is_training:
            self.training(FLAGS.gpu, optimizer_type=FLAGS.optimizer)
    
    
    def inference(self, num_layers, hidden_size, output_size, feature_dimension, gpu, keep_prob=1.0):
        """Build the LSTM model up to where it may be used for inference.

        Args:
            input_ph: Feature/image placeholder, from inputs().
            num_layers: Number of LSTM layers.
            hidden_size: Number of hidden units
            output_size: Dimension of the output tensor. (4 states)
            
        Returns:
            softmax_linear: Output tensor with the computed logits.
        """
                
        device_name='/cpu:0'
        if gpu:
            device_name='/gpu:0'
        # Placeholder for input 
        with tf.device(device_name):
            self._inputs = tf.placeholder(tf.float32, [self._batch_size, self._num_steps, feature_dimension], name=self._prefix+"_input_ph")
            
        print "inference ",self.prefix,": batch ", self._batch_size," size ",hidden_size," keep_prob ",keep_prob," layers ",num_layers," ",num_layers
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
                
                if self.is_training and keep_prob < 1:
                    self._inputs = tf.nn.dropout(self.inputs, keep_prob)
                
                #unfolled a greater network over num_steps timesteps concatenating the output
                outputs = []
                state = self.initial_state
                states = []
                for time_step in range(self._num_steps):
                    if time_step > 0: tf.get_variable_scope().reuse_variables()
                    if time_step == 1: self._state = state # the next run will need the state after the first step to continue
                    (cell_output, state) = cell(self._inputs[:, time_step, :], state)
                    # what is the size of this state? [batchsize * layers * hidden_size *2 (for output and cell state)]
                    outputs.append(cell_output)
                    states.append(state)
                if self._num_steps == 1: self._state = state # if only 1 step is taken, the state is not saved yet.
                # make it a nice tensor of shape [batch_size, num_steps*hidden_size]
                # reshape so each row correspond to 1 output of hidden_size length
                output = tf.reshape(tf.concat(1, outputs), [-1, hidden_size])
                self._states = tf.concat(1,states)
                #import pdb; pdb.set_trace()
            
            #collect the data
            self.cell_output_hist = tf.histogram_summary(self._prefix+"_cell_output", cell_output)
            self.state_hist = tf.histogram_summary(self._prefix+"_state", state)
        
            
        # Add a softmax layer mapping the output to 4 logits defining the network
        with tf.name_scope(self._prefix+'_softmax_linear'):
            with tf.device(device_name):
                weights = tf.get_variable("weights",[hidden_size, output_size])        
                biases = tf.get_variable('biases', [output_size])
                # [?, output_size]=[?, hidden_size]*[hidden_size, output_size]
                # with ? = num_steps * batch_size
                logits = tf.matmul(output, weights) + biases
            
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
                #cross_entropy = -SUM t(x)*ln(l(x))
                #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.logits, tf.reshape(self.targets,[batch_size*self._num_steps, output_size]), name=self._prefix+'_xentropy')
                #loss = tf.reduce_mean(cross_entropy, name=self._prefix+'_xentropy_mean')
                
                targs = tf.argmax(tf.reshape(self.targets,[self.batch_size*self._num_steps, output_size]), 1)
                loss = tf.nn.seq2seq.sequence_loss_by_example(
                    [self.logits],
                    [targs],
                    [tf.ones([self.batch_size*self.num_steps])]) #weights
                loss = tf.reduce_sum(loss)/self.batch_size            
                
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
    


