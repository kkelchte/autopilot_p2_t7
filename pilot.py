"""Build the pilot LSTM network"""

import tensorflow as tf
import numpy as np
import math

class ModelError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class LSTMModel(object):

    def __init__(self, is_training, config):
        self._batch_size = batch_size = config.batch_size
        self._num_steps = config.num_steps
        self._is_training = is_training
        self._prefix = config.prefix
        
            
        # Build LSTM graph
        self._logits = self.inference(config.num_layers, config.hidden_size, config.output_size,
                                    config.feature_dimension, config.keep_prob, config.gpu)
        self._cost = self.loss(config.output_size, config.gpu)
        
        if self._is_training:
            self.training(optimizer_type=config.optimizer)
        
        
        #learning_rate_summary = tf.scalar_summary("learning rate", self.lr)
    
    
    def inference(self, num_layers, hidden_size, output_size, feature_dimension, keep_prob, gpu=True):
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
            self._inputs = tf.placeholder(tf.float32, [self.batch_size, self._num_steps, feature_dimension], name=self._prefix+"_input_ph")
        print "inference ",self.prefix,": batch ", self.batch_size," size ",hidden_size," keep_prob ",keep_prob," layers ",num_layers," ",int(num_layers)
        # Build the LSTM Graph for testing
        with tf.variable_scope("LSTM"):
            with tf.device(device_name):
                lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=0.9) #'good' results: 0.9. Init: 0. add forget_bias in order to reduce scale of forgetting ==> remember longer initially: f = f'(0.1) + f_b(0.9) = 1.0 ==> keep all? ==> c = f * c_-1 . Default value is 1.
                if self.is_training and keep_prob < 1:
                    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
                cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * int(num_layers))
                #import pdb; pdb.set_trace()
                self._initial_state = cell.zero_state(self.batch_size, tf.float32)
                
                if self.is_training and keep_prob < 1:
                    self._inputs = tf.nn.dropout(self.inputs, keep_prob)
                    
                #unfolled a greater network over num_steps timesteps concatenating the output
                outputs = []
                state = self.initial_state
                states = []
                for time_step in range(self._num_steps):
                    if time_step > 0: tf.get_variable_scope().reuse_variables()
                    if time_step == 1: self._state = state # the next run will need the state after the first step to continue
                    (cell_output, state) = cell(self.inputs[:, time_step, :], state)
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
                weights = tf.Variable(
                    tf.truncated_normal([hidden_size, output_size],
                                        stddev=1.0 / math.sqrt(float(hidden_size))),
                    name='weights')
                #weights = tf.Variable(tf.constant(0.1,shape=[hidden_size, output_size]),
                #                      "weights")
                biases = tf.Variable(tf.zeros([output_size]),
                                    name='biases')
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
            for i in xrange(self._num_steps-1, self.batch_size*self._num_steps,self._num_steps):
                # for each num_steps period you should keep the last prediction
                self.preds_val.append(tf.scalar_summary(self._prefix+"_predictions_b"+str((i-self._num_steps+1)/self._num_steps), preds[i]))
            
            #write away
            #self.preds_val_tot = []
            #for i in range(self._num_steps*self.batch_size):
            #    self.preds_val_tot.append(tf.scalar_summary(self._prefix+"_predictions", preds[i]))
        return logits
    
    def loss(self, output_size, gpu=True):
        """ Ops required to generate loss.
        """
        device_name='/cpu:0'
        if gpu:
            device_name='/gpu:0'
            #device_name='/cpu:0'
        
        # Necessary placeholders for input
        with tf.device(device_name):
            self._targets = tf.placeholder(tf.float32, [self.batch_size, self._num_steps, output_size], name=self._prefix+"_targets_ph")
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
                
            #in case num_steps is the full movie length => return the full movie length    
            #self.targs_val_tot=[]
            #for i in range(self._num_steps*self.batch_size):
            #    self.targs_val_tot.append(tf.scalar_summary(self.prefix+"_targets", targs[i]))
        return loss
    
    def training(self, gpu=True, optimizer_type='GradientDescent'):
        """
        ops required to compute and apply gradients
        """
        device_name='/cpu:0'
        if gpu:
            device_name='/gpu:0'
            #device_name='/cpu:0'
        
        with tf.device(device_name):
            self._lr = tf.Variable(0.0, trainable=False, name=self._prefix+"_learning_rate")
        
        with tf.name_scope(self._prefix+'_training'):
            with tf.device(device_name):
                tvars = tf.trainable_variables()
                grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars), 5)# max_grad_norm = 5
                #import pdb; pdb.set_trace()
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
                #elif optimizer_type == 'Adadelta':
                #    optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.001, rho=0.95, epsilon=1e-08, use_locking=False, name='Adadelta')
                else : 
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
                            self.logits_hist]+self.preds_val+self.targs_val #self.loss_val, 
        return tf.merge_summary(list_of_tensors)
    
    def merge_all(self):
        """
        Specify which values are written down in the logfile and visualized on the tensorboard
        for the case that the network is unrolled over all frames and all batches (return all values)
        NOT WORKING so far...
        """
        for i in range(self.targs_val_tot):
            summary_string = tf.merge_summary(targs_val_tot[i])
            writer.add_summary(summary_string, i)
    
    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))
        
    @property
    def prefix(self):
        return self._prefix

    @property
    def is_training(self):
        return self._is_training
    
    @property
    def writer(self):
        return self._writer
    #@property
    #def preds(self):
    #    return self._preds
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
    def cost(self):
        return self._cost
        
    @property
    def state(self):
        return self._state
    
    @property
    def states(self):
        return self._states

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op
    
    @property
    def num_steps(self):
        return self._num_steps
    
    @property
    def batch_size(self):
        return self._batch_size
    


