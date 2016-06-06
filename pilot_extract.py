"""
Extract the finetuned inception v3 features of pilot data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from os import listdir
from os.path import isfile,join, isdir

import tensorflow as tf

from tensorflow.examples.inception import inception_eval
from tensorflow.examples.inception.pilot_data import PilotData
from tensorflow.examples.inception import inception_model as inception

import pilot_read

FLAGS = tf.app.flags.FLAGS

BATCH_SIZE = 32
data_list='imglist_generated_tmp.txt'

#tf.app.flags.DEFINE_string('checkpoint_dir', '/esat/qayd/kkelchte/tensorflow/lstm_logs/finetuning_inception/',
#    """Directory where to read model checkpoints.""")
#tf.app.flags.DEFINE_string('eval_dir', '/esat/qayd/kkelchte/tensorflow/lstm_logs/eval_finetuning_inception/',
#    """Directory where to write event logs.""")

def extract():
    '''
    Extract CNN features of appearance images (not flow) using the inception google network 
    pretrained on imagenet.
    '''
    #Prepare data to image and label
    chosen_set='generated'
    movies_dir="/esat/qayd/kkelchte/simulation/"+chosen_set
    movies=[mov for mov in sorted(listdir(movies_dir)) if (isdir(join(movies_dir, mov)) and mov != "cache" and mov != "val_set.txt" and mov != "train_set.txt" and mov != "test_set.txt")]
    
    checkpoint_dir = '/esat/qayd/kkelchte/tensorflow/lstm_logs/finetuning_inception/'
    eval_dir = '/esat/qayd/kkelchte/tensorflow/lstm_logs/eval_finetuning_inception/'
    #Create graph
    with tf.Graph().as_default():
        #Get data as a 4D tensor and labels in a 1D tensor
        labels, images = pilot_read.inputs(data_list, BATCH_SIZE, data_type='app', num_preprocess_threads=8)
        
        # Number of classes in the Dataset label set plus 1.
        # Label 0 is reserved for an (unused) background class.
        num_classes = 5 #4+1
        
        #I should get the aux_logits as they are the features of the one before last layer
        #ERROR of T type of string is not allowed but i dont see any string...
        logits, _ = inception.inference(images, num_classes)
        import pdb; pdb.set_trace()
            
        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            inception.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        with tf.Session() as sess:
            # Build the summary operation based on the TF collection of Summaries.
            summary_op = tf.merge_all_summaries()
            summary_writer = tf.train.SummaryWriter(eval_dir,
                graph=graph)
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                if os.path.isabs(ckpt.model_checkpoint_path):
                    # Restores from checkpoint with absolute path.
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    # Restores from checkpoint with relative path.
                    saver.restore(sess, os.path.join(checkpoint_dir,
                                                    ckpt.model_checkpoint_path))

                # Assuming model_checkpoint_path looks something like:
                #   /my-favorite-path/imagenet_train/model.ckpt-0,
                # extract global_step from it.
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                print('Succesfully loaded model from %s at step=%s.' %
                        (ckpt.model_checkpoint_path, global_step))
            else:
                print('No checkpoint file found')
                return
            init_op = tf.initialize_all_variables()
            sess.run(init_op)
            
            # get coordinator for arranging the different threads
            coord = tf.train.Coordinator()
            # start all queue runners in the graph and returns a list of threads
            threads = tf.train.start_queue_runners(coord=coord)
            
            logits = sess.run([logits])
            #coordinator asks all threads to stop
            coord.request_stop()
            #wait till all threads are stopped
            coord.join(threads)
            

####

def main(unused_argv=None):
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  extract()

if __name__ == '__main__':
  tf.app.run()
