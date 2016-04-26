import pilot_data
import os
import tensorflow as tf
from os import listdir
from os.path import isfile,join
import numpy as np

#def inference():

#def loss():

#def training():



print "-----------------------------"
print "Run test module of pilot_data"
print "-----------------------------"
objects = ['dumpster', 'box']
data_dir = '/esat/qayd/kkelchte/simulation/data/'
test_dir = '/esat/qayd/kkelchte/tensorflow/cnn_logs'
num_epochs = 3

if tf.gfile.Exists(test_dir):
    tf.gfile.DeleteRecursively(test_dir)
tf.gfile.MakeDirs(test_dir)

batch_size = 100
files = []
for data_object in objects:
    directoryname = data_dir + data_object+'_one_cw'
    directoryname = join(directoryname,'RGB')
    print directoryname
    if not os.path.isdir(directoryname):
        print "Error: Directory does not exist."
    # get a list in Comma-Separated Value format 
    [files.append(f) for f in sorted(listdir(directoryname)) if isfile(join(directoryname,f))]

filename_queue = tf.train.string_input_producer(files)
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
import pdb; pdb.set_trace()
record_defaults = [["no_file.jpg"]]
image = tf.decode_csv(value, record_defaults=record_defaults)
import pdb; pdb.set_trace()
with tf.Session() as sess:
    # Start populating the filename queue.
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    #sess.run(tf.initialize_all_variables())
    for i in range(1200):
        # Retrieve a single instance:
        image_name = sess.run(image)
        import pdb; pdb.set_trace()

    coord.request_stop()
    coord.join(threads)
#images, labels = pilot_data.inputs(objects, data_dir, batch_size)

#with tf.Graph().as_default(), tf.Session() as session:
    
