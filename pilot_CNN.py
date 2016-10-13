"""
This script extract features from a pretrained CNN saved in a .pb, .txt. and .pbtxt file.
In this case the inception network trained on imagenet.

"""
import os, sys, re
import tensorflow as tf
from os import listdir
from os.path import isfile,join,isdir
import numpy as np
from PIL import Image
import string
import pilot_read as pire
import scipy.io as sio
import time
import pyinotify
import stat

from os.path import isdir, join, isfile

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'model_file', '/users/visics/kkelchte/tensorflow/examples/tutorial_imageclassification/inception/classify_image_graph_def.pb',
    """Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt.""")

tf.app.flags.DEFINE_string(
    "machine","qayd", #features_log/big_app",
    "define the machine on which the LSTM is loaded and to which features are written.")

tf.app.flags.DEFINE_string(
    'tensor', 'pool_3:0',
    """the name of the tensor from which the features are extracted."""
    """choose pool_3:0 or softmax:0 for inception network"""
    """choose final_result:0 for finetuned network""")

tf.app.flags.DEFINE_string(
    'network', 'inception',
    """the name of the CNN network from which features are extracted.
    this name is picked for naming convention. Pick inception, pcnn or finetuned.""")

tf.app.flags.DEFINE_boolean(
    "online", False,
    "Choose online evaluation when script waits for features and outputs control")

tf.app.flags.DEFINE_boolean(
    "gpu", True,
    "Choose if it has to fit on GPU or not")

tf.app.flags.DEFINE_boolean(
    "ssh", False,
    "In case I ssh to jade the source folder of features should be changed and the time delay should be added.")

tf.app.flags.DEFINE_string(
    'data_dir_offline', '/esat/emerald/tmp/remote_images/',
    """Path to data where images are saved""")

tf.app.flags.DEFINE_string(
    'data_dir_online', '/esat/emerald/tmp/remote_images/set_online',
    """Path to data where images are saved online. This is the source directory for the network.""")

tf.app.flags.DEFINE_string(
    'chosen_set',"",
    """Define from which movie in remote_images pilot_CNN needs to extract the features"""
    )

frame=0 # frame count for online feature extraction
current_image="" # first image in queue ready to be processed
last_image="" #last image that was processed
feature_tensor=None #the tensor that should be invoked to obtain the feature
delay=0

def print_time(start_time):
    '''Print the time passed after the start_time defined in hours, minutes and seconds
    Arg:
        start_time: the moment from which you started counting the time.
    Returns:
        string with time message.
    '''
    duration = (time.time()-start_time)
    return print_duration(duration)
    
def print_duration(duration):
    m, s = divmod(duration, 60)
    h, m = divmod(m, 60)
    return "time: %d:%02d:%02d" % (h, m, s)
    
def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    
    with tf.gfile.FastGFile(FLAGS.model_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    
def extract_offline():
    '''
    Extract CNN features of appearance and flow images using the inception google network 
    pretrained on imagenet.
    '''
    if len(FLAGS.chosen_set) != 0:
        chosen_set=FLAGS.chosen_set
    else:
        chosen_set=''
    
    movies_dir=FLAGS.data_dir_offline+chosen_set
    
    movies=[mov for mov in sorted(listdir(movies_dir)) if (isdir(join(movies_dir, mov)) and mov != "cache" and mov != "val_set.txt" and mov != "train_set.txt" and mov != "test_set.txt" and mov != "notify.sh" and mov != "set_1" and mov != "set_2" and mov != "set_online" and mov != "set_3" and mov != "failures" and mov!='log' and mov != 'expert')]
    
    #if len(FLAGS.movie) != 0:
        #movies = [FLAGS.movie]
    #else:
        #movies = ["set_7"]
    # creates the graph from saved GraphDef
    create_graph()
    
    with tf.Session() as sess:
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        
        # Some useful tensors:
        # 'softmax:0': A tensor containing the normalized prediction across
        #   1000 labels.
        # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
        #   float description of the image.
        # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
        #   encoding of the image.
        # Runs the softmax tensor by feeding the image_data as input to the graph.
        feature_tensor = sess.graph.get_tensor_by_name(FLAGS.tensor)
        
        movie_features = []
        i = 0
        for m in movies:
            starttime = time.time()
            print 'movie: ', m,': ',i+1,' of ', len(movies)
            i = i+1
            #create dir if necessary
            directory=join(movies_dir,m,'cnn_features')
            if not os.path.exists(directory):
                os.makedirs(directory)
                os.chmod(directory, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            movie_features_app = []
            rgb_images=[img for img in sorted(listdir(join(movies_dir,m,'RGB'))) if isfile(join(movies_dir,m,'RGB',img))]
            for img in rgb_images:
                #APP
                rgbfile = join(movies_dir,m,'RGB',img)
                #print 'img: ', img, ' of ', len(rgb_images)
                image_data = tf.gfile.FastGFile(rgbfile, 'rb').read()
                features = sess.run(feature_tensor,{'DecodeJpeg/contents:0': image_data})
                features = np.squeeze(features)
                #print 'output: ', features
                movie_features_app.append(features)
            #import pdb; pdb.set_trace()
            movie_features_app = np.concatenate([movie_features_app])
            d = {'features': movie_features_app, 'object': m}
            sio.savemat(join(movies_dir,m,'cnn_features','app_'+m+'_'+FLAGS.network+'.mat'), d, appendmat=False)

            if isdir(join(movies_dir,m,'depth')):
                movie_features_flow = []
                #flow_images=[img for img in sorted(listdir(join(movies_dir,m,'flow'))) if isfile(join(movies_dir,m,'flow',img))]
                flow_images=[img for img in sorted(listdir(join(movies_dir,m,'depth'))) if isfile(join(movies_dir,m,'depth',img))]
                for img in flow_images:
                    #FLOW
                    flowfile = join(movies_dir,m,'depth',img)
                    image_data = tf.gfile.FastGFile(flowfile, 'rb').read()
                    features = sess.run(feature_tensor,{'DecodeJpeg/contents:0': image_data})
                    features = np.squeeze(features)
                    movie_features_flow.append(features)
                movie_features_flow = np.concatenate([movie_features_flow])
                #import pdb; pdb.set_trace()
                d = {'features': movie_features_flow, 'object': m}
                #sio.savemat(join(movies_dir,m,'cnn_features','flow_'+m+'_'+FLAGS.network+'.mat'), d, appendmat=False)
                sio.savemat(join(movies_dir,m,'cnn_features','depth_'+m+'_'+FLAGS.network+'.mat'), d, appendmat=False)
            
            print 'Duration of last movie: ', print_time(starttime), '. Estimated time: ', print_duration(int((time.time()-starttime)*(len(movies)-i)))
        

def extract_online():
    global frame
    global delay
    global current_image
    global last_image
    global feature_tensor
    '''Use pyinotify to read images from /esat/qayd/kkelchte/simulation/remote_images
    extract features from next-to-last layer of inception3 and save in /esat/qayd/kkelchte/simulation/remote_features
    '''
    #source_dir='/esat/sadr/tmp/remote_images/'#Dont forget to adapt sshfs bridge on laptop as well if i use different source folder.
    source_dir=FLAGS.data_dir_online+'/RGB'
    print source_dir
    #source_dir='/esat/qayd/kkelchte/simulation/remote_images/'
    #source_dir='/esat/qayd/kkelchte/simulation/remote_images_qayd/'
    if FLAGS.ssh:
        des_dir ='/esat/'+FLAGS.machine+'/tmp/remote_features/'
    else:
        des_dir='/esat/qayd/kkelchte/simulation/remote_features/'
    if FLAGS.gpu or FLAGS.ssh:
        device_name='/gpu:0'
    else:
        device_name='/cpu:0'
    with tf.device(device_name):
        create_graph()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
        #   float description of the image.
        # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
        #   encoding of the image.
        # Runs the softmax tensor by feeding the image_data as input to the graph.
        feature_tensor = sess.graph.get_tensor_by_name(FLAGS.tensor)
        ##initialize notifier
        # watch manager
        wm = pyinotify.WatchManager()
        wm.add_watch(source_dir, pyinotify.IN_CREATE)
        # event handler
        eh = MyEventHandler()
        # notifier working in the same thread
        notifier = pyinotify.Notifier(wm, eh, timeout=10)
        
        def on_loop(notifier):
            global delay
            global current_image
            global last_image
            global frame
            global feature_tensor
            #import pdb; pdb.set_trace()
            if current_image == last_image:
                #print 'wait'
                return
            else:
                print "run through network: ",frame
                try:
                    if FLAGS.ssh: time.sleep(0.01) #wait till file is fully arrived on local harddrive when using ssh
                    #obtain image
                    image_data = tf.gfile.FastGFile(current_image, 'rb').read()
                    features = sess.run(feature_tensor,{'DecodeJpeg/contents:0': image_data})
                    
                    #copy image location if features are correctly extracted.
                    last_image='%s' % current_image
                    
                    features = np.squeeze(features)
                    features = np.asarray([[features]])
                    #print "shape check: 1,1,2048 = ", features.shape
                    d = {'features': features}
                    sio.savemat(des_dir+str(frame)+'.mat', d)
                    frame=frame+1
                    print "delay due to this image: ", time.time()-delay
                except Exception as e:
                    print "[CNN] skip image due to error."
        
        notifier.loop(callback=on_loop)

class MyEventHandler(pyinotify.ProcessEvent):
    global current_image
    global last_image
    global delay
    #Object that handles the events posted by the notifier
    def process_IN_CREATE(self, event):
        global current_image
        global last_image
        global delay
        current_image = event.pathname
        print "received image: ", current_image
        print "last image: ", last_image
        print "total delay from previous frame: ", time.time()-delay
        delay=time.time()    
#####################################################################
def main(_):
    '''
    call whatever function you would like to test.
    '''
    if FLAGS.online:
        FLAGS.ssh = True
        extract_online()
    else:
        extract_offline()

if __name__ == '__main__':
    print "------------------------------------------"
    print "Local run of pilot_CNN for testing network"
    print "------------------------------------------"
    tf.app.run()
    
    
