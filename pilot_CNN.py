import os, sys, re
import tensorflow as tf
from os import listdir
from os.path import isfile,join
import numpy as np
from PIL import Image
import string
import pilot_read as pire
import scipy.io as sio
import time

from os.path import isdir, join, isfile

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'model_dir', '/users/visics/kkelchte/tensorflow/examples/tutorial_imageclassification/inception/',
    """Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt.""")

tf.app.flags.DEFINE_string(
    'tensor', 'final_result:0',
    """the name of the tensor from which the features are extracted."""
    """choose pool_3:0 or softmax:0 for inception network"""
    """choose final_result:0 for finetuned network""")

tf.app.flags.DEFINE_string(
    'network', 'finetuned',
    """the name of the CNN network from which features are extracted.
    this name is picked for naming convention. Pick inception, pcnn or finetuned.""")

### DEFINE CNN NETWORK ###

#def inference():

#def loss():

#def train():

#def test():

def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(os.path.join(
        FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    
def extract():
    '''
    Extract CNN features of appearance and flow images using the inception google network 
    pretrained on imagenet.
    '''
    chosen_set='generated'
    movies_dir="/esat/qayd/kkelchte/simulation/"+chosen_set
    movies=[mov for mov in sorted(listdir(movies_dir)) if (isdir(join(movies_dir, mov)) and mov != "cache" and mov != "val_set.txt" and mov != "train_set.txt" and mov != "test_set.txt")]
    
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
            movie_features_app = []
            rgb_images=[img for img in sorted(listdir(join(movies_dir,m,'RGB'))) if isfile(join(movies_dir,m,'RGB',img))]
            for img in rgb_images:
                #APP
                rgbfile = join(movies_dir,m,'RGB',img)
                #print 'img: ', img, ' of ', len(rgb_images)
                image_data = tf.gfile.FastGFile(rgbfile, 'rb').read()
                features = sess.run(feature_tensor,{'DecodeJpeg/contents:0': image_data})
                features = np.squeeze(features)
                movie_features_app.append(features)
            movie_features_flow = []
            flow_images=[img for img in sorted(listdir(join(movies_dir,m,'flow'))) if isfile(join(movies_dir,m,'flow',img))]
            for img in flow_images:
                #FLOW
                flowfile = join(movies_dir,m,'flow',img)
                image_data = tf.gfile.FastGFile(flowfile, 'rb').read()
                features = sess.run(feature_tensor,{'DecodeJpeg/contents:0': image_data})
                features = np.squeeze(features)
                movie_features_flow.append(features)
            movie_features_app = np.concatenate([movie_features_app])
            movie_features_flow = np.concatenate([movie_features_flow])
            
            d = {'features': movie_features_app, 'object': m}
            sio.savemat(join(movies_dir,m,'cnn_features','app_'+m+'_'+FLAGS.network+'.mat'), d, appendmat=False)
            d = {'features': movie_features_flow, 'object': m}
            sio.savemat(join(movies_dir,m,'cnn_features','flow_'+m+'_'+FLAGS.network+'.mat'), d, appendmat=False)
            
            print 'duration: ', int(time.time()-starttime), '. Estimated time: ', int((time.time()-starttime)*(len(movies)-i))
        
    

#####################################################################
def main(_):
    '''
    call whatever function you would like to test.
    '''
    extract()

if __name__ == '__main__':
    print "------------------------------------------"
    print "Local run of pilot_CNN for testing network"
    print "------------------------------------------"
    tf.app.run()
    
    
