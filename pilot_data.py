"""
 This code is used by pilot_train for getting the data and the ground truth from the images
"""
 
import os
import re
import logging
import scipy.io as sio
import numpy as np
import tensorflow as tf
from os import listdir
from os.path import isfile,join, isdir
import time
import random

from PIL import Image
import skimage
import skimage.transform
from skimage import io

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean(
    "normalized", False,
    "Whether or not the input data is normalized (mean substraction and divided by the variance) of the training data.")
tf.app.flags.DEFINE_string(
    "network", "inception",
    "Define from which CNN network the features come: pcnn or inception or logits_clean or logits_noisy or stijn.")
tf.app.flags.DEFINE_string(
    "feature_type", "app",
    "app or flow or or depth_estimate or both.")
tf.app.flags.DEFINE_integer(
    "feature_dimension", 2048,
    "define the size of the features according to n-step-FC and network.")
tf.app.flags.DEFINE_string(
    "dataset", "generated",
    "pick the dataset in data_root from which your movies can be found.")
tf.app.flags.DEFINE_string(
    "data_root", "/esat/emerald/tmp/remote_images",
    "Define the root folder of the different datasets.")
tf.app.flags.DEFINE_integer(
    "sample", "1",
    "Choose the amount the movies are subsampled. 16 is necessary to fit on a 2Gb GPU for both pcnn features.")
tf.app.flags.DEFINE_boolean(
    "one_hot", True,
    "Whether or not the input data has one_hot labels after the -gt tag or not.")
tf.app.flags.DEFINE_boolean(
    "read_from_file", True,
    "Whether or not labels are written in the RGB files or in a file: control_info.txt")
tf.app.flags.DEFINE_string(
    "data_type", "listed",
    "Choose how the data need to be prepared: grouped means into batches of sequences of similar length. Batched means that it splits up the data over batches of 1000 frames. Otherwise it will take each movie as a separate batch and work with batch size of 1 [default].")
tf.app.flags.DEFINE_integer(
    "batch_length",100,
    "In case of batched data_type choose the length in which the sequences are sliced.")
tf.app.flags.DEFINE_integer(
    "max_batch_size",100,
    "In case of batched data_type choose the maximum number of slices there can be in one batch.")
tf.app.flags.DEFINE_boolean(
    "continuous", True,
    "Define the type of the output labels.")
tf.app.flags.DEFINE_boolean(
    "short_labels", False,
    "Use only 2 labels, one for up/down, one for left/right.")
tf.app.flags.DEFINE_integer(
    "max_num_windows",500,
    "Define max number of initial_state calculations for one epoch.")
tf.app.flags.DEFINE_boolean(
    "sliding_window", False,
    "Choose whether windows are picked randomly or are sliding over.")
tf.app.flags.DEFINE_float(
    "scale", "0.2", 
    "Define the rate at which the learning rate decays.")
tf.app.flags.DEFINE_boolean(
    "preloading", True,
    "Choose whether the data is totally loaded in the beginning or only at the moment it is necessary.")
tf.app.flags.DEFINE_integer(
    "cut_end", 0,
    "Choose how many frames of the end are left out in order to avoid too large gradients.")

class DataError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
 
def extract_features(filename, network='inception', load_data=True):
    """Extract the features from a mat file into an ndarray [features of frame 0, frame 1, frame 2, ...] 
    Args:
        filename: the path pointing to a mat file keeping the features
        network: the type of features found in this file ==> need of different extraction
    Returns:
        2darray object containing features of each frame [movie_length, feature_dimension]
    """
    #print 'Extracting '+str(filename)
    #print 'network: ', network
    
    #define feature size
    if network == 'pcnn': FLAGS.feature_dimension = 4096
    elif network == 'inception': FLAGS.feature_dimension = 2048
    elif (network == 'logits') or (network == 'logits_noisy'): FLAGS.feature_dimension = 4
    elif network == 'stijn': FLAGS.feature_dimension = 4070
    
    #check out the features
    data = sio.loadmat(filename)
    if network == 'pcnn': 
        feats = data['features'][0,0][1]      
    elif network == 'inception':
        feats = data['features']
    elif (network == 'logits') or (network == 'logits_noisy'): 
        feats = data['features']
    elif network == 'stijn':
        data=data['gazebo_sim_dataset']
        feats = data[0,0]['labels']
    
    #check loaded feature dimension
    if feats.shape[1]!=FLAGS.feature_dimension:
        loggin.warning("dimensions mismatch with stijn features, should be "+str(FLAGS.feature_dimension))
    
    #cut the end of the movie in order to avoid strong gradients that are irrelevant
    if FLAGS.cut_end != 0:
        feats = feats[0:-FLAGS.cut_end, :]
    
    #send array of one zero back in case of no preloading
    if not load_data: return np.zeros((feats.shape[0], 1), dtype=np.int8)
    return feats

   
def load_images(directoryname, depth=False, load_data=True):
    """Load the image in an 2d array [image 0, image 1, ... ]
    Args:
        directory name: path pointing to RGB dir with images
    Returns:
        2darray object containing oncatenated image for each frame
    """
    if not isdir(directoryname):
        raise IOError("directory does not exist")
    image_names = [ f for f in listdir(directoryname) if f[-3:]=='jpg' ]
    scale=0.2
    if FLAGS.scale and FLAGS.scale != 4: scale = FLAGS.scale
    max_shape=int(360*640*3*scale*scale)
    if depth: max_shape=int(360*640*scale*scale)
    FLAGS.feature_dimension = max_shape
    
    #cut the end of the movie in order to avoid strong gradients that are irrelevant
    if FLAGS.cut_end != 0:
        image_names = image_names[0:-FLAGS.cut_end, :]
    
    if not load_data: return np.zeros([len(image_names), 1], dtype=np.int8)
    
    images = np.zeros([len(image_names),max_shape])
    #print 'shape of feature of: ',images.shape
    c=0
    for im_n in image_names:
        #print 'im_n ',im_n
        filename=join(directoryname, im_n)
        im_array = io.imread(filename)
        im_array = skimage.transform.rescale(im_array, scale)
        #img=Image.fromarray(im_array)
        #img.show()
        #import pdb; pdb.set_trace()
        ###normalize to floats between 0 and 1
        images[c,:] = np.reshape(im_array, [1,max_shape])
        c=c+1
    return images
    
def load_image(path_to_image, depth=False):
    """Read in image from file and export as proper feature."""
    scale = 0.2
    if(FLAGS.scale != 4): scale=FLAGS.scale 
    max_shape=int(360*640*3*scale*scale)
    if(depth):max_shape=int(360*640*scale*scale)
    if not os.path.isfile(path_to_image):
        raise IOError("path_to_image does not exist.")
    success=False
    while not success:
        try:
            #import pdb; pdb.set_trace()
            im_array = io.imread(path_to_image)
            im_array = skimage.transform.rescale(im_array, 0.2)
            success = True
        except:
            print 'failed to load',path_to_image
            time.sleep(0.005)
    return np.reshape(im_array, [1,1,max_shape])
    
def get_labels_from_directory(directoryname):
    """This functions extract the ground truth from the images in the RGB directory defined by the directoryname
    and return an array with labels for each frame"""
    directoryname = join(directoryname,'RGB')
    print 'Get labels from '+str(join(directoryname))
    if not os.path.isdir(directoryname):
        raise IOError("directory does not exist: "+str(directoryname))
        return -1
    files = [f for f in listdir(directoryname) if isfile(join(directoryname,f))]
    #obtain the label indicated after 'gt' from the names of the files
    labels = [re.search(r'gt\d*',name).group(0)[2:] for name in sorted(files)]
    
    #move all the labels one step earlier in time: remove the first item and repeat the last item
    #this makes the control appears before the image instead of the otherway around how the images are labelled.
    labels.remove(labels[0])
    labels.insert(-1, labels[-1])
    
    #go from list of strings to array of integers
    labels = np.array(labels).reshape((-1,1))
    
    #cut the end of the movie in order to avoid strong gradients that are irrelevant
    if FLAGS.cut_end != 0:
        labels = labels[0:-FLAGS.cut_end]
    
    return labels

def create_one_hot_labels(labels):
    """
    Args: take an 2d array [num_frames, 1] and converts them in one hot labels.
    Return: 2d array [num_frames, n] with n the number of discrete options.
    create labels [1], [3] into labels [0100], [0001].
    of np arrays
    """
    # get maximum ~ number of columns
    mx = int(max(max(labels)))
    oh_labels = np.zeros([labels.shape[0], mx+1])
    for i in range(labels.shape[0]):
        oh_labels[i, int(labels[i,0])]=1
    return oh_labels

def get_labels_from_file(directory):
    """
    read out the labels from a txt control file.
    first 10 ints are the index -space- 6 floats defining the twist or 41 bits in case of discrete
    """
    filename=join(directory,"control_info.txt")
    if not os.path.isfile(filename):
        raise IOError("control file does not exist: "+str(filename))
        return -1
    # read the filename and return the list.
    list_f = open(filename,'r')
    # read lines in a list
    list_c = list_f.readlines()
    # cut the newline at the end
    list_c = [l[:-1] for l in list_c]
    # cut the 10 continuous floats label in the beginning
    list_c = [l[10:] for l in list_c]
    # extract a label
    labels = np.zeros([int(len(list_c)), len(re.findall("\s*\S+", list_c[0]))])
    for i in range(len(list_c)):
        #print i,': ', len(re.findall("\s*\S+", list_c[i])), ' : ',re.findall("\s*\S+", list_c[i])
        lbl_l=[float(s) for s in re.findall("\s*\S+",list_c[i])]
        lbl = []
        for f in lbl_l:
        	if abs(f) > 1 and f > 0:
        		lbl.append(1)
        	elif abs(f) > 1 and f < 0:
        		lbl.append(-1)
        	else:
        		lbl.append(f)
        labels[i,:]=np.array(lbl)
    
    # in case of normal inception features: RGB is only every second label:
    #if FLAGS.network == 'inception':#and not FLAGS.feature_type == both
    #    labels=labels[range(1,len(list_c),2)]
    
    # use only the 2 numbers that are not 0.8 or 0 as labels 
    if FLAGS.short_labels:
    	labels=labels[:,[2,5]]
    
    #cut the end of the movie in order to avoid strong gradients that are irrelevant
    if FLAGS.cut_end != 0:
        labels = labels[0:-FLAGS.cut_end, :]
        
    return labels

def prepare_data(data_objects, load_data=True):
    """
    Create batches for the listed data objects by obtaining the features and the labels.
    Creating different batches of the same size
    []
    Used by prepare_data_list and pilot_train
    returns a 3D array with first dimension the index of the movie, [movie, frame, feature]
    Args:
        data_object: list of objects like modelaaa or dumpster found in the dataset
        FLAGS
            sample_step: the amount the movies are subsampled
            feature_type: app ~ appearance, flow ~ optical flow
            dataset: the directory in the data_root path where the objects are found
            normalized: boolean defining whether or not the data should be normalized (no clear improvements on this one so far)
            network: the features comming from the Pose-CNN network 'pcnn' or from the Inception network 'inception'
    """
    # get data 
    data = []
    labels = []
    min_steps = 100000
    if FLAGS.feature_type == 'both':
        #feature_types = ['app', 'flow']
        feature_types = ['depth_estimate', 'app']
    else:
        feature_types = [FLAGS.feature_type]
    # loop over trajectories
    for data_object in data_objects:
        #directory = FLAGS.data_root+FLAGS.dataset+'/'+data_object
        object_name = os.path.basename(data_object)
        #real labels from file or from RGB-directory
        if FLAGS.read_from_file:
            #labels_obj = get_labels_from_file(directory)
            labels_obj = get_labels_from_file(data_object)
        else:
            #labels_obj = get_labels_from_directory(directory)
            labels_obj = get_labels_from_directory(data_object)
        if not FLAGS.one_hot:
            labels_obj = create_one_hot_labels(labels_obj)
            
        minfeat = labels_obj.shape[0]# see that labels, app and flow features are from the same number of frames
        feats_obj = []        
        for ftype in feature_types: #loop over feature types 'app' and 'flow' or one of them
            if FLAGS.network == 'pcnn': 
                feats = extract_features(data_object+'/cnn_features/'+ftype+'_'+object_name+'.mat', FLAGS.network, load_data=load_data)
            elif FLAGS.network == 'inception' or FLAGS.network == 'logits' or FLAGS.network == 'logits_noisy' or FLAGS.network == 'stijn': 
                feats = extract_features(data_object+'/cnn_features/'+ftype+'_'+object_name+'_'+FLAGS.network+'.mat', FLAGS.network, load_data=load_data)
            elif FLAGS.network == 'no_cnn':
                feats = load_images(data_object+'/RGB', load_data=load_data)
            elif FLAGS.network == 'no_cnn_depth':
                feats = load_images(data_object+'/depth', depth=True, load_data=load_data)
            else: raise IOError("network not known: ", FLAGS.network)
            feats_obj.append(feats)
        #flow is on 2RBG images ==> skip first label to be equal
        if 'flow' in feature_types:
            labels_obj = labels_obj[1:]
        if len(feature_types) == 2:
            feats_obj[0] = feats_obj[0][0:-1] #shorten app list to be equal to flow list
            
        feats_obj = np.concatenate(feats_obj,1) #concatenate the list of features in the second dimension
        print "shape features:",feats_obj.shape
        
        #clean up the -1 s and zeros that come from stijns features
        if FLAGS.network == 'stijn':
            rgb_indices=[]
            #check out the features of value -1 and remove these features + labels from the list
            # Assume that if stijns depth estimate is used this is in front of the array of feature types
            for i in range(feats_obj.shape[0]):
                if not (feats_obj[i,0] == -1 and feats_obj[i,1] == -1 and feats_obj[i,2] == -1) and not (feats_obj[i,0] == 0 and feats_obj[i,1] == 0 and feats_obj[i,2] == 0): rgb_indices.append(i)
            try:
                feats_obj=feats_obj[rgb_indices,:]
                labels_obj=labels_obj[rgb_indices,:]
            except IndexError:
                print 'IndexError: ',rgb_indices[-1],'>=',min([labels_obj.shape[0], feats_obj.shape[0]])
        
        # make them equal lenght: might have missed out some label/feature        
        labels_obj = labels_obj[0:min(labels_obj.shape[0],feats_obj.shape[0])]
        feats_obj = feats_obj[0:min(labels_obj.shape[0],feats_obj.shape[0])]
        if abs(labels_obj.shape[0]-feats_obj.shape[0])>50:
            raise IOError("labels vs features mismatch: "+str(labels_obj.shape[0])+"!="+str(feats_obj.shape[0]))
        
        data.append(feats_obj)
        labels.append(labels_obj)
        if min_steps > labels_obj.shape[0]:#get number of frames of shortest movie
            min_steps = labels_obj.shape[0]   
    
    if FLAGS.sample>=min_steps:
        raise IOError("downsampling is larger than total data set "+str(FLAGS.sample)+">="+str(min_steps))
        
    # resize according to num_steps of shortest video
    # and downsample in time to get a frame every 10 or 100 frames
    ds_data = np.zeros((len(data),int(min_steps/FLAGS.sample),data[0].shape[1]))
    ds_labels = np.zeros((len(data), int(min_steps/FLAGS.sample), labels[0].shape[1]))
    #print ds_data.shape
    #print ds_labels.shape
    
    #loop over batches (in this implementation is this always 1 batch)
    for b in range(len(data)):
        j=0
        for i in range(min_steps): #loop over different timesteps
            if (i%FLAGS.sample) == 0 and j<ds_data.shape[1]:
                ds_data[b,j]= data[b][i]
                ds_labels[b,j]=labels[b][i]
                j=j+1
    
    #import pdb;pdb.set_trace()
    #Normalize the subsampled features if necessary
    if FLAGS.normalized and load_data:
        norm_data = np.zeros(ds_data.shape)
        data = sio.loadmat('mean_variance_'+FLAGS.feature_type+"_"+FLAGS.network+'.mat')
        mean = data['mean']
        variance = data['variance']
        epsilon = 0.001
        for i in range(ds_data.shape[0]):
            for j in range(ds_data.shape[1]):
                norm_data[i,j,:]=(ds_data[i,j,:]-mean[0])/(np.sqrt(variance[0])+epsilon)
        ds_data = norm_data
    return ds_data, ds_labels

    #return data, labels

def prepare_data_list(data_objects):
    """
    Returns a list of tuples contains (features, labels)_movie. Batchsize of features and labels = 1.
    features and labels have size [batchsize, numberframes, feature/label dimension]
    """
    object_list = []
    for d_object in data_objects:
        print "object: ",os.path.basename(d_object)
        object_data, object_labels = prepare_data([d_object], FLAGS.preloading)
        object_tuple = (object_data, object_labels, [d_object])
        object_list.append(object_tuple)
    return object_list

def prepare_data_grouped(data_objects):
    """
    Create a list of tuples. Each tuple contains a data and label. The batchsizes are grouped in the movies
    with equal length. [batchsize, numberframes, feature/label dimension]
    """
    #list and sort the data according to the length
    all_data = prepare_data_list(data_objects)
    
    #sort data
    sizes = [tu[0].shape[1] for tu in all_data]
    sizes = np.asarray(sizes)
    inds = np.arange(len(all_data))
    dtype = [('length', int), ('index', int)]
    sizes = [(sizes[i], inds[i]) for i in range(len(inds))]
    sizes = np.array(sizes, dtype=dtype)
    #print sizes
    sizes = np.sort(sizes, order='length')
    #print sizes
    #create a list of differences in length
    differences = [(0, sizes[0][1])]
    differences[1:] = [(sizes[i+1][0]-sizes[i][0], sizes[i+1][1]) for i in range(len(inds)-1)]
    #print differences
    dtype = [('difference', int), ('index', int)]
    differences = np.array(differences, dtype=dtype)
    differences = np.sort(differences, order='difference')
    #print differences
    ### ! changed minimum difference from 50 to 100 to 150 !
    split_indices = [differences[i][1] for i in range(len(differences)) if differences[i][0]>(150/FLAGS.sample)]
    #print split_indices
    
    # create a list of groups (lists) of movies which length  differs more than 50 frames
    groups = []
    current_group = []
    for s in sizes:
        #print 'size ',s[0],' index ',s[1]
        if s[1] in split_indices:
            groups.append(current_group)
            print 'create new group: ', len(groups), 'last group len: ', len(current_group)
            #create new group
            current_group = []
        #add to current group
        current_group.append(all_data[s[1]])
    groups.append(current_group)
    print 'finished: groups: ', len(groups), 'last group len: ', len(current_group)
    
    # for each group create an array  of equal length and batches.
    # if the movie is longer than average the last frames are cut away
    # if the movie is shorter than average the last frame is repeated a number of times
    batch_list = []
    for g in groups:
        #get average length of the movie[i][label][batch][frames]
        average = int(sum([len(g[i][1][0][:]) for i in range(len(g))])/len(g))
        print 'group average: ', average
        batch_data = []
        batch_labels = []
        for m in g:
            # m: tuple: (data,labels)
            # data: np.array [batch, frames, features]
            length = m[0].shape[1]
            if length > average:
                data = m[0][:,0:average,:]
                labels = m[1][:,0:average,:]
            elif length < average:
                frame = m[0][:,-1,:]
                frames = np.zeros((m[0].shape[0],average-length,m[0].shape[2]))
                frames = frames + frame
                data = np.concatenate((m[0], frames), axis=1);
                label = m[1][:,-1,:]
                labels = np.zeros((m[1].shape[0],average-length,m[1].shape[2]))
                labels = labels + label
                labels = np.concatenate((m[1], labels), axis=1);
            else:
                data = m[0]
                labels = m[1]
            batch_data.append(data)
            batch_labels.append(labels)
        #import pdb; pdb.set_trace()
        batch_data = np.concatenate(batch_data)    
        batch_labels = np.concatenate(batch_labels)   
        batch_list.append((batch_data, batch_labels))
    return batch_list

def prepare_data_batched(data_objects):
    """get batches of length of 1000 frames from the set of data objects
    args:
    data_objects = the list of sequences on which can be trained
    returns:
    A list of tuples. Each tuple contains a data and label batch. The batches are all of 1000 frames and picked among the different sequences. [batchsize, numberframes, feature/label dimension]
    """
    #list the data
    all_data = prepare_data_list(data_objects)
    data_batched = []
    for sequence in all_data:
        #sequence[0]#data
        #sequence[1]#labels
        num_batch = min([FLAGS.max_batch_size, int(sequence[0].shape[1]/FLAGS.batch_length)]) #number of batches for this sequence
        batch_data = np.zeros((num_batch, FLAGS.batch_length, sequence[0].shape[2]))
        batch_labels = np.zeros((num_batch, FLAGS.batch_length, sequence[1].shape[2]))
        for i in range(num_batch):
            batch_data[i:i+1,:,:]=sequence[0][:,i:i+FLAGS.batch_length,:]
            batch_labels[i:i+1,:,:]=sequence[1][:,i:i+FLAGS.batch_length,:]
            #import pdb; pdb.set_trace()
        data_tuple = (batch_data, batch_labels)
        print "data batch shape: ",batch_data.shape
        data_batched.append(data_tuple)
    return data_batched
    
def prepare_data_general(data_objects):
    print "data_type: ",FLAGS.data_type
    if FLAGS.data_type == "grouped":
        return prepare_data_grouped(data_objects)
    elif FLAGS.data_type == "batched":
        return prepare_data_batched(data_objects)
    else:
        return prepare_data_list(data_objects)
    
def get_objects():
    # given a dataset read out the train, validate and test_set.txt lists
    # return 3 lists of objects
    directory =os.path.join(FLAGS.data_root,FLAGS.dataset)
    print 'directory ', directory
    # read training objects from train_set.txt
    training_objects = get_list(directory,'train_set.txt')
    
    # read validate objects from validat_set.txt
    try: 
        validate_objects = get_list(directory,'val_set.txt')
    except IOError as e:
        validat_objects = None
        print 'val_set.txt does not exist.'
    # read test objects from test_set.txt
    test_objects = get_list(directory,'test_set.txt')
    return training_objects, validate_objects, test_objects

def get_list(directory,filename):
    # read the filename and return the list.
    list_f = open(join(directory,filename),'r')
    # read lines in a list
    list_o = list_f.readlines()
    # cut the newline at the end
    list_o = [l[:-1] for l in list_o]
    
    # cut the path, keep the name
    #list_o = [os.path.basename(l) for l in list_o]
    # or .. keep the path
    
    return list_o

def pick_random_windows(data_list, window_sizes, batch_sizes):
    '''create batches of different windowsizes for training different unrolled models.
    The batches are picked randomly from all the datalist. The batch and window sizes are defined in the arguments.
    Args:
        data_list: the data list containing tuples with data and labels of batches (of equal length) [works with listed data?]
        windowsizes: a list with the windowsizes needed corresponding to the list of unrolled networks to be trained
        batchsizes: a list of batchsizes for each window set according to the scale factor depending on the size of your RAM and GPU mem
    Return:
        indices: a nested list of lists of tuples with each tuple the according indices for chosen group, chosen batch, chosen timestep. 
        			The data is picked randomly from the data list. The outer list corresponds to the number of windowsizes/models.
        			The innerlist is the amount of times a window batch is picked in order to correspond to 1 epoch. This depends of the amount of data.
        			Each element in the list is a array with sets of movie-indices. Every set of movie indices is an array of 3 numbers:
        			[tuple index, movie index, start index] tuple index corresponds to the tuple in the data list, movie index is the index in the batch of that tuple and start index is the position of the window.
        
    '''
    output_size = data_list[0][1].shape[2]
    #print 'output_size: ', output_size
    
    ##preparation = define number of times the windowed-batched data should be iterated in one epoch
    if len(window_sizes) != len(batch_sizes):
    	raise SyntaxError('Number of windowsizes is not equal to number of batchsizes.')
    #number of frames in one batch if there were no extra innerlist iterations	
    b_num_frames = sum([ window_sizes[i] * batch_sizes[i] for i in range(len(window_sizes))])
    #total number available for one epoch
    d_num_frames = sum([ data_list[t][0].shape[0]*data_list[t][0].shape[1] for t in range(len(data_list))])
    #number of times the window-batches are repeated in one epoch
    b_scale = max([1, int(d_num_frames/b_num_frames)])
    if FLAGS.max_num_windows <= max(batch_sizes):
        b_scale = 1
    elif not FLAGS.fc_only: #in case initial_state calculations are necessary use upper limit of max_num_windows
        #else: #also clip for fc_only because otherwise innerlist becomes huge
        b_scale = min(b_scale, FLAGS.max_num_windows/max(batch_sizes))
    print "multiply window batches ",b_scale," times in one epoch."
    total_indices = [] #to be returned in the end
    for wi in range(len(window_sizes)):#outer list over models
    	#create list of tuples of movie and batch indices with movies longer than the required size
    	possible_movies=[ (ti, bi) for ti in range(len(data_list)) for bi in range(data_list[ti][0].shape[0]) if data_list[ti][0].shape[1] > window_sizes[wi]]
    	#print possible_movies
    	if len(possible_movies) == 0:
    		raise SyntaxError('Windowsize '+str(window_sizes[wi])+' is too big for this set of data ['+str(max([data_list[i][0].shape[1] for i in range(len(data_list))]))+' max]. Consider using fully unrolled option with --batchwise_learning True.')
    	local_indices = [] #list of length b_scale with tuples corresponding to this window size
    	for li in range(b_scale):
    		b_indices = [] #an array of indices for 1 batch for this window size
    		for bi in range(batch_sizes[wi]):
    			tup_ind, mov_ind = random.choice(possible_movies)
    			start_ind = random.choice(range(data_list[tup_ind][0].shape[1]-window_sizes[wi]))
    			b_indices.append(np.asarray([[tup_ind, mov_ind, start_ind]]))
    		b_indices = np.concatenate(b_indices)
    		local_indices.append(b_indices)
    	total_indices.append(local_indices)
    return total_indices

def pick_sliding_windows(data_list, window_sizes, batch_sizes):
    '''ERROR: SLIDING WINDOW SLIDES ONLY OVER SLIDING WINDOW LENGTH AND NOT FULL SEQUENCE create batches of with windowsizes for training different unrolled models.
    The batches are slided over different steps in the second dimension.
    
    Args:
        data_list: the data list containing tuples with data and labels of batches (of equal length) [works with listed data?]
        windowsizes: a list with the windowsizes needed corresponding to the list of unrolled networks to be trained
        batchsizes: a list of batchsizes for each window set according to the scale factor depending on the size of your RAM and GPU mem
    Return:
        indices: a nested list of lists of tuples with each tuple the according indices for chosen group, chosen batch, chosen timestep. 
        		The outer list corresponds to the number of windowsizes/models. The innerlist is the index of the next step of the sliding window of that batch.
        		Each element in the list is a array with sets of movie-indices. Every set of movie indices is an array of 3 numbers:
        		[tuple index, movie index, start index] tuple index corresponds to the tuple in the data list, movie index is the index in the batch of that tuple and start index is the position of the window.
        
    '''
    feature_size = data_list[0][0].shape[2]
    output_size = data_list[0][1].shape[2]
    if len(window_sizes) != len(batch_sizes):
    	raise SyntaxError('Number of windowsizes is not equal to number of batchsizes.')
    	
    total_indices = [] #to be returned in the end
    for wi in range(len(window_sizes)):#outer list over models
    	#create list of tuples of movie and batch indices with movies longer than the required size
    	possible_movies=[ (ti, bi) for ti in range(len(data_list)) for bi in range(data_list[ti][0].shape[0]) if  data_list[ti][0].shape[1] > window_sizes[wi]]
    	#number of movies over which needs to be slided
    	batch_size = batch_sizes[wi]
    	if len(possible_movies) < batch_size:
    		raise SyntaxError('Windowsize '+str(window_sizes[wi])+' is too big for this set of data ['+str(max([data_list[i][0].shape[1] for i in range(len(data_list))]))+' max]. Consider using fully unrolled option with --batchwise_learning True. Or set the sample rate lower.')
    	
    	#import pdb; pdb.set_trace()
    	
    	b_indices = []
    	for bi in range(batch_size): #select a batch size number of movies over which is slided
            #choose the movie over which the time window is slided
            tup_ind, b_ind = random.choice(possible_movies)
            b_indices.append((tup_ind, b_ind))
        
        #print b_indices 
        #!!! we assume that all movies are more or less of equal length after grouping !!!
        local_indices = []
    	for start_ind in range(data_list[0][0].shape[1]-window_sizes[wi]):
            int_mat = np.zeros([batch_size, 3], dtype=np.int)
            int_mat[:,2]=start_ind #start index
            int_mat[:,1]=[t[1] for t in b_indices] #movie index
            int_mat[:,0]=[t[0] for t in b_indices] #tuple index
            local_indices.append(int_mat)
            #print str(int_mat)
        total_indices.append(local_indices)
    return total_indices

def copy_windows_from_data(data_list, window_size, indices):
    ''' create batch of window_size length
    Args:
        data_list: list of tuples containing data and labels each arrays of size [batchsize, framelen, featuresize]
        windowsize: the windowsize needed corresponding to the list of unrolled networks to be trained
        batchsize: batchsizes for the window set according to the scale factor depending on the size of your RAM and GPU mem
        indices: matrix with a row for each batchelement containing: [tuple index, movie index, start index] tuple index corresponds to the tuple in the data list, movie index is the index in the batch of that tuple and start index is the position of the window.
    Returns:
         the data and labels according to the windowsize and batchsize as set by the arguments.
    '''
    batch_size = indices.shape[0]
    data = np.zeros((batch_size, window_size, FLAGS.feature_dimension))
    labels = np.zeros((batch_size, window_size, data_list[0][1].shape[2]))
    try:
        for i in range(batch_size):
            tup_ind, mov_ind, start_ind = indices[i]
            labels[i] = data_list[tup_ind][1][mov_ind][start_ind:start_ind+window_size][:]
            #print 'i: ',i,' tup ', tup_ind,' mov ',mov_ind,' start ',start_ind
            if FLAGS.preloading:
                data[i] = data_list[tup_ind][0][mov_ind][start_ind:start_ind+window_size][:]
        if not FLAGS.preloading:
            data = get_data_from_indices(data_list, window_size, indices)
            #print 'data received in copy window: ',data.shape
            #print data
    except ValueError:
        print '[pilot_data]data: ', str(data[i].shape) ,' doesnt fit with original data: ', str(data_list[0][0][mov_ind][start_ind:start_ind+window_size][:].shape)
        print '[pilot_data] tup ', tup_ind,' mov ',mov_ind,' start ',start_ind,' windowsize: ',window_size,' data length ', str(data_list[0][0][mov_ind].shape)
        return None, None
    return data, labels

def get_data_from_indices(data_list, window_size, indices):
    '''get the features/images from the memory
    Args:
        data_list: list of tuples containing both labels and features. Features are filled with 1 zero instead of the feature vector.
        window_size: the number of time steps that needs to be obtained.
        indices: a list of arrays with 3 elements which indicates the tuple (~which data_batch/data_object), which movie(0), @ which start index
    Returns:
        data: 3d array [batchsize, time_steps, feature_dim]
    '''
    #FLAGS.preloading = True
    movies=[i for i in range(indices.shape[0])]
    # if i want to sort and reuse loaded movies:
    #indices.view('i8,i8,i8').sort(order=['f1'], axis=0)
    data = np.zeros([indices.shape[0], window_size, FLAGS.feature_dimension])
    
    def MyLoop(coord, movies):
        while not coord.should_stop():
            try:
                i_ind = movies.pop()
                print i_ind,' of ', movies
                tup_ind, mov_ind, start_ind = indices[i_ind]
                d_object = data_list[tup_ind][2][mov_ind]
                #print str(i_ind),': obtain data from ',d_object,' : ', str(mov_ind),' starting at ', str(start_ind)
                data_full, labels = prepare_data([d_object], load_data=True) #make sure you load even though preloading might be false.
                #print data_full
                data[i_ind] = data_full[0][start_ind:start_ind+window_size][:]
                #print data[i_ind]
            except IndexError:
                #print 'Fetching data finished. Wait for threads to stop.'
                coord.request_stop()
        if len(movies)==0:
            coord.request_stop()
        coord.request_stop()
    try:
        # Main code: create a coordinator.
        coord = tf.train.Coordinator()
        # Create 10 threads that run 'MyLoop()'
        #num_threads=min(10,len(all_movies)/2)
        # each batch for a certain model can be prepared with another thread
        num_threads=len(movies)
        num_threads=min(FLAGS.max_num_threads, num_threads)
        threads = [tf.train.threading.Thread(target=MyLoop, args=(coord,movies)) for i in xrange(num_threads)]
        print 'number of threads: ',num_threads
        # Start the threads and wait for all of them to stop.
        for t in threads: t.start()
        coord.join(threads, stop_grace_period_secs=240) #wait max 4minutes to stop threads before raising exception
    except RuntimeError:
        print "Thread took more than 4 minutes to sleep so we sleep for an extra 4 minutes..."
        time.sleep(240)
    except Exception as e:
        print "Thread is still not ready so something is probably wrong? Or this is raised by another type of exception.", e.value
    
    #FLAGS.preloading = False
    return data
#####################################################################

if __name__ == '__main__':
    print "-----------------------------"
    print "Run test module of pilot_data"
    print "-----------------------------"
    #data_objects=['set_2', 'set_1'];
    #FLAGS.batch_length = 10;
    #FLAGS.dataset='../../tmp/remote_images'
    #FLAGS.sample = 1
    #FLAGS.one_hot=True
    #FLAGS.data_type = "batched"
    #training_data_list = prepare_data_general(data_objects)
    #import pdb; pdb.set_trace()
    ##training_objects=['modeldaa']
    #calculate_mean_variance(data_objects = training_objects,feature_type='both', network='inception')
    #result_data, result_labels =prepare_data(training_objects, feature_type='both', dataset='generated', normalized=True)
    #time.sleep(10)
    #print result_data.shape
    #print result_data
    #mylist=prepare_data_grouped(data_objects, feature_type='app', sample_step=8)
    #print type(mylist)
    #validate_objects = [os.path.join(FLAGS.data_root,FLAGS.dataset,o) for o in validate_objects]
    #test_objects = [os.path.join(FLAGS.data_root,FLAGS.dataset,o) for o in test_objects]
    
    FLAGS.dataset='sequential_oa'
    training_objects, validate_objects, test_objects = get_objects()
    
    #FLAGS.dataset='sequential_oa'
    #training_objects = ['sequential_oa_0000_0_1', 'sequential_oa_0000_0_2']
    #training_objects = [os.path.join(FLAGS.data_root,FLAGS.dataset,o) for o in training_objects]
    #FLAGS.network = 'no_cnn_depth'
    FLAGS.scale = 0.1
    
    
    
    #FLAGS.dataset='tiny_set'
    #training_objects = ['0000', '0010']
    #training_objects = [os.path.join(FLAGS.data_root,FLAGS.dataset,o) for o in training_objects]
    ##FLAGS.network = 'inception'
    #FLAGS.preloading = False
    #FLAGS.scale = 0.1

    #print training_objects
    FLAGS.data_type='grouped'
    print FLAGS.sample
    trainingset = prepare_data_general(training_objects)
    
    #trainingset = prepare_data_list(training_objects)
    print trainingset[0][0].shape
    window_sizes=[20]
    batch_sizes=[5]
    FLAGS.fc_only = False
    inds_r = pick_random_windows(trainingset, window_sizes, batch_sizes)
    inds = pick_sliding_windows(trainingset, window_sizes, batch_sizes)
    print 'inds_r ', len(inds_r),', ', len(inds_r[0]), ', ', inds_r[0][0].shape
    print 'inds ', len(inds),', ', len(inds[0]), ', ', inds[0][0].shape
    data, labels = copy_windows_from_data(trainingset, window_sizes[0], inds[0][0])
    print data.shape
    print labels.shape
    print data[0][0]
    import pdb; pdb.set_trace()
    
    
    
    #window_indices = pick_random_windows(trainingset, window_sizes, batch_sizes)
    #data, targets = copy_windows_from_data(trainingset, 4, window_indices[0][0])
    #print 'windowindices: ',len(window_indices),'',len(window_indices[0]),' ',window_indices[0][0].shape
    #print 'data: ',data.shape
    
    #print type(trainingset)
    
    #training_data, training_labels = prepare_data(training_objects, sample_step=100, feature_type='both')
    #print "labels: "+str(training_labels[1,0])
    #print training_labels
    #print training_data.shape
    #print training_labels.shape
    
    #validate_data, validate_labels = prepare_data(validate_object)
    #test_data, test_labels = prepare_data(test_object)

        
    
