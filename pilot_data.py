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
from os.path import isfile,join
import time
import random

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
tf.app.flags.DEFINE_string(
    "dataset", "generated",
    "pick the dataset in /esat/qayd/kkelchte/simulation from which your movies can be found. Options: data or generated.")
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
    "batch_length",1000,
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
    
class DataError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
 
def extract_features(filename, network='pcnn'):
    """Extract the features from a mat file into an ndarray [features of frame 0, frame 1, frame 2, ...] 
    Args:
        filename: the path pointing to a mat file keeping the features
        network: the type of features found in this file ==> need of different extraction
    Returns:
        ndarray object containing features of each frame
    """
    #print 'Extracting '+str(filename)
    #print 'network: ', network
    data = sio.loadmat(filename)
    if network == 'pcnn': 
        feats = data['features'][0,0][1]
        if feats.shape[1]!=4096:
            loggin.warning("dimensions mismatch with pcnn features, should be 4096")       
    elif network == 'inception': 
        feats = data['features']
        if feats.shape[1]!=2048:
                loggin.warning("dimensions mismatch with inception features, should be 2048")       
    elif (network == 'logits') or (network == 'logits_noisy'): 
        feats = data['features']
        if feats.shape[1]!=4:
                loggin.warning("dimensions mismatch with logits features, should be 4")
    elif network == 'stijn':
        data=data['gazebo_sim_dataset']
        feats = data[0,0]['labels']
        if feats.shape[1]!=4070:
                loggin.warning("dimensions mismatch with stijn features, should be 4070")
    return feats
   
def get_labels_from_directory(directoryname):
    """This functions extract the ground truth from the images in the RGB directory defined by the directoryname
    and return an array with labels for each frame"""
    directoryname = join(directoryname,'RGB')
    print 'Get labels from '+str(join(directoryname))
    if not os.path.isdir(directoryname):
        raise DataError("directory does not exist: "+str(directoryname))
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
        raise DataError("control file does not exist: "+str(filename))
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
    if FLAGS.network == 'inception':#and not FLAGS.feature_type == both
        labels=labels[range(1,len(list_c),2)]
    
    # use only the 2 numbers that are not 0.8 or 0 as labels 
    if FLAGS.short_labels:
    	labels=labels[:,[2,5]]
    return labels

def prepare_data(data_objects):
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
            dataset: the directory in the 'qayd/kkelchte/simulation/' path where the objects are found
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
        directory = '/esat/qayd/kkelchte/simulation/'+FLAGS.dataset+'/'+data_object
        #real labels from file or from RGB-directory
        if FLAGS.read_from_file:
            labels_obj = get_labels_from_file(directory)
        else:
            labels_obj = get_labels_from_directory(directory)
        if not FLAGS.one_hot:
            labels_obj = create_one_hot_labels(labels_obj)
            
        #import pdb; pdb.set_trace()
        minfeat = labels_obj.shape[0]# see that labels, app and flow features are from the same number of frames
        feats_obj = []        
        for ftype in feature_types: #loop over feature types 'app' and 'flow' or one of them
            if FLAGS.network == 'pcnn': 
                feats = extract_features(directory+'/cnn_features/'+ftype+'_'+data_object+'.mat', FLAGS.network)
            elif FLAGS.network == 'inception' or FLAGS.network == 'logits' or FLAGS.network == 'logits_noisy' or FLAGS.network == 'stijn': 
                feats = extract_features(directory+'/cnn_features/'+ftype+'_'+data_object+'_'+FLAGS.network+'.mat', FLAGS.network)
            else: raise DataError("network not known: ", FLAGS.network)
            feats_obj.append(feats)
        #flow is on 2RBG images ==> skip first label to be equal
        if 'flow' in feature_types:
            labels_obj = labels_obj[1:]
        if len(feature_types) == 2:
            feats_obj[0] = feats_obj[0][0:-1] #shorten app list to be equal to flow list
            
        feats_obj = np.concatenate(feats_obj,1) #concatenate the list of features in the second dimension
        #print "shape features:",feats_obj.shape
    
        if FLAGS.network == 'stijn':
            rgb_indices=[]
            #check out the features of value -1 and remove these features + labels from the list
            # Assume that if stijns depth estimate is used this is in front of the array of feature types
            for i in range(feats_obj.shape[0]):
                if not (feats_obj[i,0] == -1 and feats_obj[i,1] == -1 and feats_obj[i,2] == -1) and not (feats_obj[i,0] == 0 and feats_obj[i,1] == 0 and feats_obj[i,2] == 0):
                    rgb_indices.append(i)
            #print 'rgb indices: ',rgb_indices
            try:
	        feats_obj=feats_obj[rgb_indices,:]
	        labels_obj=labels_obj[rgb_indices,:]
    	    except IndexError:
    	        print 'IndexError: ',rgb_indices[-1],'>=',min([labels_obj.shape[0], feats_obj.shape[0]])
        
        #import pdb;pdb.set_trace()

        # make them equal lenght: might have missed out some label/feature        
        labels_obj = labels_obj[0:min(labels_obj.shape[0],feats_obj.shape[0])]
        feats_obj = feats_obj[0:min(labels_obj.shape[0],feats_obj.shape[0])]
        
        if not labels_obj.shape[0]==feats_obj.shape[0]:
            raise DataError("labels vs features mismatch: "+str(labels_obj.shape[0])+"!="+str(feats_obj.shape[0]))
        
        data.append(feats_obj)
        labels.append(labels_obj)
        if min_steps > labels_obj.shape[0]:#get number of frames of shortest movie
            min_steps = labels_obj.shape[0]   

    #import pdb; pdb.set_trace()
    if FLAGS.sample>=min_steps:
        raise DataError("downsampling is larger than total data set "+str(FLAGS.sample)+">="+str(min_steps))
        
    # resize according to num_steps of shortest video
    # and downsample in time to get a frame every 10 or 100 frames
    ds_data = np.zeros((len(data),int(min_steps/FLAGS.sample),data[0].shape[1]))
    ds_labels = np.zeros((len(data), int(min_steps/FLAGS.sample), labels[0].shape[1]))
    
    #loop over batches (in this implementation is this always 1 batch)
    for b in range(len(data)):
        j=0
        for i in range(min_steps):
            if (i%FLAGS.sample) == 0 and j<ds_data.shape[1]:
                ds_data[b,j]= data[b][i]
                ds_labels[b,j]=labels[b][i]
                j=j+1
    
    #import pdb;pdb.set_trace()
    #Normalize the subsampled features if necessary
    if FLAGS.normalized:
        norm_data = np.zeros(ds_data.shape)
        data = sio.loadmat('mean_variance_'+FLAGS.feature_type+"_"+FLAGS.network+'.mat')
        mean = data['mean']
        variance = data['variance']
        epsilon = 0.001
        for i in range(ds_data.shape[0]):
            for j in range(ds_data.shape[1]):
                norm_data[i,j,:]=(ds_data[i,j,:]-mean[0])/(np.sqrt(variance[0])+epsilon)
        ds_data = norm_data
        #import pdb; pdb.set_trace()
    return ds_data, ds_labels

    #return data, labels

def prepare_data_list(data_objects):
    """
    Returns a list of tuples contains (features, labels)_movie. Batchsize of features and labels = 1.
    features and labels have size [batchsize, numberframes, feature/label dimension]
    """
    object_list = []
    for d_object in data_objects:
        print "object: ",d_object
        object_data, object_labels = prepare_data([d_object])
        object_tuple = (object_data, object_labels)
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
    ### ! changed minimum difference from 50 to 100 !
    split_indices = [differences[i][1] for i in range(len(differences)) if differences[i][0]>(100/FLAGS.sample)]
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
    directory = '/esat/qayd/kkelchte/simulation/'+FLAGS.dataset+'/'
    print 'directory ', directory
    # read training objects from train_set.txt
    training_objects = get_list(directory,'train_set.txt')
    # read validate objects from validat_set.txt
    validate_objects = get_list(directory,'val_set.txt')
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
    #import pdb; pdb.set_trace()
    # cut the path, keep the name
    list_o = [os.path.basename(l) for l in list_o]
    return list_o

def pick_random_windows(data_list, window_sizes, batch_sizes):
    '''create batches of different windowsizes for training different unrolled models.
    The batches are picked randomly from all the datalist. The batch and window sizes are defined in the arguments.
    Args:
        data_list: the data list containing tuples with data and labels of batches (of equal length) [works with listed data?]
        windowsizes: a list with the windowsizes needed corresponding to the list of unrolled networks to be trained
        batchsizes: a list of batchsizes for each window set according to the scale factor depending on the size of your RAM and GPU mem
    Return:
        windowed_data: a nested list of lists of tuples with each tuple the according windowsize and batchsize as set by the arguments. [DEPRECATED]
        indices: a nested list of lists of tuples with each tuple the according indices for chosen group, chosen batch, chosen timestep. 
        			The data is picked randomly from the data list. The outer list corresponds to the number of windowsizes/models.
        			The innerlist is the amount of times a window batch is picked in order to correspond to 1 epoch. This depends of the amount of data.
        			Each element in the list is a array with sets of movie-indices. Every set of movie indices is an array of 3 numbers:
        			[tuple index, movie index, start index] tuple index corresponds to the tuple in the data list, movie index is the index in the batch of that tuple and start index is the position of the window.
        
    '''
    feature_size = data_list[0][0].shape[2]
    output_size = data_list[0][1].shape[2]
    #print 'feature_size: ', feature_size
    #print 'output_size: ', output_size
    
    ##preparation = define number of times the windowed-batched data should be iterated in one epoch
    if len(window_sizes) != len(batch_sizes):
    	raise SyntaxError('Number of windowsizes is not equal to number of batchsizes.')
    	
    b_num_frames = sum([ window_sizes[i] * batch_sizes[i] for i in range(len(window_sizes))])
    d_num_frames = sum([ data_list[t][0].shape[0]*data_list[t][0].shape[1] for t in range(len(data_list))])
    b_scale = max([1, int(d_num_frames/b_num_frames)]) #number of times the window-batches are repeated in one epoch
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

def copy_windows_from_data(data_list, window_size, indices):
    ''' create batch of window_size length
    Args:
        data_list: list of tuples containing data and labels each arrays of size [batchsize, framelen, featuresize]
        windowsize: the windowsize needed corresponding to the list of unrolled networks to be trained
        batchsize: batchsizes for the window set according to the scale factor depending on the size of your RAM and GPU mem
        indices: [tuple index, movie index, start index] tuple index corresponds to the tuple in the data list, movie index is the index in the batch of that tuple and start index is the position of the window.
    Returns:
         the data and labels according to the windowsize and batchsize as set by the arguments.
    '''
    batch_size = indices.shape[0]
    data = np.zeros((batch_size, window_size, data_list[0][0].shape[2]))
    labels = np.zeros((batch_size, window_size, data_list[0][1].shape[2]))
    for i in range(batch_size):
        tup_ind, mov_ind, start_ind = indices[i]
        data[i] = data_list[tup_ind][0][mov_ind][start_ind:start_ind+window_size][:]
        labels[i] = data_list[tup_ind][1][mov_ind][start_ind:start_ind+window_size][:]
    return data, labels
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
    FLAGS.dataset='../../../emerald/tmp/remote_images/tiny_set'
    #FLAGS.dataset='generated'
    FLAGS.sample = 1
    #training_objects, validate_objects, test_objects = get_objects()
    #testset = prepare_data_list(test_objects)
    #validationset = prepare_data_list(validate_objects)
    #trainingset = prepare_data_grouped(training_objects)
    trainingset = prepare_data_list(['0000'])
    
    window_sizes=[1]
    batch_sizes=[10]
    
    indices = pick_random_windows(trainingset, window_sizes, batch_sizes)
    import pdb; pdb.set_trace()
    #print type(trainingset)
    
    #training_data, training_labels = prepare_data(training_objects, sample_step=100, feature_type='both')
    #print "labels: "+str(training_labels[1,0])
    #print training_labels
    #print training_data.shape
    #print training_labels.shape
    
    #validate_data, validate_labels = prepare_data(validate_object)
    #test_data, test_labels = prepare_data(test_object)

        
    
