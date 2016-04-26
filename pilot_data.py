 #
 # This code is used by pilot_train for getting the data and the ground truth from the images
 #
 
import os
import re
import logging
import scipy.io as sio
import numpy as np
import tensorflow as tf
from os import listdir
from os.path import isfile,join


class DataError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
 
def extract_features(filename):
    """Extract the PCNN features from a mat file into an ndarray [features of frame 0, frame 1, frame 2, ...] """
    print 'Extracting '+str(filename)
    data = sio.loadmat(filename)
    feats = data['features'][0,0][1]
    if feats.shape[1]!=4096:
        loggin.warning("dimensions mismatch, should be 4096")
    return feats
    
def get_labels(directoryname):
    """This functions extract the ground truth from the images in the RGB directory defined by the directoryname
    and return an array with labels for each frame"""
    directoryname = join(directoryname,'RGB')
    print 'Get labels from '+str(join(directoryname))
    if not os.path.isdir(directoryname):
        raise DataError("directory does not exist: "+str(directoryname))
        return -1
    files = [f for f in listdir(directoryname) if isfile(join(directoryname,f))]
    #obtain the label indicated after 'gt' from the names of the files
    labels = [re.search(r'gt\d',name).group(0)[2] for name in sorted(files)] 
    return labels

def one_hot_labels(labels):
    """
    create labels [1], [3] into labels [0100], [0001].
    """
    # get maximum ~ number of columns
    mx = int(max(labels))
    oh_labels = np.zeros([len(labels), mx+1])
    #print oh_labels.shape
    for i in range(len(labels)):
        oh_labels[i,int(labels[i])]=1
    return oh_labels
    
def get_objects(dataset):
    # given a dataset read out the train, validate and test_set.txt lists
    # return 3 lists of objects
    directory = '/esat/qayd/kkelchte/simulation/'+dataset+'/'
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
    # cut the newline and _one_cw at the end
    list_o = [l[:-8] for l in list_o]
    # cut the path, keep the name
    list_o = [l[len(directory):] for l in list_o]
    return list_o

def prepare_data(data_objects, sample_step=1,feature_type='app', dataset='data'):
    """
    Create batches for the listed data objects by obtaining the features and the labels.
    Creating different batches of the same size
    """
    
    # get training data 
    data = []
    labels = []
    min_steps = 100000
    
    if feature_type == 'both':
        feature_types = ['app', 'flow']
    else:
        feature_types = [feature_type]
    
    for data_object in data_objects:
        directory = '/esat/qayd/kkelchte/simulation/'+dataset+'/'+data_object+'_one_cw'
        labels_obj = one_hot_labels(get_labels(directory))
        
        minfeat = labels_obj.shape[0]# see that labels, app and flow features are from the same number of frames
        feats_obj = []        
        for ftype in feature_types: #loop over feature types 'app' and 'flow' or one of them
            feats = extract_features(directory+'/cnn_features/'+ftype+'_'+data_object+'_one_cw.mat')
            feats_obj.append(feats)
            #todo check and adapt if app and flow features dont have the same number of frames
            #if minfeat > feats.shape[0]:
            #    minfeat = feats.shape[0]
        
        #check here number of frames between app - flow - labels that you take minimum
        if 'flow' in feature_types:
            feats_obj[0] = feats_obj[0][0:-1] #shorten rgb list
            labels_obj = labels_obj[0:-1]
            
        #for ftype in feature_types:
        #    if feats_obj[ftype].shape[0] != minfeat: feats_obj[ftype]=feats_obj[ftype][]
        #import pdb; pdb.set_trace()
        feats_obj = np.concatenate(feats_obj,1) #concatenate the list of features in the second dimension
                
        print "shape features:",feats_obj.shape
        
        if not labels_obj.shape[0]==feats_obj.shape[0]:
            raise DataError("labels vs features mismatch: "+str(labels_obj.shape[0])+"!="+str(feats_obj.shape[0]))
        
        data.append(feats_obj)
        labels.append(labels_obj)
        if min_steps > labels_obj.shape[0]:#get number of frames of shortest movie
            min_steps = labels_obj.shape[0]   

    #import pdb; pdb.set_trace()
    if sample_step>=min_steps:
        raise DataError("downsampling is larger than total data set "+str(sample_step)+">="+str(min_steps))
        
    # resize according to num_steps of shortest video
    # and downsample in time to get a frame every 10 or 100 frames
    ds_data = np.zeros((len(data),int(min_steps/sample_step)+1,data[0].shape[1]))
    ds_labels = np.zeros((len(data), int(min_steps/sample_step)+1, labels[0].shape[1]))
    
    for b in range(len(data)):
        j=0
        for i in range(min_steps):
            if (i%sample_step) == 0:
                #import pdb; pdb.set_trace()
                ds_data[b,j]= data[b][i]
                ds_labels[b,j]=labels[b][i]
                j=j+1
    
    return ds_data, ds_labels

    #return data, labels

#####################################################################

def inputs(data_object, data_dir, batch_size):
    """Read images of data_objects from data_dir and
    return in batches according to batch_size
    Args:
        data_object: list of objects
        data_dir: main directory to find data
        batch_size: number of images per batch
    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """
    


#####################################################################
if __name__ == '__main__':
    print "-----------------------------"
    print "Run test module of pilot_data"
    print "-----------------------------"
    
    dataset='generated'
    training_objects, validate_objects, test_objects = get_objects(dataset)
    
    print validate_objects
    print test_objects
    
    #training_data, training_labels = prepare_data(training_objects, sample_step=100, feature_type='both')
    #print "labels: "+str(training_labels[1,0])
    #print training_labels
    #print training_data.shape
    #print training_labels.shape
    
    #validate_data, validate_labels = prepare_data(validate_object)
    #test_data, test_labels = prepare_data(test_object)

        
    
