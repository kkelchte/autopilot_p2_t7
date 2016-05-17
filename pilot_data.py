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
import time


class DataError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
 
def extract_features(filename):
    """Extract the PCNN features from a mat file into an ndarray [features of frame 0, frame 1, frame 2, ...] """
    #print 'Extracting '+str(filename)
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


def prepare_data(data_objects, sample_step=1,feature_type='app', dataset='data', normalized=False):
    """
    Create batches for the listed data objects by obtaining the features and the labels.
    Creating different batches of the same size
    []
    Used by prepare_data_list and pilot_train
    returns a 3D array with first dimension the index of the movie, [movie, frame, feature]
    """
    
    # get data 
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
        if 'flow' in feature_types:
            feats_obj[0] = feats_obj[0][0:-1] #shorten app list
            labels_obj = labels_obj[0:-1]
            
        #check here number of frames between app - flow - labels that you take minimum
        #for ftype in feature_types:
        #    if feats_obj[ftype].shape[0] != minfeat: feats_obj[ftype]=feats_obj[ftype][]
        feats_obj = np.concatenate(feats_obj,1) #concatenate the list of features in the second dimension
        #print "shape features:",feats_obj.shape
        
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
                ds_data[b,j]= data[b][i]
                ds_labels[b,j]=labels[b][i]
                j=j+1
    #Normalize the subsampled features if necessary
    if normalized:
        norm_data = np.zeros(ds_data.shape)
        data = sio.loadmat('mean_variance_'+feature_type+'.mat')
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

def prepare_data_list(data_objects, sample_step=1,feature_type='app', dataset='generated', normalized=False):
    """
    Returns a list of tuples containing an array of features and an array of labels for each data object.
    Mainly used for validation and test set as there is no unrolled network, so the sizes dont have to be grouped.
    Used by prepare_data_grouped for reading in all data
    """
    object_list = []
    for d_object in data_objects:
        print "object: ",d_object
        object_data, object_labels = prepare_data([d_object], sample_step=sample_step, feature_type=feature_type, dataset=dataset, normalized=normalized)
        object_tuple = (object_data, object_labels)
        object_list.append(object_tuple)
    return object_list

def prepare_data_grouped(data_objects, sample_step=1,feature_type='app', dataset='generated', normalized=False):
    """
    Create a list of tuples. Each tuple contains a data and label. 
    """
    #list and sort the data according to the length
    all_data = prepare_data_list(data_objects, sample_step=sample_step, feature_type=feature_type, dataset=dataset, normalized=normalized)
    
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
    split_indices = [differences[i][1] for i in range(len(differences)) if differences[i][0]>(50/sample_step)]
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

def calculate_mean_variance(data_objects, dataset = 'generated',feature_type='both'):
    """
    go through all data_objects and calculate the mean and variance.
    Args:
        data_objects: list of objects in dataset
        dateset: the folder in qayd/kkelchte/simulation where the objects can be found
        feature_type: appearance app, flow or both correspond for the .mat files
    """
    #Define number of total frames in data_objects
    #total = 0
    #for data_object in data_objects:
    #    directory = '/esat/qayd/kkelchte/simulation/'+dataset+'/'+data_object+'_one_cw'
    #    for ftype in feature_types: #loop over feature types 'app' and 'flow' or one of them
    #        total = total + extract_features(directory+'/cnn_features/'+ftype+'_'+data_object+'_one_cw.mat').shape[0]
    #print 'Total: ', total
    # calculate the mean of all feature vectors over all data objects
    if feature_type == 'both':
        feature_types = ['app', 'flow']
    else:
        feature_types = [feature_type]
    means = []
    nums  = []
    for ftype in range(len(feature_types)): #loop over feature types 'app' and 'flow' or one of them
        mean = np.zeros(4096, dtype=np.float64)
        num = 0 #keep track of number of features
        for data_object in data_objects:
            directory = '/esat/qayd/kkelchte/simulation/'+dataset+'/'+data_object+'_one_cw'
            feats = extract_features(directory+'/cnn_features/'+feature_types[ftype]+'_'+data_object+'_one_cw.mat')
            #print 'feats: ',feats[0]
            mean = mean + feats.sum(axis=0)
            num = num + feats.shape[0]
        mean = mean / num
        means.append(mean)
        nums.append(num)
        print 'mean: ', mean
    # calculate the variance of all feature vectors over all data objects
    variances = []
    for ftype in range(len(feature_types)): #loop over feature types 'app' and 'flow' or one of them
        variance = np.zeros(4096, dtype=np.float64)
        for data_object in data_objects:
            directory = '/esat/qayd/kkelchte/simulation/'+dataset+'/'+data_object+'_one_cw'
            feats = extract_features(directory+'/cnn_features/'+feature_types[ftype]+'_'+data_object+'_one_cw.mat')
            diff = np.zeros(4096, dtype=np.float64)
            for feat in feats:
                diff = diff + (feat - mean)**2
            variance = variance + diff/nums[ftype]
            
        variances.append(variance)
        print 'variance: ', variance
    mean = np.concatenate(means)
    #import pdb; pdb.set_trace()
    variance = np.concatenate(variances)
    info = {"mean":mean, "variance":variance}
    sio.savemat('mean_variance_'+feature_type+'.mat', info, appendmat=False)
    
    
    
#####################################################################

if __name__ == '__main__':
    print "-----------------------------"
    print "Run test module of pilot_data"
    print "-----------------------------"
    dataset='generated'
    #training_objects, validate_objects, test_objects = get_objects(dataset)
    training_objects=['modeldaa','modelbbc','modelabe','modelbae','modelacc','modelbca']
    #training_objects=['modeldaa']
    #calculate_mean_variance(data_objects = training_objects,feature_type='both')
    result_data, result_labels =prepare_data(training_objects, feature_type='both', dataset='generated', normalized=True)
    #time.sleep(10)
    print result_data.shape
    print result_data
    #mylist=prepare_data_grouped(data_objects, feature_type='app', sample_step=8)
    #print type(mylist)

    #training_objects, validate_objects, test_objects = get_objects(dataset)
    #testset = prepare_data_list(test_objects, feature_type='both')
    #validationset = prepare_data_list(validate_objects, feature_type='both')
    #trainingset = prepare_data_grouped(training_objects, feature_type='app', sample_step=16)
    
    #print type(trainingset)
    
    #training_data, training_labels = prepare_data(training_objects, sample_step=100, feature_type='both')
    #print "labels: "+str(training_labels[1,0])
    #print training_labels
    #print training_data.shape
    #print training_labels.shape
    
    #validate_data, validate_labels = prepare_data(validate_object)
    #test_data, test_labels = prepare_data(test_object)

        
    
