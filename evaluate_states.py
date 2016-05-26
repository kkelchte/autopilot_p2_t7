#Plot some nice plots of the innerstates after a training period

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

plt.close('all')

logfolder='/esat/qayd/kkelchte/tensorflow/lstm_logs/'
run='big_lr_0001_size_10_net_logits_app'
data=sio.loadmat(logfolder+run+'/trainstates_100.mat')
states=data['states'][0]
targets=data['targets']
targets = np.argmax(targets, axis=1)
hsize = 10
layers = 2
num_steps  = states.shape[0]/(hsize * layers * 2)
width = 0.2

data_states = np.zeros((layers, hsize, num_steps))
data_outputs = np.zeros((layers, hsize, num_steps))

#loop over steps to rearrange the data
for step in range(num_steps):
	#data related to 1 step
	step_data = states[step*layers*hsize*2 : (step+1)*layers*hsize*2]
	for layer in range(layers):
		#data of this step related to current layer
		layer_data = step_data[layer*hsize*2:(layer+1)*hsize*2]
		state = layer_data[0:hsize]
		output = layer_data[hsize:]
		data_states[layer,:,step] = state
		data_outputs[layer,:,step] = output

#
#display a very raw overview of the 2 layers and the steps in x axis,
#different plots show different units
#figure 1: state, figure 2: output
fig, ax = plt.subplots(hsize, sharex=True, sharey=True)
for u in range(hsize):
	ax[u].set_title('hidden unit: '+str(u+1))
	ax[u].bar(np.arange(num_steps), data_states[0,u,:], width, color='r')
	ax[u].bar(np.arange(num_steps)+width, data_states[1,u,:], width, color='y')
#plt.setp([a.get_xticklabels() for a in ax], visible=False)
#plt.setp([a.get_yticklabels() for a in ax], visible=False)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.3)
#plt.show()
#import pdb;pdb.set_trace()
fig, ax = plt.subplots(hsize, sharex=True, sharey=True)
for u in range(hsize):
	ax[u].set_title('hidden unit: '+str(u+1))
	ax[u].bar(np.arange(num_steps), data_outputs[0,u,:], width, color='r')
	ax[u].bar(np.arange(num_steps)+width, data_outputs[1,u,:], width, color='y')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.3)

#plt.show()

#calculate average over different phases 0-1-2-1-2-1-2-1-2-1-3
average_states = []
average_outputs = []
phases=[0,1,2,1,2,1,2,1,2,1,3]
step=0
for phase in phases:
    average_state = np.zeros((layers,hsize))
    average_output = np.zeros((layers,hsize))
    phase_len = 0
    while (phase == targets[step]) :
        average_state = average_state+data_states[:,:,step]
        average_output = average_output+data_outputs[:,:,step]
        step=step+1
        phase_len=phase_len+1
        print average_state
        if step >= targets.shape[0]: break;
	average_state = average_state/phase_len
    average_output = average_output/phase_len
    average_states.append(average_state)
    average_outputs.append(average_output)

average_states = np.concatenate([average_states])
average_outputs = np.concatenate([average_outputs])
#import pdb; pdb.set_trace()
fig, ax = plt.subplots(hsize, sharex=True, sharey=True)
for u in range(hsize):
	ax[u].set_title('hidden unit: '+str(u+1))
	ax[u].bar(np.arange(len(phases)), average_states[:,0,u], width, color='r')
	ax[u].bar(np.arange(len(phases))+width, average_states[:,1,u], width, color='y')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.3)

fig, ax = plt.subplots(hsize, sharex=True, sharey=True)
for u in range(hsize):
	ax[u].set_title('hidden unit: '+str(u+1))
	ax[u].bar(np.arange(len(phases)), average_outputs[:,0,u], width, color='r')
	ax[u].bar(np.arange(len(phases))+width, average_outputs[:,1,u], width, color='y')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.3)

plt.show()

#
