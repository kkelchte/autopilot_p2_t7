import random
import shutil
import os, sys, re
from os import listdir
from os.path import isdir, join, isfile
#from 108 movies make a train-validate-test split of 80-15-13 movies
num_test = 13
num_val = 15
root = "/esat/qayd/kkelchte/simulation/generated"
val_f = open(join(root,"val_set.txt"), 'w')
test_f = open(join(root,"test_set.txt"), 'w')
train_f = open(join(root,"train_set.txt"), 'w')

total_mov = []
for i in ['a','b','c','d','e','f']:
    for j in ['a','b','c','d','e','f']:
        for k in ['a','c','e']:
            total_mov.append(i+j+k) 
#print total_mov

test_mov = []
for i in range(num_test):
    sel = random.randint(1,len(total_mov)-1)
    xyz = total_mov.pop(sel)
    test_mov.append(xyz)
    test_f.write(join(root,"model"+xyz+"_one_cw\n"))
                 
val_mov = []
for i in range(num_val):
    sel = random.randint(1,len(total_mov)-1)
    xyz = total_mov.pop(sel)
    val_mov.append(xyz)
    val_f.write(join(root,"model"+xyz+"_one_cw\n"))

for xyz in total_mov:
    train_f.write(join(root,"model"+xyz+"_one_cw\n"))
    
test_f.close()
val_f.close()
train_f.close()

print "done"
#print test_mov
#print val_mov
#print total_mov

#print len(test_mov)
#print len(val_mov)
#print len(total_mov)

#for xyz in test_mov:
#    shutil.copytree(root+"/generated/model"+xyz+"_one_cw", root+"/generated_test/model"+xyz+"_one_cw")
    
#for xyz in total_mov:
#    shutil.copytree(root+"/generated/model"+xyz+"_one_cw", root+"/generated_train/model"+xyz+"_one_cw")

#for xyz in val_mov:
#    shutil.copytree(root+"/generated/model"+xyz+"_one_cw", root+"/generated_val/model"+xyz+"_one_cw")



    
    

