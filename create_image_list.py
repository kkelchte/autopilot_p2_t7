#Get from a dataset in /esat/qayd/kkelchte/simulation all the RGB and/or flow images.
import os, sys, re
from os import listdir
from os.path import isdir, join, isfile
###

chosen_set='generated' #generated_train or generated_test
data_types=['RGB']#, 'flow'] #or flow or both

###
dest= "imglist_"+chosen_set
#for t in data_types:
#    dest=dest+"_"+t
dest=dest+".txt"

if os.path.exists(dest):
    os.remove(dest)   
dest_file = open(dest, 'w')
movies_dir="/esat/qayd/kkelchte/simulation/"+chosen_set

movies=[mov for mov in sorted(listdir(movies_dir)) if (isdir(join(movies_dir, mov)) and mov != "cache")]
for mov in movies:
    #for dt in data_types:
    rgb_images=[img for img in sorted(listdir(join(movies_dir,mov,'RGB'))) if isfile(join(movies_dir,mov,'RGB',img))]
    for img in rgb_images:
        dest_file.write(join(movies_dir,mov,'RGB',img)+' ')
        dest_file.write(join(movies_dir,mov,'flow',img)+' ')
        match=re.search(r'-gt\d', img)
        label=match.group(0)[3]
        dest_file.write(label+'\n')
            
dest_file.close()
