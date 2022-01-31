import os
import numpy as np

pattern_path='/raid/pan/pattern_dataset/'
ori_path='/raid/pan/ori_dataset/'
save_path='/home/pan/Desktop/vit_rec/my0812/'

#mirflickr25k
train_ori_mk_files=[]
val_ori_mk_files=[]
train_pattern_mk_files=[]
val_pattern_mk_files=[]

files=os.listdir(pattern_path+'mirflickr25k_1600/train/')
files.sort()
for file in files:
    train_pattern_mk_files.append(pattern_path+'mirflickr25k_1600/train/'+file)
    train_ori_mk_files.append(ori_path + 'mirflickr25k/train/' + file[:-3]+'jpg')

files=os.listdir(pattern_path+'mirflickr25k_1600/val/')
files.sort()
for file in files:
    val_pattern_mk_files.append(pattern_path+'mirflickr25k_1600/val/'+file)
    val_ori_mk_files.append(ori_path + 'mirflickr25k/val/' + file[:-3]+'jpg')




#fruits,PetImages
ori_fP_files = []
pattern_fP_files = []
for i in range(2):
    if i==0:
        ori_folder_name='fruits_modified/'
        pattern_folder_name = 'fruits_1600/fruits_modified/'
    else:
        ori_folder_name='PetImages/'
        pattern_folder_name = 'PetImages_1600/'

    folders=os.listdir(pattern_path+pattern_folder_name)
    folders.sort()
    for folder in folders:
        files=os.listdir(pattern_path+pattern_folder_name+folder+'/')
        files.sort()
        for file in files:
            if os.path.exists(ori_path+ori_folder_name+folder+'/'+file[:-3]+'jpg'):
                ori_fP_files.append(ori_path+ori_folder_name+folder+'/'+file[:-3]+'jpg')
                pattern_fP_files.append(pattern_path + pattern_folder_name + folder + '/' + file)
            if os.path.exists(ori_path + ori_folder_name + folder + '/' + file[:-3] + 'JPEG'):
                ori_fP_files.append(ori_path + ori_folder_name + folder + '/' + file[:-3] + 'JPEG')
                pattern_fP_files.append(pattern_path + pattern_folder_name + folder + '/' + file)

train_patterns=train_pattern_mk_files+pattern_fP_files
train_targets=train_ori_mk_files+ori_fP_files

val_patterns=val_pattern_mk_files
val_targets=val_ori_mk_files

np.save(save_path+'train_patterns.npy',train_patterns)
np.save(save_path+'train_targets.npy',train_targets)
np.save(save_path+'val_patterns.npy',val_patterns)
np.save(save_path+'val_targets.npy',val_targets)
