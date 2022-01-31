import os
import shutil
pattern_path='/raid/pan/pattern_dataset/'
ori_path='/raid/pan/ori_dataset/mirflickr25k/'


files=os.listdir(pattern_path+'mirflickr25k_1600/')
files.sort()
for i in range(len(files)):
    cur=pattern_path+'mirflickr25k_1600/'+files[i]
    if i<1000:
        dst=pattern_path+'mirflickr25k/val/'+files[i]
    else:
        dst=pattern_path+'mirflickr25k/train/'+files[i]
    shutil.move(cur, dst)
