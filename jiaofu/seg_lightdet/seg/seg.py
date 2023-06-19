import shutil
import glob
import os
dir = '/root/rail/new/masks/*.png'
save = '/root/rail/new/img/'
for file in glob.glob(dir):
    filename = os.path.basename(file).split('_mask.png')[0]+'.jpg'
    savepath = os.path.join(save,filename)
    shutil.copy(file,savepath)






