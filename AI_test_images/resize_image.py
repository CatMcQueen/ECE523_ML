from PIL import Image
import os
import numpy as np

folders = os.listdir('.')
for folder in folders:
  newfolder = 'train/' + folder
  if((folder[-1] == 'f' or folder[-1] == 'm') ):
    os.mkdir(newfolder)
    for filename in os.listdir(folder):
      if filename.endswith('tif'):
         infile  = folder + '/' + filename
         image   = Image.open(infile)
         imsmall = image.resize((224, 224))
         filenm  = filename[:-4]
         outfile = 'train/' + folder + '/' + filenm + '.png'
         outimg  = imsmall.save(outfile)
         
