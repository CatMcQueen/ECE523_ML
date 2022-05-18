from PIL import Image
import os
import numpy as np

folders = os.listdir('.')
for folder in folders:
  newfolder = 'small/test/' + folder
  if((folder[-1] == 'f' or folder[-1] == 'm') ):
    os.mkdir(newfolder)
    for filename in os.listdir(folder):
      if filename.endswith('tif'):
         infile  = folder + '/' + filename
         rotations = [0, 90, 180, 270]
         im      = Image.open(infile)
         for i in range(10):
           widthval   = np.random.randint(1100, 2880)
           heightval  = np.random.randint(900, 2160)
           startw = np.random.randint(0, (2880-widthval))
           starth = np.random.randint(0, (2160-heightval))
           stopw = startw + widthval
           stoph = starth + heightval
           area = (startw, starth, stopw, stoph)
           image = im.crop(area)
           idx     = np.random.randint(0,4)
           idx2    = np.random.randint(0,30)
           imsmall = image.resize((224, 224))
           # then do random flips and stuff
           imagei   = imsmall.rotate(rotations[idx])
           if (idx2 % 3 == 0):
             imagei = imagei.transpose(Image.FLIP_TOP_BOTTOM)
           if (idx2 % 4 == 0):     
             imagei = imagei.transpose(Image.FLIP_LEFT_RIGHT) 
 
           fileval = str(1 + i)
           #outfile = 'small/' + folder + '_' + filename[:-4] + '_' + str(idx) + '.png'
           outfile = 'small/test/' + folder + '/'+ filename + '_' + fileval + '.png'
           outimg  = imagei.save(outfile)

