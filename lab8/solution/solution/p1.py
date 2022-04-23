#--
# p1.py
# basic image handling in python
# @author: letsios, sklar
# @created: 28 Jan 2021
#--

import numpy
import imageio
import matplotlib.pyplot as plt

#-read an image from a file
im = imageio.imread('../data/pear.jpg')

#-im is a numpy array, so let's see how big it is
print(im.shape)  

#-plot the image
plt.imshow( im )
plt.show()
