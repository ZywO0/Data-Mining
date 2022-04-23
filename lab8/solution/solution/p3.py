#--
# p3.py
# compute convex hull in image
# @author: letsios, sklar
# @created: 28 Jan 2021
#--

import numpy as np
import imageio
import matplotlib.pyplot as plt
import skimage.color as color 
import skimage.filters as filters
import skimage.util as util
import skimage.morphology as morphology


#-read image from a file into an image object named 'im'
im = imageio.imread( '../data/strawberry.jpg' )

#-convert the image to greyscale
img = color.rgb2gray( im )

#-convert the image to B&W
threshold = filters.threshold_otsu( img )
print('Otsu method threshold = ', threshold)
binary_img = img > threshold

#-invert the image
inverted_binary_img = util.invert( binary_img )

#-compute convex hull
hull = morphology.convex_hull_image( inverted_binary_img )

#-plot the results
fig, (ax0, ax1, ax2, ax3) = plt.subplots( nrows=1, ncols=4, figsize=(8, 3), sharex=True, sharey=True )

ax0.imshow( im )
ax0.axis( 'off' )
ax0.set_title( 'original image' )

ax1.imshow( binary_img, cmap=plt.cm.gray, interpolation='nearest' )
ax1.axis( 'off' )
ax1.set_title( 'b&w image' )

ax2.imshow( inverted_binary_img, cmap=plt.cm.gray, interpolation='nearest' )
ax2.axis( 'off' )
ax2.set_title( 'inverted image' )

ax3.imshow( hull, cmap=plt.cm.gray, interpolation='nearest' )
ax3.axis( 'off' )
ax3.set_title( 'convex hull' )

fig.tight_layout()

plt.savefig( '../images/myhull.png' )
plt.show()
