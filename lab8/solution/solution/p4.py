#--
# p4.py
# find contours in image
# @author: letsios, sklar
# @created: 28 Jan 2021
#--

import numpy as np
import imageio
import matplotlib.pyplot as plt
import skimage.color as color 
import skimage.filters as filters
import skimage.measure as measure


#-read image from a file into an image object named 'im'
im = imageio.imread( '../data/green-bridge.jpg' )

#-convert the image to greyscale
img = color.rgb2gray( im )

#-find a good threshold value
threshold = filters.threshold_otsu( img )
print('Otsu method threshold = ', threshold)

#-find contours at the threshold value found above
contours = measure.find_contours( img, threshold )

#-plot the results
fig, (ax0, ax1, ax2) = plt.subplots( nrows=1, ncols=3, figsize=(8, 3), sharex=True, sharey=True )

ax0.imshow( im )
ax0.axis( 'off' )
ax0.set_title( 'original image' )

ax1.imshow( img, cmap=plt.cm.gray, interpolation='nearest' )
ax1.axis( 'off' )
ax1.set_title( 'greyscale image' )

for n, contour in enumerate( contours ):
    ax2.plot( contour[:,1], contour[:,0], 'k-', linewidth=2 )
ax2.axis( 'off' )
ax2.set_title( 'image contours' )

fig.tight_layout()

plt.savefig( '../images/mycontours.png' )
plt.show()
