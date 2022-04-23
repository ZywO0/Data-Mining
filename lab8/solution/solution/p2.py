#--
# p2.py
# demonstrates canny edge detection on an image read from a file
# based on http://scikit-image.org/docs/stable/auto_examples/edges/plot_canny.html#sphx-glr-auto-examples-edges-plot-canny-py
# @author: letsios, sklar
# @created: 28 Jan 2021
#--

import numpy as np
import imageio
import matplotlib.pyplot as plt
import skimage.color as color 
import skimage.feature as feature

#-read image from a file into an image object named 'im'
im = imageio.imread( '../data/strawberry.jpg' )

#-convert the image to greyscale for canny edge detection
img = color.rgb2gray( im )

#-detect edges using Canny algorithm for two values of sigma
edges1 = feature.canny( img, sigma=1 ) #default
edges2 = feature.canny( img, sigma=3 )

#-plot the results
fig, (ax0, ax1, ax2, ax3) = plt.subplots( nrows=1, ncols=4, figsize=(8, 3), sharex=True, sharey=True )

ax0.imshow( im )
ax0.axis( 'off' )
ax0.set_title( 'original image' )

ax1.imshow( img, cmap=plt.cm.gray, interpolation='nearest' )
ax1.axis( 'off' )
ax1.set_title( 'greyscale image' )

ax2.imshow( edges1, cmap=plt.cm.gray, interpolation='nearest' )
ax2.axis( 'off' )
ax2.set_title( 'Canny filter, $\sigma=1$' )

ax3.imshow( edges2, cmap=plt.cm.gray, interpolation='nearest' )
ax3.axis( 'off' )
ax3.set_title( 'Canny filter, $\sigma=3$' )

fig.tight_layout()

plt.savefig( '../images/mycanny.png' )
plt.show()
