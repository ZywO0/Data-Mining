#--
# p5.py
# apply straight line Hough transform to image
# @author: letsios, sklar
# @created: 28 Jan 2021
#--

import numpy as np
import imageio
import matplotlib.pyplot as plt
import skimage.color as color 
import skimage.feature as feature
import skimage.transform as transform

#-read image from a file into an image object named 'im'
im = imageio.imread( '../data/green-bridge.jpg' )

#-convert the image to greyscale
img = color.rgb2gray( im )

#-perform Canny edge detection
edges = feature.canny( img )

#-apply classic straight-line Hough transform
lines = transform.probabilistic_hough_line( edges, threshold=10, line_length=100, line_gap=3 )

#-plot the results
fig, (ax0, ax1, ax2, ax3) = plt.subplots( nrows=1, ncols=4, figsize=(8, 3), sharex=True, sharey=True )

ax0.imshow( im )
ax0.axis( 'off' )
ax0.set_title( 'original image' )

ax1.imshow( img, cmap=plt.cm.gray, interpolation='nearest' )
ax1.axis( 'off' )
ax1.set_title( 'greyscale image' )

ax2.imshow( edges, cmap=plt.cm.gray, interpolation='nearest' )
ax2.axis( 'off' )
ax2.set_title( 'canny edges' )

for line in lines:
    p0, p1 = line
    ax3.plot(( p0[0], p1[0] ), ( p0[1], p1[1] ))
ax3.set_xlim(( 0, img.shape[1] ))
ax3.set_ylim(( img.shape[0], 0 ))
ax3.set_title( 'Probabilistic Hough' )

fig.tight_layout()

plt.savefig( '../images/myhough.png' )
plt.show()
