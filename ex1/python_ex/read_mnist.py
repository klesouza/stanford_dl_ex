from struct import unpack
import gzip
from numpy import zeros, uint8, array

def get_labeled_data(imagefile, labelfile, specif_labels=None):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    # Open the images with gzip in read binary mode
    images = gzip.open(imagefile, 'rb')
    labels = gzip.open(labelfile, 'rb')

    # Read the binary data

    # We have to get big endian unsigned int. So we need '>I'

    # Get metadata for images
    images.read(4)  # skip the magic_number
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    # Get metadata for labels
    labels.read(4)  # skip the magic_number
    N = labels.read(4)
    N = unpack('>I', N)[0]

    if number_of_images != N:
        raise Exception('number of labels did not match the number of images')

    # Get the data
    x = []#zeros((N, rows, cols), dtype=uint8)  # Initialize numpy array
    y = []#zeros((N, 1), dtype=uint8)  # Initialize numpy array
    j=0
    for i in range(N):
        #if i % 1000 == 0:
            #print("i: %i" % i)
        tmp_label = labels.read(1)
        val = unpack('>B', tmp_label)[0]
        if specif_labels is None or val in specif_labels:
            y.append(val)
            x.append(zeros((rows*cols), dtype=uint8))
            for row in range(rows*cols):
                tmp_pixel = images.read(1)  # Just a single byte
                tmp_pixel = unpack('>B', tmp_pixel)[0]
                x[j][row] = tmp_pixel
            j +=1
    y = array(y, dtype=uint8)
    x = array(x, dtype=uint8)
    x = array(x[0:y.shape[0],:], dtype=uint8) 
    return (x, y)

from pylab import imshow, show, cm


def view_image(image, label=""):
    """View a single image."""
    print("Label: %s" % label)
    imshow(image, cmap=cm.gray)
    show()