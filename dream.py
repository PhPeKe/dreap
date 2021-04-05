import matplotlib.pyplot as plt
import keras.backend as K
import multiprocessing
import tensorflow as tf
import warnings
from keras.applications.vgg19 import VGG19
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import DepthwiseConv2D
import numpy as np
from scipy.optimize import minimize
import scipy.ndimage

from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from skimage.transform import pyramid_gaussian, rescale

import parameters as pm


import keras.backend as K
import keras
if K.backend() == 'tensorflow':
    keras.backend.set_image_data_format("channels_last")

use_gpu = True


session = tf.Session()
K.set_session(session)


def show_image(image):
    fig, ax = plt.subplots(figsize=(18, 15))
    ax.imshow(image)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def preprocess_image(image):
#    return preprocess_input(np.expand_dims(image.astype(K.floatx()), 0))
    return preprocess_input(np.expand_dims(image.astype(K.floatx()), 0))


def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def postprocess_image(image):
    image[:, :, :, 0] += 103.939
    image[:, :, :, 1] += 116.779
    image[:, :, :, 2] += 123.68
    return np.clip(image[:, :, :, ::-1], 0, 255).astype('uint8')[0]


original_image = pm.dImg
final_image = pm.dOut

original_image_array = imread(original_image)
filt = matlab_style_gauss2D(pm.filter_size, pm.filter_sigma)
#show_image(original_image_array)

# Ignore some warnings from scikit-image
warnings.simplefilter("ignore")

# Create gaussian pyramid
original_image_as_float = img_as_float(original_image_array)

pyramid = list(pyramid_gaussian(original_image_as_float, downscale=pm.downscale, max_layer=pm.max_layer, multichannel=True))

# Convert each image to unsigned byte (0-255)
for i, image in enumerate(pyramid):
    pyramid[i] = img_as_ubyte(pyramid[i])
    print('Image {}) Size: {}'.format(i, pyramid[i].shape))

convnet = VGG19(include_top=False, weights='imagenet')


layers = {
    'block5_conv1': 0.001,
    'block5_conv2': 0.001,
    #'block5_conv3': 0.002,
    #'block5_conv4': 0.005,
}



####################

kernel_weights = matlab_style_gauss2D(shape=pm.filter_size, sigma=pm.filter_sigma)
in_channels = 1  # the number of input channels
kernel_weights = np.expand_dims(kernel_weights, axis=-1)
kernel_weights = np.repeat(kernel_weights, in_channels, axis=-1) # apply the same filter on all the input channels
kernel_weights = np.expand_dims(kernel_weights, axis=-1)

g_layer = DepthwiseConv2D(pm.filter_size, use_bias=False, padding='same', depth_multiplier=1)

input_tensor = convnet.layers[0].input

regularized_input = g_layer(input_tensor)

kernel_weights = np.concatenate([kernel_weights, kernel_weights, kernel_weights], axis=2)

g_layer.set_weights([kernel_weights])
g_layer.trainable = False

########################

image_l2_weight = 0.005

loss_tensor = 0.0

for layer, weight in layers.items():
    loss_tensor += (-weight * K.sum(K.square(convnet.get_layer(layer).output)))

loss_tensor += image_l2_weight * K.sum(K.square(input_tensor))

_loss_function = K.function(inputs=[input_tensor], outputs=[loss_tensor])

loss_gradient = K.gradients(loss=loss_tensor, variables=[input_tensor])
_gradient_function = K.function(inputs=[input_tensor], outputs=loss_gradient)


def loss(x, shape):
    return _loss_function([x.reshape(shape)])[0]


def gradient(x, shape):
    # TODO: smooth this x
    return _gradient_function([x.reshape(shape)])[0].flatten().astype(np.float64)


def process_image(image, iterations=1):

    for i in range(iterations):
        print('Iteration',i)
        # Perform optimization
        # Create bounds
        bounds = np.ndarray(shape=(image.flatten().shape[0], 2))
        bounds[:, 0] = -128.0
        bounds[:, 1] = 128.0
        # Initial value
        x0 = image.flatten()
        result = minimize(fun=loss,
                          x0=x0,
                          args=list(image.shape),
                          jac=gradient,
                          method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 1})
        image = np.copy(result.x.reshape(image.shape))
    image = postprocess_image(np.copy(result.x.reshape(image.shape)))

    #if pm.filter_dream:
    #    print('Filtering!')
    #    for layer in range(result.shape[2]):
    #        result[:, :, layer] = scipy.signal.convolve2d(result[:, :, layer], filt, mode='same')

    result = image
    print("result shape", result.shape)

    return result


processed_image = None

for i, image in enumerate(pyramid[::-1]):
    print('Processing pyramid image: {} {}'.format(len(pyramid)-i, image.shape))

    if processed_image is None:
        processed_image = process_image(preprocess_image(image), iterations=pm.iterations)
    else:
        h, w = image.shape[0:2]
        ph, pw = processed_image.shape[0:2]
        rescaled_image = rescale(processed_image, order=5, scale=(float(h)/float(ph), float(w)/float(pw),1))
        combined_image = img_as_ubyte((1.2*img_as_float(image) + 0.8*rescaled_image) / 2.0)
        processed_image = process_image(preprocess_image(combined_image), iterations=pm.iterations)

#        processed_image = scipy.ndimage.gaussian_laplace(processed_image, sigma=2)
imsave(final_image, processed_image)
