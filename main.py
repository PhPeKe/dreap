import os
import sys
import scipy.misc
from skimage import io
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from nst_utils import *
import numpy as np
import tensorflow as tf
from functions import *
import parameters as pm

# Specify pictures
cImg = pm.cImg
sImg = pm.sImg
print("\nPictures:\n",cImg, "\n",sImg)

# Set layers for style
standard = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

shallow = [
    ('conv1_1', 0.5),
    ('conv2_1', 0.5)]

late = [
    ('conv4_1', 0.5),
    ('conv5_1', 0.5)]

deep = [
    ('conv1_2', 0.4),
    ('conv2_2', 0.4),
    ('conv3_2', 0.5),
    ('conv4_2', 0.4),
    ('conv5_2', 0.4)]

deeper = [
    ('conv3_3', 0.33),
    ('conv4_3', 0.33),
    ('conv5_3', 0.33)]

deeeper = [
    ('conv3_4', 0.5),
    ('conv4_4', 0.5),
    ('conv5_4', 0.5)]



layers = [shallow, deep]
layernames = ["shallow", "deep"]

tf.reset_default_graph()


# Load pretrained model
print("Loading pretrained model...")
model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
print("Finished loading pretrained model")

for i,layer in enumerate(layers):
    STYLE_LAYERS = layer
    # Reset the graph

    # Start interactive session
    print("Starting session...")
    sess = tf.InteractiveSession()

    # Read in final content image
    print("Reading and resizing images")
    content_image = io.imread(cImg)
    content_image = reshape_and_normalize_image(content_image)

    # Read in findal style image
    style_image = io.imread(sImg)
    style_image = reshape_and_normalize_image(style_image)

    # Initialize generated image
    print("Initializing generated image")
    generated_image = generate_noise_image(content_image)


    # Assign the content image to be the input of the VGG model.
    print("Assigning content image...")
    sess.run(model['input'].assign(content_image))

    # Select the output tensor of layer conv4_2
    #out = model['conv4_1']
    out = model['conv4_2'] #['conv5_4']

    # Set a_C to be the hidden layer activation from the layer we have selected
    a_C = sess.run(out)

    # Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2']
    # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
    # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
    a_G = out

    # Compute the content cost
    J_content = compute_content_cost(a_C, a_G)

    # Assign the input of the model to be the "style" image
    sess.run(model['input'].assign(style_image))

    # Compute the style cost
    J_style = compute_style_cost(model, STYLE_LAYERS, sess)
    J = total_cost(J_content, J_style)

    # define optimizer
    optimizer = tf.train.AdamOptimizer(2.0)

    # define train_step
    train_step = optimizer.minimize(J)

    # Run the model
    parameters = [J, J_content, J_style, cImg, sImg, save_image]
    model_nn(sess, generated_image, model, train_step, parameters, num_iterations=pm.n_iter, interval=pm.interval, add=layernames[i])
