from skimage import io
from nst_utils import *
from functions import *
import parameters as pm

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

prepare_pictures(pm.sImg, pm.cImg)

# Specify pictures
cImg = "input/cimg.png"
sImg = "input/simg.png"
print("\nPictures:\n", cImg, "\n", sImg)

# Set layers for style
standard = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

standard_conv1 = [
    ('conv1_1', 1.0),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

standard_conv2 = [
    ('conv1_1', 0.2),
    ('conv2_1', 1.0),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

standard_conv3 = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 1.0),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

standard_conv4 = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 1.0),
    ('conv5_1', 0.2)]

standard_conv5 = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 1.0)]

many = [
    ('conv1_1', 0.2),
    ('conv1_2', 0.2),
    ('conv2_1', 0.2),
    ('conv2_2', 0.2),
    ('conv3_1', 0.2),
    ('conv3_2', 0.2),
    ('conv3_3', 0.2),
    ('conv3_4', 0.2),
    ('conv4_1', 0.2),
    ('conv4_2', 0.2),
    ('conv4_3', 0.2),
    ('conv4_4', 0.2),
    ('conv5_1', 0.2),
    ('conv5_2', 0.2),
    ('conv5_3', 0.2),
    ('conv5_4', 0.2)]

shallow = [
    ('conv1_1', 0.5),
    ('conv2_1', 0.5)]

late = [
    ('conv4_1', 0.5),
    ('conv5_1', 0.5)]

deep = [
    ('conv1_2', 0.5),
    ('conv2_2', 0.6),
    ('conv3_2', 0.7),
    ('conv4_2', 0.6),
    ('conv5_2', 0.5)]

deep_flat = [
    ('conv1_2', 0.5),
    ('conv2_2', 0.5),
    ('conv3_2', 0.5),
    ('conv4_2', 0.5),
    ('conv5_2', 0.5)]

deep_low = [
    ('conv1_2', 0.2),
    ('conv2_2', 0.2),
    ('conv3_2', 0.2),
    ('conv4_2', 0.2),
    ('conv5_2', 0.2)]

deeper = [
    ('conv3_3', 0.33),
    ('conv4_3', 0.33),
    ('conv5_3', 0.33)]

deeeper = [
    ('conv3_4', 0.5),
    ('conv4_4', 0.5),
    ('conv5_4', 0.5)]

conv1_1 = [
    ('conv1_1', 0.2)]
conv1_2 = [
    ('conv1_2', 0.2)]
conv2_1 = [
    ('conv2_1', 0.2)]
conv2_2 = [
    ('conv2_2', 0.2)]
conv3_1 = [
    ('conv3_1', 0.2)]
conv3_2 = [
    ('conv3_3', 0.2)]
conv3_3 = [
    ('conv3_3', 0.2)]
conv3_4 = [
    ('conv3_4', 0.2)]
conv4_1 = [
    ('conv4_1', 0.2)]
conv4_2 = [
    ('conv4_2', 0.2)]
conv4_3 = [
    ('conv4_3', 0.2)]
conv4_4 = [
    ('conv4_4', 0.2)]
conv5_1 = [
    ('conv5_1', 0.2)]
conv5_2 = [
    ('conv5_2', 0.2)]
conv5_3 = [
    ('conv5_3', 0.2)]
conv5_4 = [
    ('conv5_4', 0.2)]

layerdict = {"deep": deep,
             "deeper": deeper,
             "deeeper": deeeper,
             "deepflat": deep_flat,
             "deeplow": deep_low,
             "standard": standard,
             "standard_conv1": standard_conv1,
             "standard_conv2": standard_conv2,
             "standard_conv3": standard_conv3,
             "standard_conv4": standard_conv4,
             "standard_conv5": standard_conv5,
             "many": many,
             "late": late,
             "shallow": shallow,
             "conv1_1": conv1_1,
             "conv1_2": conv1_2,
             "conv2_1": conv2_1,
             "conv2_2": conv2_2,
             "conv3_1": conv3_1,
             "conv3_2": conv3_2,
             "conv3_2": conv3_2,
             "conv3_2": conv3_2,
             "conv4_1": conv4_1,
             "conv4_2": conv4_2,
             "conv4_3": conv4_3,
             "conv4_4": conv4_4,
             "conv5_1": conv5_1,
             "conv5_2": conv5_2,
             "conv5_3": conv5_3,
             "conv5_4": conv5_4,
             "shallow": shallow
             }

layernames = pm.layernames
if "all" in layernames:
    layers = [layerdict[layer] for layer in layerdict.keys()]
    layernames = list(layerdict.keys())
else:
    layers = [layerdict[layer] for layer in layernames]
# Override for all layer configs
print("Selected layers:\n"+str(layernames))

tf.reset_default_graph()


# Load pretrained model
print("Loading pretrained model...")
model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
print("Finished loading pretrained model")

for i, layer in enumerate(layers):
    STYLE_LAYERS = layer
    # Reset the graph

    # Start interactive session
    print("Starting session...")
    sess = tf.InteractiveSession(config=config)

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

    #     # Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2']
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
    optimizer = tf.train.AdamOptimizer(pm.learning_rate)

    # define train_step
    train_step = optimizer.minimize(J)

    # Run the model
    parameters = [J, J_content, J_style, cImg, sImg, save_image]
    model_nn(sess, generated_image, model, train_step, parameters, num_iterations=pm.n_iter, interval=pm.interval,
             add=layernames[i])

    sess.close()
