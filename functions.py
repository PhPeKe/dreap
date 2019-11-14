import tensorflow as tf
import parameters as pm

# GRADED FUNCTION: compute_content_cost

def compute_content_cost(a_C, a_G):
    """
    Computes the content cost

    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

    Returns:
    J_content -- scalar that you compute using equation 1 above.
    """

    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape a_C and a_G (≈2 lines)
    a_C_unrolled = tf.reshape(tf.transpose(a_C), (n_H * n_W, n_C))#tf.reshape(tf.transpose(a_C), (n_H * n_W, n_C))
    a_G_unrolled = tf.reshape(tf.transpose(a_G), (n_H * n_W, n_C))#tf.reshape(tf.transpose(a_G), (n_H * n_W, n_C))

    # compute the cost with tensorflow (≈1 line)
    J_content = tf.reduce_sum(tf.divide(tf.square((tf.subtract(a_C_unrolled, a_G_unrolled))), (4.* n_H * n_W * n_C)))
    #tf.divide(1, (4 * n_H * n_C * n_W)) * tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))

    return J_content

#########################################################################################################################

def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)

    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """

    # Create Gram matrix by multiplying with transposed
    GA = tf.matmul(A, tf.transpose(A))

    return GA

#########################################################################################################################

# GRADED FUNCTION: compute_content_cost

def compute_content_cost(a_C, a_G):
    """
    Computes the content cost

    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

    Returns:
    J_content -- scalar that you compute using equation 1 above.
    """

    # YOUR CODE HERE
    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape a_C and a_G (≈2 lines)
    a_C_unrolled = tf.reshape(tf.transpose(a_C), (n_H * n_W, n_C))#tf.reshape(tf.transpose(a_C), (n_H * n_W, n_C))
    a_G_unrolled = tf.reshape(tf.transpose(a_G), (n_H * n_W, n_C))#tf.reshape(tf.transpose(a_G), (n_H * n_W, n_C))

    # compute the cost with tensorflow (≈1 line)
    J_content = tf.reduce_sum(tf.divide(tf.square((tf.subtract(a_C_unrolled, a_G_unrolled))), (4.* n_H * n_W * n_C)))
    #tf.divide(1, (4 * n_H * n_C * n_W)) * tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))
    # YOUR CODE ENDS HERE

    return J_content

#########################################################################################################################
# GRADED FUNCTION: compute_layer_style_cost

def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

    Returns:
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    # Retrieve dimensions from a_G
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape the images to have them of shape (n_C, n_H*n_W)
    a_S = tf.reshape(tf.transpose(a_S), (n_C, n_H * n_W))
    a_G = tf.reshape(tf.transpose(a_G), (n_C, n_H * n_W))

    # Computing gram_matrices for both images S and G
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the loss
    J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS, GG)) / (4* (n_H * n_W * n_C)**2))

    return J_style_layer

#########################################################################################################################

def compute_style_cost(model, STYLE_LAYERS, sess):
    """
    Computes the overall style cost from several chosen layers

    Arguments:
    model -- our tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them

    Returns:
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    # initialize the overall style cost
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:

        # Select the output tensor of the currently selected layer
        out = model[layer_name]

        # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        a_S = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name]
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out

        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style

#########################################################################################################################
# GRADED FUNCTION: total_cost

def total_cost(J_content, J_style, alpha = pm.alpha, beta = pm.beta):
    """
    Computes the total cost function

    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost

    Returns:
    J -- total cost as defined by the formula above.
    """

    # YOUR CODE HERE
    J = alpha * J_content + beta * J_style
    # YOUR CODE ENDS HERE

    return J

#########################################################################################################################
def model_nn(sess, input_image, model, train_step, parameters, num_iterations = 200, interval = 20, add=""):

    # Initialize global variables (you need to run the session on the initializer)
    # YOUR CODE HERE
    sess.run(tf.initializers.global_variables())
    # YOUR CODE ENDS HERE

    # Run the noisy input image (initial generated image) through the model. Use assign().
    sess.run(model["input"].assign(input_image))
    J, J_content, J_style, cImg, sImg, save_image = parameters

    for i in range(num_iterations):

        # Run the session on the train_step to minimize the total cost
        sess.run(train_step)

        # Compute the generated image by running the session on the current model['input']
        generated_image = sess.run(model["input"])

        print("Iteration " + str(i) + " :")
        # Print every 20 iteration.
        if i%interval == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))

            # save current generated image in the "/output" directory
            save_image("output/" + str(i) + add + ".png", generated_image)

    # save last generated image
    save_image('output/generated_image' + add + '.jpg', generated_image)

    return generated_image

#########################################################################################################################

