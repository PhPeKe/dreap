# Relative style/content params
alpha = 1      # Importance of content
beta = 1000       # Importance of style
noise = 0.2     # Initial "disturbance" of content image

# Image params
#height = 960
#width = 720
height = 789
width = 1052
cImg = "images/valley_content.jpg"             # content image
sImg = "images/style_paint.jpg"                    # style image

# Control params
n_iter = 1000    # number of iterations for the nn
interval = 100  # interval at which images are saved

layernames = ["deep"]  # names of layers to use for the cost functions
