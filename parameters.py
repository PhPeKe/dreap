## NST ##
# Relative style/content params
alpha = 500  # 500      # Importance of content
beta = 1       # Importance of style
noise = 0.4    # Initial "disturbance" of content image
learning_rate = 2.0

# Image params
#height = 960
#width = 720
# 1.3814
height = 1148  # 552  #
width = 831  # 400  #
# cImg = "outputcal/500deep.png"             # content image
#cImg = "images/brazil2.jpg"             # content image
#sImg = "images/stream_style_cropped.jpeg"                    # style image
cImg = "images/mamapedi_edit.jpg"             # content image
sImg = "images/style_paint_rot.jpg"                    # style image

# Control params
n_iter = 300    # number of iterations
interval = 10  # interval at which images are saved

# Notes
layernames = ["deep"]  # names of layers to use for the cost functions
# - First do few iterations with different alpha/beta ratios, after that different layer configurations


## Dream ##
dImg = "images/adam_content.jpg"
dOut = "deepdream/filtering/testingiter2nf.png"
dOut = "output/adamdeepdream6filter.png"
downscale = 2
iterations = 6
filter_dream = True
filter_grads = False
filter_size = (20, 20)
filter_sigma = 5
max_layer = 5

layers = {}

## SRI ##
use_folder = False
