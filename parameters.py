# Relative style/content params
alpha = 40      # Importance of content
beta = 30       # Importance of style
noise = 0.3     # Initial "disturbance" of content image

# Image params
height = 1026
width = 770
cImg = "img/railway_scaled.jpg"             # content image
sImg = "img/realist_sunset_style_full.jpg"  # style image

# Control params
n_iter = 200    # number of iterations for the nn
interval = 200  # interval at which images are saved
