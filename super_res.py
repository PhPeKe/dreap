import numpy as np
from PIL import Image
import os
from ISR.models import RRDN, RDN
import keras.backend as K
import keras
if K.backend() == 'tensorflow':
    keras.backend.set_image_data_format("channels_last")
for img_name in os.listdir("low/"):
	img = Image.open("low/"+img_name)
	lr_img = np.array(img)
	# rrdn = RRDN(weights='gans')
	rdn = RRDN(weights='gans')  # , arch_params={'x': 4.5}
	sr_img = rdn.predict(lr_img)
	img_pred = Image.fromarray(sr_img)
	img_pred.save("high/highres_"+img_name)
