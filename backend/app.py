from instance_segmentation import InstanceSegmentation
from flask import Flask, render_template, send_file, jsonify, request
import os
import sys
import json
import numpy as np
import cv2
from tensorflow import keras
from PIL import Image
import io

from PIL import Image
import numpy as np
from skimage import transform






app = Flask(__name__)
instance_segmentation = None
flags={'segment':False}


@app.route('/instance_segmentation', methods=['POST'])
def home():
  if flags['segment']:
    instance_segmentation.predict()
    print(f"Segmented image saved", file=sys.stdout)
    return ''
  else:
    return 'Sorry, we have not deployed the model yet'

@app.route('/house_drawing_classifier/', methods=['POST'])
def receive(): 
  
    #img = cv2.imdecode(np.fromstring(request.files['image'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
  
  image = request.files["image"]
  np_image = Image.open(io.BytesIO(image.read()))
  np_image = np.array(np_image).astype('float32')/255
  np_image = transform.resize(np_image, (520, 292, 3))
  np_image = np.expand_dims(np_image, axis=0)
  
  output = house_classifier.predict(np_image)
  print(output[0][0])
  if output[0][0] > 0.5:
    return "It is a house drawing"
  else:
    return "It is NOT a house drawing"


@app.after_request
def set_response_headers(response):
  response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
  response.headers['Pragma'] = 'no-cache'
  response.headers['Expires'] = '0'
  return response

if __name__ == '__main__':
  house_classifier = keras.models.load_model('models/correctnes_classifier_house.h5') 
  if flags['segment']: 
    instance_segmentation = InstanceSegmentation(weight_path='predictive_models/mask_rcnn_coco.h5')
  print(f"RCNN model loaded", file=sys.stdout)
  app.run(debug=True, host='0.0.0.0')
