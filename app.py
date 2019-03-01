from __future__ import division, print_function
# coding=utf-8
import os
import numpy as np

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import json

# Ignore warnings
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

""" --- IMPORT LIBRARIES --- """

import numpy as np
import pickle
import pathlib
import time
import base64
import torch
from torch import nn, optim
from torch.autograd.variable import Variable
import torch.nn.functional as F
import cv2
from PIL import Image
import io

from models import models_interface

CAT2CLASS = {0:"Male", 1:"Female"}

""" -- MODEL LOAD -- """

# We establish a seed for the replication of the experiments correctly
seed = 0
torch.manual_seed(seed=seed)
torch.cuda.manual_seed(seed=seed)

model_type, optimizador = "MLP", "SGD"

print("{} -  using {} - MNIST!)".format(model_type, optimizador))
states_path = "models/mlp_mnist.pt"
MODEL = models_interface.load_model(model_type, states_path=states_path)
MODEL = MODEL.cpu()
MODEL.eval()

SOFTMAX = nn.Softmax()

# Define a flask app
app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predictModel', methods=['GET', 'POST'])
def predictModel():
    if request.method == 'POST':
        data_url = str(request.data)
        #content = data_url.split(';')[1]
        #image_encoded = content.split(',')[1]
        image_encoded = data_url.split(',')[1]
        image_bytes = io.BytesIO(base64.b64decode(image_encoded))
        im = Image.open(image_bytes)
        arr = np.array(im)[:,:,3]
        
        arr_small = cv2.resize(arr, (28, 28)).reshape(28,28,1)
        arr_small = torch.from_numpy(arr_small.transpose(2,0,1)).type('torch.FloatTensor')
        arr_small = arr_small.unsqueeze(0)

        # We make the prediction of the current face
        with torch.no_grad():
            prediction = MODEL(Variable(arr_small).cpu())
        preds_classes = torch.argmax(prediction, dim=1)
        confianza = SOFTMAX(prediction)
        print(prediction)

        info = {}
        info["number"] = preds_classes.item()

        # We return to the browser what we find as a json object
        return json.dumps(info)
    return None


if __name__ == '__main__':
    # Serve the app with gevent
    app.debug = True
    port = int(os.environ.get('PORT', 5000))
    print("\n########################################")
    print('--- Running on port {} ---'.format(port))
    print("########################################\n")
    http_server = WSGIServer(('', port), app)
    http_server.serve_forever()