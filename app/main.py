from __future__ import division, print_function
# coding=utf-8
import os
import numpy as np

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, Blueprint
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

from .static.models import models_interface

main = Blueprint('main', __name__)

CAT2CLASS = {0: "Male", 1: "Female"}

""" -- MODEL LOAD -- """

# We establish a seed for the replication of the experiments correctly
seed = 0
torch.manual_seed(seed=seed)
torch.cuda.manual_seed(seed=seed)

model_type, optimizador = "MLP", "SGD"

print("{} -  using {} - MNIST!)".format(model_type, optimizador))
states_path = "app/static/models/mlp_mnist.pt"
MODEL = models_interface.load_model(model_type, states_path=states_path)
MODEL = MODEL.cpu()
MODEL.eval()

SOFTMAX = nn.Softmax()


@main.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@main.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data_url = str(request.data)

        image_encoded = data_url.split(',')[1]
        image_bytes = io.BytesIO(base64.b64decode(image_encoded))
        im = Image.open(image_bytes)
        arr = np.array(im)[:, :, 3]

        arr_small = cv2.resize(arr, (28, 28)).reshape(28, 28, 1)
        arr_small = torch.from_numpy(arr_small.transpose(2, 0, 1)).type('torch.FloatTensor')
        arr_small = arr_small.unsqueeze(0)

        # We make the prediction of the current face
        with torch.no_grad():
            prediction = MODEL(Variable(arr_small).cpu())
        preds_classes = torch.argmax(prediction, dim=1)
        confianza = SOFTMAX(prediction)
        print(prediction)

        # We return to the browser what we find as a json object
        return json.dumps({"number": preds_classes.item()})
    return None
