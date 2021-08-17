from __future__ import division, print_function
# coding=utf-8
import os
import numpy as np

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, Blueprint
from werkzeug.utils import secure_filename

import json

# Ignore warnings
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

""" --- IMPORT LIBRARIES --- """

import numpy as np
import base64
import torch
from torch import nn
from torch.autograd.variable import Variable
import cv2
from PIL import Image
import io
import uuid
import pandas as pd
import time
from datetime import datetime
import pytz

from .static.models import models_interface

main = Blueprint('main', __name__)

CAT2CLASS = {0: "Male", 1: "Female"}
USER_DATA_DIR = "app/static/user_data"
os.makedirs(USER_DATA_DIR, exist_ok=True)
DF_PATH = os.path.join(USER_DATA_DIR, "info.csv")

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
    with open(os.path.join(USER_DATA_DIR, "access.txt"), "a") as myfile:
        myfile.write(f"{datetime.now(tz=pytz.timezone('Europe/Madrid')).strftime('%Y-%m-%d %H:%M:%S')}\n")
    return render_template('index.html')


@main.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        start_time = time.time()
        data_url = str(request.data)

        image_encoded = data_url.split(',')[1]
        image_bytes = io.BytesIO(base64.b64decode(image_encoded))
        im = Image.open(image_bytes)
        arr = np.array(im)[:, :, 3]

        arr_small = cv2.resize(arr, (28, 28)).reshape(28, 28, 1)
        arr_small = torch.from_numpy(arr_small.transpose(2, 0, 1)).type('torch.FloatTensor')
        arr_small = arr_small.unsqueeze(0)

        # Make the prediction of the current face
        with torch.no_grad():
            prediction = MODEL(Variable(arr_small).cpu())
        preds_classes = torch.argmax(prediction, dim=1)
        confianza = SOFTMAX(prediction)

        # Saves the prediction in a file to latex inspect
        filename = uuid.uuid4().hex
        cv2.imwrite(os.path.join(USER_DATA_DIR, f'{filename}.png'), arr)
        info = {str(clase):prob for clase, prob in enumerate(confianza.view(-1).cpu().numpy())}
        info["file"] = filename
        info["label"] = -1
        info["execution"] = str(time.time() - start_time)

        if os.path.isfile(DF_PATH):
            df = pd.read_csv(DF_PATH)
            df = df.append(info, ignore_index=True)
        else:
            df = pd.DataFrame(info, index=[0])
        df.to_csv(DF_PATH, index=False)

        # Return to the browser what we find as a json object
        return json.dumps({"number": preds_classes.item()})
    return json.dumps({'status': 'error', 'message': 'Bad request'})

@main.route('/ping', methods=['GET', 'POST'])
def ping():
    return json.dumps({'result': 'mnist_draw_pong'})