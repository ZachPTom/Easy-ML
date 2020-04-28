import math
import re
import json
import requests

import boto3
import numpy as np
from flask import Flask, render_template, redirect, request, session, url_for
import os

from sagemaker.amazon.amazon_estimator import get_image_uri
from werkzeug.utils import secure_filename
import sagemaker.amazon.common as smac
import sagemaker
import io


app = Flask(__name__)
app.config.from_json('app.cfg', silent=True)


@app.route('/')
def root():
    return render_template("index.html")


@app.route('/knn')
def knn_page():
    return render_template("knn.html")


@app.route('/knn/uploader', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'POST':
        d = request.files['data']
        result = request.form
        # form data for k value and sample size
        session['k'] = result['k']
        session['sample_size'] = result['size']

        data_dir = '.\\tmp\\userData\\raw'
        path = os.path.join(data_dir, secure_filename(d.filename))
        print(path)

        d.save(path)
        endpoint_name = "".join(secure_filename(d.filename).split("."))
        session['end_point'] = endpoint_name

        pre_process(secure_filename(d.filename))

        print("Moving on to training")
        return redirect(url_for('train'))
    else:
        return "Uploading your file"


def pre_process(filename):
    data_dir = ".\\tmp\\userData\\"
    processed_sub_dir = "standardized"
    raw_file = os.path.join(data_dir, "raw", filename)

    print("raw data from {}".format(raw_file))
    raw = np.loadtxt(raw_file, delimiter=',')

    np.random.seed(1)
    np.random.shuffle(raw)
    session['feature_size'] = int(raw.shape[1] - 1)

    train_size = int(0.9 * raw.shape[0])
    train_feat = raw[:train_size, :-1]
    train_label = raw[:train_size, -1]
    test_feat = raw[train_size:, :-1]
    test_label = raw[train_size:, -1]

    # save to an s3 bucket
    buf = io.BytesIO()
    smac.write_numpy_to_dense_tensor(buf, train_feat, train_label)
    buf.seek(0)

    bucket = "cs218project2"
    prefix = "proofOfConcept-2020-04-27"
    key = 'sessionData'

    print("uploading")
    train_path = os.path.join(prefix, 'train', key)
    boto3.resource('s3').Bucket(bucket).Object(train_path).upload_fileobj(buf)
    session['train'] = 's3://{}/{}'.format(bucket, train_path)
    print('uploaded training data location: {}'.format(session['train']))

    buf = io.BytesIO()
    smac.write_numpy_to_dense_tensor(buf, test_feat, test_label)
    buf.seek(0)

    print("uploading")
    test_path = os.path.join(prefix, 'test', key)
    boto3.resource('s3').Bucket(bucket).Object(test_path).upload_fileobj(buf)
    session['test'] = 's3://{}/{}'.format(bucket, test_path)
    print('uploaded test data location: {}'.format(session['test']))


@app.route('/knn/train', methods=["POST", "GET"])
def train():
    if request.method == 'POST':
        t = create_knn()
        t_predictor = t.deploy(initial_instance_count=1,
                               content_type='text/csv',
                               instance_type='ml.t2.medium',
                               endpoint_name=session['end_point'])

        return redirect(url_for('predict'))
    else:
        return render_template("train.html")


def create_knn():
    role = 'CS218WebApp'
    params = {
        'feature_dim': session['feature_size'],
        'predictor_type': 'classifier',
        'k': session['k'],
        'sample_size': session['sample_size']
    }
    estimator = sagemaker.estimator.Estimator(get_image_uri(boto3.Session().region_name, "knn"),
                                              role=role,
                                              train_instance_count=1,
                                              train_instance_type='ml.m5.2xlarge',
                                              sagemaker_session=sagemaker.Session(),
                                              hyperparameters=params
                                              )

    fit_input = {'train': session['train'], 'test': session['test']}
    estimator.fit(fit_input)
    return estimator


@app.route('/knn/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # invoke lambda function that gets prediction from trained model
        result = request.form
        # POST to api
        d = {"data": result["point"], "endpoint": session['end_point']}
        js_string = json.dumps(d)
        print(js_string)
        r = requests.post("https://hynsmnoix7.execute-api.us-east-1.amazonaws.com/test/predict",
                          data=js_string)

        return render_template("knnResult.html", result=r.text)
    if request.method == 'GET':
        return render_template("knnPredict.html")


if __name__ == '__main__':
    app.run(debug=True)
