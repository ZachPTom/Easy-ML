import json
import requests

import boto3
import numpy as np
from flask import Flask, render_template, redirect, request, session, url_for
import os

from numpy import genfromtxt
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
def upload_knn():
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
        session['end_point'] = ''.join(e for e in endpoint_name if e.isalnum())

        pre_process(secure_filename(d.filename))

        print("Moving on to training")
        return redirect(url_for('train_knn'))
    else:
        return "Uploading your file"


def pre_process(filename):
    data_dir = ".\\tmp\\userData\\"
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
def train_knn():
    if request.method == 'POST':
        t = create_knn()
        t.deploy(initial_instance_count=1,
                 content_type='text/csv',
                 instance_type='ml.t2.medium',
                 endpoint_name=session['end_point'])

        return redirect(url_for('predict'))
    else:
        return render_template("knnTrain.html")


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
        r = requests.post("https://hynsmnoix7.execute-api.us-east-1.amazonaws.com/Prod/knn/predict",
                          data=js_string)

        return render_template("knnResult.html", result=r.text)
    if request.method == 'GET':
        return render_template("knnPredict.html")


@app.route('/factorization')
def factorization():
    return render_template("factorization.html")


@app.route('/factorization/uploader', methods=["POST", "GET"])
def upload_factor():
    if request.method == 'POST':
        d = request.files['data']
        labels = request.files['labels']

        result = request.form
        # form data for k value and sample size
        session['factors'] = result['factors']

        data_dir = '.\\tmp\\userData\\raw'
        data_path = os.path.join(data_dir, secure_filename(d.filename))
        label_path = os.path.join(data_dir, secure_filename(labels.filename))

        d.save(data_path)
        labels.save(label_path)
        endpoint_name = "".join(secure_filename(d.filename).split("."))
        session['end_point'] = ''.join(e for e in endpoint_name if e.isalnum())

        upload_factor_files(data_path,
                            label_path)

        print("Moving on to training")
        return redirect(url_for('train_factor'))


@app.route('/factorization/train', methods=["POST", "GET"])
def train_factor():
    if request.method == 'POST':
        t = submit_training_job()
        t.deploy(initial_instance_count=1,
                 content_type='text/csv',
                 instance_type='ml.t2.medium',
                 endpoint_name=session['end_point'])

        return redirect(url_for('predict_factor'))
    else:
        return render_template("factorizationTrain.html")


@app.route('/factorization/predict', methods=['POST', 'GET'])
def predict_factor():
    if request.method == 'POST':
        result = request.form

        d = {"data": result["point"], "endpoint": session['end_point']}
        js_string = json.dumps(d)

        r = requests.post("https://hynsmnoix7.execute-api.us-east-1.amazonaws.com/Prod/factorization/predict",
                          data=js_string)

        return render_template('factorizationResult.html', result=r.text)
    else:
        return render_template('factorizationPredict.html')


def upload_factor_files(data, labels):
    # S3 path for data, data is uploaded to s3://{bucket}/{prefix}/{key} where key is the file name
    bucket = 'cs218project2'
    prefix = 'train'
    key = 'train.protobuf'  # or 'test.protobuf'

    # convert data to numpy format for protobuf transformation
    formatted_data = genfromtxt(data, dtype='f', delimiter=',')
    session['feature_size'] = formatted_data.shape[1]
    formatted_labels = genfromtxt(labels, dtype='f')
    print(formatted_data)
    print(formatted_labels)

    # transform data to protobuf format
    buf = io.BytesIO()
    sagemaker.amazon.common.write_numpy_to_dense_tensor(buf, formatted_data, formatted_labels)
    buf.seek(0)

    # upload the data to S3
    boto3.resource('s3').Bucket(bucket).Object(os.path.join(prefix, key)).upload_fileobj(buf)

    train_data_path = f's3://{bucket}/{prefix}\\{key}'
    session['train_data_path'] = train_data_path
    print(f'training data located at: s3://{bucket}/{prefix}/{key}')


def submit_training_job():
    role = 'CS218WebApp'

    # path_to_test_data = f's3://ml-web-app/test/test.protobuf'
    # job_name = 'iris-train'

    container = get_image_uri(boto3.Session().region_name, 'factorization-machines')

    estimator = sagemaker.estimator.Estimator(container,
                                              role,
                                              train_instance_count=1,
                                              train_instance_type='ml.c4.xlarge',
                                              sagemaker_session=sagemaker.Session())

    estimator.set_hyperparameters(feature_dim=session['feature_size'],
                                  predictor_type='regressor',
                                  num_factors=session['factors'])

    # run training job

    estimator.fit({'train': session['train_data_path']})
    return estimator
    # training_job_name = estimator.latest_training_job.job_name


if __name__ == '__main__':
    app.run(debug=True)
