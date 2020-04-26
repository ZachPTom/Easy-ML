import sagemaker.amazon.common as smac
import boto3
import io
import os
from numpy import genfromtxt


def compute_ML():
    """ Upload, Train, and Predict """
    upload()


def upload():
    data = 'iris.csv'
    labels = 'labels.csv'

    # S3 path for data, data is uploaded to s3://{bucket}/{prefix}/{key} where key is the file name
    bucket = 'ml-web-app'
    prefix = 'train'
    key = 'train.protobuf'  # or 'test.protobuf'

    # convert data to numpy format for protobuf transformation
    formatted_data = genfromtxt(data, delimiter=',')
    formatted_labels = genfromtxt(labels)

    # transform data to protobuf format
    buf = io.BytesIO()
    smac.write_numpy_to_dense_tensor(buf, formatted_data, formatted_labels)
    buf.seek(0)

    # upload the data to S3
    with open(data, "rb") as f:
        boto3.resource('s3').Bucket(bucket).Object(os.path.join(prefix, key)).upload_fileobj(f)

    path_to_train_data = f's3://{bucket}/{prefix}/{key}'

    print(f'uploaded training data location: s3://{bucket}/{prefix}/{key}')


if __name__ == '__main__':
    compute_ML()
