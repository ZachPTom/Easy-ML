import sagemaker
import boto3
import io
import os
import csv
from numpy import genfromtxt
from sagemaker.amazon.amazon_estimator import get_image_uri


def compute_ML():
    """ Upload, Train, and Predict """
    train_data_path, bucket, formatted_data = upload()
    submit_training_job(train_data_path, bucket, formatted_data)


def upload():
    data = 'iris_train.csv'
    labels = 'labels_train.csv'

    # S3 path for data, data is uploaded to s3://{bucket}/{prefix}/{key} where key is the file name
    bucket = 'ml-web-app-sagemaker'
    prefix = 'train'
    key = 'train.protobuf'  # or 'test.protobuf'

    # convert data to numpy format for protobuf transformation
    formatted_data = genfromtxt(data, dtype='f', delimiter=',')
    formatted_labels = genfromtxt(labels, dtype='f')
    print(formatted_data)
    print(formatted_labels)

    # transform data to protobuf format
    buf = io.BytesIO()
    sagemaker.amazon.common.write_numpy_to_dense_tensor(buf, formatted_data, formatted_labels)
    buf.seek(0)

    # upload the data to S3
    boto3.resource('s3').Bucket(bucket).Object(os.path.join(prefix, key)).upload_fileobj(buf)

    train_data_path = f's3://{bucket}/{prefix}\{key}'

    print(f'training data located at: s3://{bucket}/{prefix}/{key}')

    return train_data_path, bucket, formatted_data


def submit_training_job(path_to_train_data, bucket, formatted_data):
    output_prefix = 'train_output'
    role = 'arn:aws:iam::450246219423:role/service-role/AmazonSageMaker-ExecutionRole-20200426T181822'

    train_data_path = path_to_train_data

    # path_to_test_data = f's3://ml-web-app/test/test.protobuf'
    # job_name = 'iris-train'

    output_path = 's3://{}/{}/factorization_machine_output'.format(bucket, output_prefix)

    container = get_image_uri(boto3.Session(region_name='us-west-1').region_name, 'factorization-machines')

    estimator = sagemaker.estimator.Estimator(container, role, train_instance_count=1,
                                              train_instance_type='ml.c4.xlarge', output_path=output_path,
                                              sagemaker_session=sagemaker.Session())

    estimator.set_hyperparameters(feature_dim=formatted_data.shape[1], predictor_type='regressor', num_factors=64)

    # run training job

    estimator.fit({'train': train_data_path})

    # training_job_name = estimator.latest_training_job.job_name


if __name__ == '__main__':
    compute_ML()
