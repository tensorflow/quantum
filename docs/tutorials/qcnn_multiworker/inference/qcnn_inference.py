# Copyright 2021 The TensorFlow Quantum Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import, division, print_function, unicode_literals
import cirq
import tensorflow as tf
import tensorflow_quantum as tfq
import numpy as np
import sympy
import argparse, datetime
from google.cloud import storage

import qcnn_common


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )


def main(args):
    qcnn_model, _, _, test_excitations, test_labels = qcnn_common.prepare_model()

    qcnn_weights_path = '/tmp/qcnn_weights.h5'
    download_blob(args.weights_gcs_bucket, args.weights_gcs_path, qcnn_weights_path)
    qcnn_model.load_weights(qcnn_weights_path)

    results = qcnn_model(test_excitations).numpy().flatten()
    loss = tf.keras.losses.mean_squared_error(test_labels, results)
    print("Results")
    print(results)
    print("Test Labels")
    print(test_labels)
    print("Mean squared error: ", loss)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--weights-gcs-bucket', help='Name of the GCS bucket for storing training weights.')
    parser.add_argument(
        '--weights-gcs-path', help='GCS url for the QCNN weights file')
    args, _ = parser.parse_known_args()

    main(args)
