from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import json
from flask import Flask, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import numpy as np
import tensorflow as tf


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
                                input_mean=0, input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels=3,
                                           name='png_reader')
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                      name='gif_reader'))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
    else:
        image_reader = tf.image.decode_jpeg(file_reader, channels=3,
                                            name='jpeg_reader')
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0);
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


UPLOAD_FOLDER = '/home/maciek/photo/'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)


@cross_origin()
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        # print(request)
        # print(file, request.files, request.args, request.form)
        if file and file.filename:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filePath = UPLOAD_FOLDER + filename

            file_name = filePath
            model_file = "/home/maciek/PycharmProjects/model/tensorflow-for-poets-2/tf_files/retrained_graph.pb"
            label_file = "/home/maciek/PycharmProjects/model/tensorflow-for-poets-2/tf_files/retrained_labels.txt"
            input_height = 224
            input_width = 224
            input_mean = 128
            input_std = 128
            input_layer = "input"
            output_layer = "final_result"

            graph = load_graph(model_file)
            t = read_tensor_from_image_file(file_name,
                                            input_height=input_height,
                                            input_width=input_width,
                                            input_mean=input_mean,
                                            input_std=input_std)

            input_name = "import/" + input_layer
            output_name = "import/" + output_layer
            input_operation = graph.get_operation_by_name(input_name)
            output_operation = graph.get_operation_by_name(output_name)

            with tf.Session(graph=graph) as sess:
                results = sess.run(output_operation.outputs[0],
                                   {input_operation.outputs[0]: t})
            results = np.squeeze(results)

            top_k = results.argsort()[-5:][::-1]
            labels = load_labels(label_file)

            m = max(results)
            import math
            for i in top_k:
                if results[i] == m:
                    # print(labels[i], results[i])
                    print(labels[i], results[i] * 100, "%")
                    return json.dumps({'name': labels[i], 'percent': str(math.floor(results[i] * 100))[:-2]})
                else:
                    return json.dumps({'name': 'Cos nie wyszlo', 'percent': '0%'})
    return json.dumps({'name': 'Cos nie wyszlo', 'percent': '0%'})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
