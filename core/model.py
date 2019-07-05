#
# Copyright 2018-2019 IBM Corp. All Rights Reserved.
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
#

from config import DEFAULT_MODEL_PATH, MODEL_NAME, MODEL_ID
from core.util import img_resize

import logging
from maxfw.model import MAXModelWrapper
import onnxruntime as rt
import numpy as np
from PIL import Image
import pickle
import io
from mtcnn.mtcnn import MTCNN
import cv2
from flask import abort

logger = logging.getLogger()


def read_image(image_data):
    try:
        image = Image.open(io.BytesIO(image_data))
        if not image.mode == 'RGB':
            image = image.convert('RGB')
        image = np.array(image)
        return image
    except IOError as e:
        logger.error(e)
        abort(400, 'Invalid file type/extension. Please provide a valid image (supported formats: JPEG, PNG, TIFF).')


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def post_process_result(probs, idxs, classes, bbox, topk=5):
    results = []
    for i in range(0, topk):
        label_idx = idxs[i]
        result = (label_idx, classes[label_idx], probs[label_idx])
        results.append(result)
    results.append(bbox)
    return results


class ModelWrapper(MAXModelWrapper):

    MODEL_META_DATA = {
        'id': MODEL_ID,
        'name': MODEL_NAME,
        'description': 'Emotion classifier trained on the FER+ face dataset',
        'type': 'Facial Recognition',
        'license': 'MIT',
        'source': 'https://developer.ibm.com/exchanges/models/all/max-facial-emotion-classifier/'
    }

    DETECTION_THRESHOLD = 0.95

    """Model wrapper for ONNX image classification model"""
    def __init__(self, model_name='emotion_ferplus', path=DEFAULT_MODEL_PATH):
        self.input_shape = (1, 1, 64, 64)
        self.img_size = 64
        self.detector = MTCNN()
        logger.info('Loading model from: {}...'.format(path))

        # Load the graph
        self.sess = rt.InferenceSession('{}/{}.onnx'.format(path, model_name))
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        logger.info('Loaded model')

        with open('{}/idx_to_label.pkl'.format(path), 'rb') as f:
            self.idx_to_label = pickle.load(f)

    def _pre_process(self, input_img):
        const = 0.35
        img_h, img_w, _ = np.shape(input_img)
        # downscale image if its w/h > 1024
        input_img = img_resize(input_img)
        img_h, img_w, _ = np.shape(input_img)

        detected = self.detector.detect_faces(input_img)
        # filter based on detection threshold
        detected = [d for d in detected if d['confidence'] > self.DETECTION_THRESHOLD]
        faces = np.empty((len(detected), self.img_size, self.img_size))
        for i, d in enumerate(detected):
            x1, y1, w, h = d['box']
            x2 = x1 + w
            y2 = y1 + h
            # convert box from [x, y, w, h] in pixels to [y1, x1, y2, x2] in normalized pixel values
            d['box'] = [float(y1) / img_h, float(x1) / img_w, float(y2) / img_h, float(x2) / img_w]
            xw1 = max(int(x1 - const * w), 0)
            yw1 = max(int(y1 - const * h), 0)
            xw2 = min(int(x2 + const * w), img_w - 1)
            yw2 = min(int(y2 + const * h), img_h - 1)
            f = cv2.resize(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], (self.img_size, self.img_size))
            f = cv2.cvtColor(f, code=cv2.COLOR_RGBA2GRAY)
            faces[i, :, :] = f
        return (faces, detected)

    def _post_process(self, post_scores):
        scores = post_scores[0]
        result = []
        for i, d in enumerate(post_scores[1]):
            bbox = d['box']
            probs = softmax(scores[i])
            probs = np.squeeze(probs)
            idxs = np.argsort(probs)[::-1]
            result.append(post_process_result(probs, idxs, self.idx_to_label, bbox, topk=8))
        return result

    def _predict(self, pre_x):
        x = pre_x[0]
        tmp = x.shape
        img_num = tmp[0]
        predict_result = []
        # to Emotion input
        for i in range(img_num):
            img_2d = np.array(x[i, :, :])
            img_2d = np.resize(img_2d, self.input_shape)
            img_2d = img_2d.astype(np.float32)
            predict_result.append(self.sess.run([self.output_name], {self.input_name: img_2d})[0].ravel())
        return (predict_result, pre_x[1])
