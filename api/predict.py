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

from core.model import ModelWrapper, read_image
from maxfw.core import MAX_API, PredictAPI, MetadataAPI
from flask_restplus import fields
from werkzeug.datastructures import FileStorage

model_wrapper = ModelWrapper()

# === Labels API

model_label = MAX_API.model('ModelLabel', {
    'id': fields.String(required=True, description='Label identifier'),
    'name': fields.String(required=True, description='Label')
})

labels_response = MAX_API.model('LabelsResponse', {
    'count': fields.Integer(required=True, description='Number of labels returned'),
    'labels': fields.List(fields.Nested(model_label), description='List of labels that can be predicted by the model')
})

class ModelLabelsAPI(MetadataAPI):
    '''API for getting information about available class labels'''
    @MAX_API.doc('get_labels')
    @MAX_API.marshal_with(labels_response)
    def get(self):
        '''Return the list of labels that can be predicted by the model'''
        result = {}
        result['labels'] = [{'id': l[0], 'name': l[1]} for l in model_wrapper.idx_to_label.items()]
        result['count'] = len(model_wrapper.idx_to_label)
        return result

# === Predict API

# Set up parser for input data (http://flask-restplus.readthedocs.io/en/stable/parsing.html)
input_parser = MAX_API.parser()
# Example parser for file input
input_parser.add_argument('image', type=FileStorage, location='files', required=True,
                          help='An image file (encoded as JPEG, PNG or TIFF)')

emotion_label = MAX_API.model('EmotionLabel', {
        'label_id': fields.String(required=False, description='Label identifier'),
        'label': fields.String(required=True, description='Class label'),
        'probability': fields.Float(required=True, description='Predicted probability for the class label')
    })

emotion_example = [
    {
        "label_id": "1",
        "label": "happiness",
        "probability": 0.9860254526138306
    },
    {
        "label_id": "0",
        "label": "neutral",
        "probability": 0.011981048621237278
    }
]

# Creating a JSON response model: https://flask-restplus.readthedocs.io/en/stable/marshalling.html#the-api-model-factory
label_prediction = MAX_API.model('LabelPrediction', {
    'detection_box': fields.List(fields.Float(required=True), description='Bounding box coordinates for the face, ' + \
        'in the form of an array of normalized coordinates [ymin, xmin, ymax, xmax]. Each coordinate is in the range [0, 1]',
        example=[0.15, 0.38, 0.53, 0.58]),
    'emotion_predictions': fields.List(fields.Nested(emotion_label),
        description='Predicted emotion labels and probabilities for the face',
        example=emotion_example)
})


predict_response = MAX_API.model('ModelPredictResponse', {
    'status': fields.String(required=True, description='Response status message', example='ok'),
    'predictions': fields.List(fields.Nested(label_prediction),
        description='Predicted bounding boxes, emotion labels and probabilities for each detected face in input image')
})

class ModelPredictAPI(PredictAPI):

    @MAX_API.doc('predict')
    @MAX_API.expect(input_parser)
    @MAX_API.marshal_with(predict_response)
    def post(self):
        """Make a prediction given input data"""
        result = {'status': 'error'}
        args = input_parser.parse_args()
        input_data = args['image'].read()
        img = read_image(input_data)
        preds = model_wrapper.predict(img)

        label_preds = []
        for res in preds:
            face_emo = {}
            emotion_predictions = []
            # Modify this code if the schema is changed
            for p in res:
                if type(p) is tuple:
                    emo={'label_id': p[0], 'label': p[1], 'probability': p[2]}
                    emotion_predictions.append(emo)
                if type(p) is list:
                    face_emo['detection_box'] = p
            face_emo['emotion_predictions'] = emotion_predictions
            label_preds.append(face_emo)
        result['predictions'] = label_preds
        result['status'] = 'ok'
        return result