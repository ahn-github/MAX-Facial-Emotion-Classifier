import pytest
import requests


def test_swagger():

    model_endpoint = 'http://localhost:5000/swagger.json'

    r = requests.get(url=model_endpoint)
    assert r.status_code == 200
    assert r.headers['Content-Type'] == 'application/json'

    json = r.json()
    assert 'swagger' in json
    assert json.get('info') and json.get('info').get('title') == 'MAX Facial Emotion Classifier'


def test_metadata():

    model_endpoint = 'http://localhost:5000/model/metadata'

    r = requests.get(url=model_endpoint)
    assert r.status_code == 200

    metadata = r.json()
    assert metadata['id'] == 'max-facial-emotion-classifier'
    assert metadata['name'] == 'MAX Facial Emotion Classifier'
    assert metadata['description'] == 'Emotion classifier trained on the FER+ face dataset'
    assert metadata['license'] == 'MIT'
    assert metadata['type'] == 'Facial Recognition'
    assert 'max-facial-emotion-classifier' in metadata['source']


def _check_response(r):
    assert r.status_code == 200
    response = r.json()
    assert response['status'] == 'ok'
    assert len(response['predictions']) == 1
    assert response['predictions'][0]['emotion_predictions'][0]['label_id'] == '1'
    assert response['predictions'][0]['emotion_predictions'][0]['probability'] > .87
    probs = [p['probability'] for p in response['predictions'][0]['emotion_predictions']]
    assert pytest.approx(sum(probs)) == 1.0

    # bounding box testing
    assert .16 > response['predictions'][0]['detection_box'][0] > .15
    assert .39 > response['predictions'][0]['detection_box'][1] > .38
    assert .54 > response['predictions'][0]['detection_box'][2] > .52
    assert .59 > response['predictions'][0]['detection_box'][3] > .58


def test_response():
    model_endpoint = 'http://localhost:5000/model/predict'
    file_path = 'samples/happy-baby.jpeg'

    with open(file_path, 'rb') as file:
        file_form = {'image': (file_path, file, 'image/jpeg')}
        r = requests.post(url=model_endpoint, files=file_form)
    _check_response(r)

    # test PNG
    file_path = 'tests/happy-baby.png'
    with open(file_path, 'rb') as file:
        file_form = {'image': (file_path, file, 'image/png')}
        r = requests.post(url=model_endpoint, files=file_form)
    _check_response(r)

    # test non-image input
    file_path = 'samples/README.md'
    with open(file_path, 'rb') as file:
        file_form = {'image': (file_path, file, 'image/jpeg')}
        r = requests.post(url=model_endpoint, files=file_form)
    assert r.status_code == 400

    # test non-face image
    file_path = 'tests/non_face.jpg'
    with open(file_path, 'rb') as file:
        file_form = {'image': (file_path, file, 'image/jpeg')}
        r = requests.post(url=model_endpoint, files=file_form)
    assert r.status_code == 200
    response = r.json()
    assert response['status'] == 'ok'
    assert response['predictions'] == []


def test_multiple_faces():
    model_endpoint = 'http://localhost:5000/model/predict'
    file_path = 'samples/group.jpeg'

    with open(file_path, 'rb') as file:
        file_form = {'image': (file_path, file, 'image/jpeg')}
        r = requests.post(url=model_endpoint, files=file_form)

    assert r.status_code == 200
    response = r.json()
    assert response['status'] == 'ok'
    preds = response['predictions']
    # expect 4 detected faces in the image
    assert len(preds) == 4
    # check bounding box coordinates each in [0, 1]
    for p in preds:
        bbox = p['detection_box']
        for b in bbox:
            assert b >= 0 and b <= 1


if __name__ == '__main__':
    pytest.main([__file__])
