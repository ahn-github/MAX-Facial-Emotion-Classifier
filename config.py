# Flask settings
DEBUG = True

# Flask-restplus settings
RESTPLUS_MASK_SWAGGER = False
SWAGGER_UI_DOC_EXPANSION = 'none'
API_TITLE = 'MAX Facial Emotion Classifier'
API_DESC = 'Predict emotional state from images of faces'
API_VERSION = '1.0.0'

# default model
MODEL_NAME = 'FER+ Emotion Recognition Model'
MODEL_ID = 'emotion_ferplus'
DEFAULT_MODEL_PATH = 'assets'
