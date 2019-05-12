# Flask settings
DEBUG = False

# Flask-restplus settings
RESTPLUS_MASK_SWAGGER = False
SWAGGER_UI_DOC_EXPANSION = 'none'
API_TITLE = 'MAX Facial Emotion Classifier'
API_DESC = 'Detect faces in an image and predict the emotional state of each person'
API_VERSION = '1.0.0'

# default model
MODEL_NAME = API_TITLE
MODEL_ID = MODEL_NAME.lower().replace(' ', '-')
DEFAULT_MODEL_PATH = 'assets'
