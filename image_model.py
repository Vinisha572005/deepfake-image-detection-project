import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

model = load_model('models/deepfake_cnn.h5')

# ---------------------------------------------------
# PREDICTION FUNCTION
# ---------------------------------------------------

def predict_image(img_path):

    img = image.load_img(
        img_path,
        target_size=(128,128)
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = img_array / 255.0

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        return "REAL IMAGE"
    else:
        return "DEEPFAKE IMAGE"