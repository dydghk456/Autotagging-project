import os
import gdown
import tensorflow as tf
from deepface.commons import functions

# pylint: disable=unsubscriptable-object

# --------------------------------
# dependency configuration

tf_version = int(tf.__version__.split(".", maxsplit=1)[0])

if tf_version == 1:
    from keras.engine import training
    import keras
    from keras.layers import (
        ZeroPadding2D,
        Input,
        Conv2D,
        BatchNormalization,
        PReLU,
        Add,
        Dropout,
        Flatten,
        Dense,
    )
else:
    from tensorflow.python.keras.engine import training
    from tensorflow import keras
    from tensorflow.keras.layers import (
        ZeroPadding2D,
        Input,
        Conv2D,
        BatchNormalization,
        PReLU,
        Add,
        Dropout,
        Flatten,
        Dense,
    )
# --------------------------------


# url = "https://drive.google.com/uc?id=1LVB3CdVejpmGHM28BpqqkbZP5hDEcdZY"


def loadModel(
    url="https://github.com/serengil/deepface_models/releases/download/v1.0/arcface_weights.h5",
):

    file_name = "/content/drive/MyDrive/chang/weights/arc_res50.h5"
    model = tf.keras.models.load_model(file_name)
    return model


