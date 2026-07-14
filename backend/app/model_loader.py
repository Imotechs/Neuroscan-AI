import os
import h5py
import numpy as np
import keras
from keras import layers

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MY_MODEL_PATH = os.path.join(MODELS_DIR, "my_model.h5")
TRANSFER_MODEL_PATH = os.path.join(MODELS_DIR, "transfer_v2_model.h5")

IMG_SIZE = 224


def load_my_model():
    loaded = keras.models.load_model(MY_MODEL_PATH)
    inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = inp
    for layer in loaded.layers:
        config = layer.get_config()
        new_layer = layer.__class__.from_config(config)
        new_layer.build(x.shape)
        new_layer.set_weights(layer.get_weights())
        x = new_layer(x)
    model = keras.Model(inputs=inp, outputs=x)
    return model


def _build_transfer_model():
    base = keras.applications.MobileNetV2(
        weights=None,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
    )
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(128, activation="relu", name="dense_2")(x)
    x = layers.Dropout(0.5, name="dropout_2")(x)
    out = layers.Dense(4, activation="softmax", name="dense_3")(x)
    model = keras.Model(inputs=base.input, outputs=out)
    return model, base


def load_transfer_model():
    model, base = _build_transfer_model()
    f = h5py.File(TRANSFER_MODEL_PATH, "r")
    mw = f["model_weights"]

    for layer in base.layers:
        weight_path = f"mobilenetv2_1.00_224/{layer.name}"
        if weight_path not in mw:
            continue
        grp = mw[weight_path]
        layer_weight_names = [w.name.split("/")[-1].split(":")[0] for w in layer.weights]
        weights = [np.array(grp[name]) for name in layer_weight_names]
        layer.set_weights(weights)

    for layer_name in ["dense_2", "dense_3"]:
        grp = mw[f"{layer_name}/sequential_1/{layer_name}"]
        kernel = np.array(grp["kernel"])
        bias = np.array(grp["bias"])
        model.get_layer(layer_name).set_weights([kernel, bias])

    f.close()
    return model
