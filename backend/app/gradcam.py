import io
import base64
import numpy as np
import cv2

IMG_SIZE = 224


def _build_gradcam_submodel(model, layer_name):
    import tensorflow as tf
    layer = model.get_layer(layer_name)
    submodel = tf.keras.Model(
        inputs=model.inputs,
        outputs=[layer.output, model.outputs[0]],
    )
    return submodel


def _compute_heatmap(model, img_array, class_idx, layer_name):
    import tensorflow as tf

    submodel = _build_gradcam_submodel(model, layer_name)

    with tf.GradientTape() as tape:
        conv_output, predictions = submodel(img_array, training=False)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    conv_out_np = conv_output.numpy()[0]

    for i in range(pooled_grads.shape[-1]):
        conv_out_np[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_out_np, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    hmax = np.max(heatmap)
    if hmax > 0:
        heatmap /= hmax

    return heatmap


def _resize_heatmap(heatmap):
    return cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)


def _apply_colormap(heatmap):
    colored = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def _blend_overlay(original_rgb, heatmap_colored, alpha=0.5):
    if original_rgb.dtype != np.uint8:
        original_rgb = (original_rgb * 255).astype(np.uint8)
    return cv2.addWeighted(original_rgb, 1 - alpha, heatmap_colored, alpha, 0)


def _img_to_base64(img_array):
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    _, buf = cv2.imencode(".png", cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    return "data:image/png;base64," + base64.b64encode(buf).decode("utf-8")


def generate_ensemble_heatmaps(models_config, img_array, class_idx):
    combined_heatmap = None

    for model, layer_name in models_config:
        heatmap = _compute_heatmap(model, img_array, class_idx, layer_name)
        heatmap_resized = _resize_heatmap(heatmap)

        if combined_heatmap is None:
            combined_heatmap = heatmap_resized
        else:
            combined_heatmap += heatmap_resized

    combined_heatmap /= len(models_config)

    original_rgb = (img_array[0] * 255).astype(np.uint8)
    analysis_colored = _apply_colormap(combined_heatmap)
    overlay = _blend_overlay(original_rgb, analysis_colored)

    return _img_to_base64(analysis_colored), _img_to_base64(overlay)
