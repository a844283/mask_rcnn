
import os
import sys
import tensorflow as tf

MASK_RCNN_DIR = os.path.abspath(os.environ.get('MASK_RCNN_DIR'))
if MASK_RCNN_DIR:
    sys.path.append(MASK_RCNN_DIR)  # To find local version of the library

from mrcnn import model as modellib
from mrcnn.config import Config


class ModelLoader:
    def __init__(self, labels):
        CB_MODEL_PATH = os.path.join(MASK_RCNN_DIR, "mask_rcnn_cb002.h5")
        if CB_MODEL_PATH is None:
            raise OSError('Model path env not found in the system.')

        class InferenceConfig(Config):
            NAME = "cb"
            NUM_CLASSES = 1 + 20  # Coffee bean has 20 classes
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        # Limit gpu memory to 30% to allow for other nuclio gpu functions. Increase fraction as you like
        import keras.backend.tensorflow_backend as ktf
        def get_session(gpu_fraction=0.333):
            gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_fraction,
            allow_growth=True)
            return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        ktf.set_session(get_session())
        # Print config details
        self.config = InferenceConfig()
        self.config.display()

        self.model = modellib.MaskRCNN(mode="inference",
            config=self.config, model_dir=MASK_RCNN_DIR)
        self.model.load_weights(CB_MODEL_PATH, by_name=True)
        self.labels = labels

    def infer(self, image, threshold):
        output = self.model.detect([image], verbose=1)[0]
        results = []
        for i in range(len(output["rois"])):
            score = output["scores"][i]
            class_id = output["class_ids"][i]
            box = output["rois"][i]
            if score >= threshold:
                y1, x1, y2, x2 = box
                width, height = x2 - x1, y2 - y1
                label = self.labels[class_id]
                results.append({
                    "confidence": str(score),
                    "label": label,
                    "points": [float(x1), float(y1), float(x2), float(y2)],
                    "type": "rectangle",
                })

        return results