import tensorflow as tf
from keras.applications.resnet50 import ResNet50

class DeepViSe:
    def __init__(self):
        self.model = self._create_model()

    def _create_model(self):
        backbone = ResNet50(weights='imagenet')


if __name__ == "__main__":
    deep_vise_model = DeepViSe()