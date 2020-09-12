import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam

class DeepViSe:
    def __init__(self):
        self.classes_size = 300
        self.model = self._create_model()
        adam = Adam(lr=0.001, epsilon=0.01, decay=0.0001)
        self.model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])

    def _create_model(self):
        backbone = ResNet50(weights='imagenet')
        limit = len(backbone.layers) - 3
        for index, layer in enumerate(backbone.layers):
            if index > limit:
                break
            layer.trainable = False
        x = backbone.layers[-3].output
        x = Dropout(rate=0.3)(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)
        y = Dense(self.classes_size, activation='linear')(x)
        model = Model(inputs=backbone.input, outputs=y)
        return model

if __name__ == "__main__":
    deep_vise_model = DeepViSe()
