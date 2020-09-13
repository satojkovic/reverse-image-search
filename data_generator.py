from keras.utils import Sequence

class ImageGenerator(Sequence):
    def __init__(self, path, fnames, labels, classes_size, batch_size, image_size=(224, 224), shuffle=True):
        self.path = path
        self.fnames = fnames
        self.labels = labels
        self.classes_size = classes_size
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle