import matplotlib
matplotlib.use('tkAgg')
from flask import Flask
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

@app.route('/')
def root():
    return 'Deep Visual Semantic Embedding'

if __name__ == "__main__":
    app.run()