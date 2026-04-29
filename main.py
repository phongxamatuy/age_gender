from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image
import io
import os

app = Flask(__name__, static_folder='.')
CORS(app)

MODEL_PATH = 'model_23k.keras'
IMG_SIZE = (64,64)

# --- Định nghĩa lại custom classes ---
class ResidualBlock(tf.keras.Model):
    def __init__(self, filters, stride=1, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(filters, (3,3), strides=stride, padding='same', use_bias=False)
        self.bn1   = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, (3,3), padding='same', use_bias=False)
        self.bn2   = layers.BatchNormalization()
        self.shortcut_conv = layers.Conv2D(filters, (1,1), strides=stride, padding='same', use_bias=False)
        self.shortcut_bn   = layers.BatchNormalization()
        self.stride  = stride
        self.filters = filters

    def call(self, x, training=False):
        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        if self.stride != 1 or shortcut.shape[-1] != self.filters:
            shortcut = self.shortcut_conv(shortcut)
            shortcut = self.shortcut_bn(shortcut, training=training)
        return tf.nn.relu(x + shortcut)

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters, 'stride': self.stride})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MyModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rescale    = layers.Rescaling(1./255)
        self.conv1      = layers.Conv2D(32, (3,3), padding='same', use_bias=False)
        self.bn1        = layers.BatchNormalization()
        self.res1  = ResidualBlock(32)
        self.res2  = ResidualBlock(32)
        self.res3  = ResidualBlock(64,  stride=2)
        self.res4  = ResidualBlock(64)
        self.res5  = ResidualBlock(128, stride=2)
        self.res6  = ResidualBlock(128)
        self.res7  = ResidualBlock(256, stride=2)
        self.res8  = ResidualBlock(256)
        self.res9  = ResidualBlock(512, stride=2)
        self.res10 = ResidualBlock(512)
        self.gap         = layers.GlobalAveragePooling2D()
        self.drop        = layers.Dropout(0.5)
        self.age_head    = layers.Dense(1, activation='linear',  name='age')
        self.gender_head = layers.Dense(1, activation='sigmoid', name='gender')

    def call(self, x, training=False):
        x = self.rescale(x)
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.res1(x,  training=training)
        x = self.res2(x,  training=training)
        x = self.res3(x,  training=training)
        x = self.res4(x,  training=training)
        x = self.res5(x,  training=training)
        x = self.res6(x,  training=training)
        x = self.res7(x,  training=training)
        x = self.res8(x,  training=training)
        x = self.res9(x,  training=training)
        x = self.res10(x, training=training)
        x = self.gap(x)
        x = self.drop(x, training=training)
        return {'age': self.age_head(x), 'gender': self.gender_head(x)}

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# --- Load model với custom_objects ---
print("Đang load model...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        'MyModel': MyModel,
        'ResidualBlock': ResidualBlock
    }
)
print("✅ Load model thành công!")


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    return np.expand_dims(img_array, axis=0)


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# --- THÊM: serve CSS và JS ---
@app.route('/style.css')
def serve_css():
    return send_from_directory('.', 'style.css')

@app.route('/script.js')
def serve_js():
    return send_from_directory('.', 'script.js')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Không tìm thấy file ảnh'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Chưa chọn file'}), 400

    try:
        img_array = preprocess_image(file.read())
        outputs   = model.predict(img_array, verbose=0)

        age         = max(0, round(float(np.squeeze(outputs['age']))))
        gender_prob = float(np.squeeze(outputs['gender']))
        gender_label = 'Nam' if gender_prob >= 0.5 else 'Nữ'
        gender_conf  = gender_prob if gender_prob >= 0.5 else 1 - gender_prob

        return jsonify({
            'success': True,
            'age': age,
            'gender': gender_label,
            'gender_confidence': round(gender_conf * 100, 1),
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("🚀 Server chạy tại http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)