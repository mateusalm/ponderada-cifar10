from flask import Flask, request, jsonify
from mxnet import gluon, nd, image
import numpy as np

app = Flask(__name__)

# Define your custom model architecture
class CustomNet(gluon.HybridBlock):
    def __init__(self, **kwargs):
        super(CustomNet, self).__init__(**kwargs)
        self.conv1 = gluon.nn.Conv2D(32, kernel_size=3, padding=1)
        self.conv2 = gluon.nn.Conv2D(64, kernel_size=3, padding=1)
        self.pool = gluon.nn.MaxPool2D(pool_size=2, strides=2)
        self.dense = gluon.nn.Dense(10)

    def hybrid_forward(self, F, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.flatten(x)
        x = self.dense(x)
        return x

# Load the custom model
def load_model():
    net = CustomNet()
    net.load_parameters('path_to_save_your_model.params')
    net.hybridize()
    return net

net = load_model()

# Define the list of classes (CIFAR-10)
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def transform_image(image_file):
    # Load image
    img = image.imread(image_file)
    # Resize to 32x32
    img = image.imresize(img, 32, 32)
    # Normalize the image
    img = img.astype('float32') / 255
    # Transpose to (3, 32, 32)
    img = nd.transpose(img, (2, 0, 1))
    # Expand dimensions to (1, 3, 32, 32)
    img = img.expand_dims(axis=0)
    return img

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        img = transform_image(file)
        prediction = net(img).softmax()
        predicted_class = np.argmax(prediction.asnumpy(), axis=1)[0]
        predicted_label = classes[predicted_class]

        return jsonify({"class": predicted_label})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
