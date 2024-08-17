import tensorflow 
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers 


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0



model = keras.Sequential(
    [
        layers.Dense(512, activation = "relu"),
        layers.Dense(256, activation = "relu"),
        layers.Dense(10)
    ]
)

model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = keras.optimizers.Adam(learning_rate = 0.001),
    metrics = ["accuracy"]
)
model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=1)
model.evaluate(x_test, y_test, batch_size=32, verbose=1)
model.save("handwriting_recognition_model.h5")
