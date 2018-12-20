from keras import models, layers
from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

def reshape(images, num, h=28, w=28, ch=1):
    print(f'reshape images with shape {images.shape} to {(num, h, w, ch)}')
    return images.reshape((num, h, w, ch))

train_images = reshape(train_images, 60000)
train_images = train_images.astype('float32') / 255
train_labels = to_categorical(train_labels)

test_images = reshape(test_images, 10000)
test_images = test_images.astype('float32') / 255
test_labels = to_categorical(test_labels)


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(train_images, train_labels, epochs=5, batch_size=4)
test_loss, test_acc = model.evaluate(test_images, test_labels)