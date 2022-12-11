import tensorflow as tf
import pathlib
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


data_dir = pathlib.Path(r"C:\Users\Admin\Downloads\alphabet1")
# Parameters
batch_size = 10
img_height = 28
img_width = 28
# amount of images in dataset
image_count = len(list(data_dir.glob('*/*.png')))
print("image_count", image_count)
# look at one image
"""a = list(data_dir.glob('а/*'))
im = PIL.Image.open(str(a[0]))
im.show()"""
# train set
train_ds = tf.keras.utils.image_dataset_from_directory (
    data_dir,
    labels='inferred',
    label_mode='categorical',
    # class_names=['а', 'к', 'х'],
    validation_split=0.2,
    color_mode="grayscale",
    subset="training",
    seed=123,
    shuffle=True,
    image_size=(img_height, img_width),
    batch_size=batch_size)
# val set
val_ds = tf.keras.utils.image_dataset_from_directory (
    data_dir,
    labels='inferred',
    label_mode='categorical',
    validation_split=0.2,
    color_mode="grayscale",
    subset="validation",
    seed=123,
    shuffle=True,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
# print (class_names)

val_batches = tf.data.experimental.cardinality(val_ds)
print("val_batches", val_batches)
test_ds = val_ds.take((4*val_batches) // 5)  #4/5 val_batches( 80%) - test, 20% -  val_ds
val_ds = val_ds.skip((4*val_batches) // 5)

print("len(train_ds)", len(train_ds))
print("len(val_ds)", len(val_ds))
print("len(test_ds)", len(test_ds))


for image, label in train_ds.take(5):
    imageShape = image.numpy().shape
    label = label.numpy()
    labelName = class_names[np.argmax(label)]
    print('Image Shape: {}, LabelName: {}'.format(imageShape, labelName))
    # print('Label: {}'.format (label))

# amount of classes
num_classes = len(class_names)
print("num_classes", num_classes)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(height_factor=(-0.05, -0.15),
                      width_factor=(-0.05, -0.15)),
    # layers.RandomTranslation(height_factor=0.2,
    #                          width_factor=0.2),
    # layers.RandomBrightness(factor=[-0.05, 0.05]),
    # layers.RandomContrast(factor=0.2),
])

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    # data_augmentation,
    keras.layers.Flatten(input_shape=(28, 28)),  # 2d-массив (28x28 пикселей) -> 1d-массив из 28 * 28 = 784 пикселей
    # tf.keras.layers.Dense(784, input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(num_classes, activation='softmax')
    # возвращает массив из num_classes вероятностных оценок, сумма которых =1.
    # Каждый узел содержит оценку, к-рая указывает вероятность того, что текущее
    # изображение принадлежит 1 из num_classes классов.
])
lr = 9E-6
opt = keras.optimizers.Adam(learning_rate=lr)
model.compile(
    optimizer=opt,
    loss=tf.losses.categorical_crossentropy,
    # (from_logits=False),
    metrics=['accuracy']
)

checkpoint = ModelCheckpoint('best_model.h5', monitor=['loss'], verbose=0, mode='min')
earlystop = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True,)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=4, verbose=1, factor=0.7, min_lr=1E-9)
list_of_callbacks = [earlystop, checkpoint,
                     learning_rate_reduction
                     ]

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=5000,
    # callbacks=list_of_callbacks
)

# pd.DataFrame(history.history).plot(figsize=(8,5))
# plt.show()

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show(block=True)

test_loss, test_acc = model.evaluate(val_ds, verbose=2)
print("test_acc", test_acc)
# model.summary()
# plot_metric(history, 'loss')
# model.evaluate(val_ds)
# weights = model.get_weights()
# print(weights)
model.save("my_h5_model.h5")
