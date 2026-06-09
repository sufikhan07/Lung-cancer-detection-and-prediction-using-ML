import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from cnn_model import build_cnn_model  


IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode="nearest"
)

valid_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)


train_data = train_datagen.flow_from_directory(
    r"./Data/train",
    target_size=IMG_SIZE, 
    batch_size=BATCH_SIZE, 
    class_mode="binary"
)

valid_data = valid_datagen.flow_from_directory(
    r"./Data/valid",
    target_size=IMG_SIZE, 
    batch_size=BATCH_SIZE, 
    class_mode="binary"
)

test_data = test_datagen.flow_from_directory(
    r"./Data/test",
    target_size=IMG_SIZE, 
    batch_size=BATCH_SIZE, 
    class_mode="binary",
    shuffle=False
)


cnn_model = build_cnn_model()


early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)


cnn_model.fit(
    train_data,
    validation_data=valid_data,
    epochs=25,
    callbacks=[early_stopping, reduce_lr]
)


cnn_model.save("lung_cancer_cnn_model.keras")



test_loss, test_acc = cnn_model.evaluate(test_data)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
