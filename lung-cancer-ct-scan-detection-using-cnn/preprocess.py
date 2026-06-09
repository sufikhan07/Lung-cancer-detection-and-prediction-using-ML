from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(rescale=1.0/255.0)
valid_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_data = train_datagen.flow_from_directory(r"./Data/train", target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary")
valid_data = valid_datagen.flow_from_directory(r"./Data/valid", target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary")
test_data = test_datagen.flow_from_directory(r"./Data/test", target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary")

print(train_data.class_indices)
