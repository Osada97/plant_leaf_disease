from fastapi import File, UploadFile
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import cv2
from pathlib import Path

# loadModels
MODEL = tf.keras.models.load_model("./Model/Potato/2")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
PAPPERMODEL = tf.keras.models.load_model("./Model/Papper/1.h5")
PAPPER_CLASS_NAMES = ["Pepper__bell___Bacterial_spot", "Pepper__bell___healthy"]


# read file as image
def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


# convert list as image and save
def list_as_a_image(data, fileName, model, className):
    Path(f"./assets/usersImages/{model}{className}").mkdir(parents=True, exist_ok=True)
    img = np.array(data).astype(np.uint8)
    data = Image.fromarray(img)
    data.save(f"./assets/usersImages/{model}{className}/{fileName}")
    return


# resize image


def resizeImage(image):
    # CLAHE
    # hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # h,s,v = hsv_img[:,:,0],hsv_img[:,:,1],hsv_img[:,:,2]
    # clahe = cv2.createCLAHE(clipLimit=1.0,tileGridSize=(2,2))
    # v = clahe.apply(v)
    # hsv_img = np.dstack((h,s,v))

    # resize_iamge
    imgResize = cv2.resize(image, (256, 256))

    return imgResize


async def predictImage(file: UploadFile = File(...), model: str = "potato"):
    print(file.file)
    image = read_file_as_image(await file.read())

    # openCV function
    AfterCVImage = resizeImage(image)

    img_batch = np.expand_dims(AfterCVImage, 0)

    # checkWithQueryString
    if model == "potato":
        prediction = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    elif model == "papper":
        prediction = PAPPERMODEL.predict(img_batch)
        predicted_class = PAPPER_CLASS_NAMES[np.argmax(prediction[0])]

    confidence = np.max(prediction[0])
    ids = np.argmax(prediction[0]).tolist()

    # save user upload files
    list_as_a_image(AfterCVImage, file.filename, model, predicted_class)

    return {
        "id": ids,
        "class": predicted_class,
        "confidence": float(confidence),
    }
