import cv2

from PIL import Image, ImageOps

cam = cv2.VideoCapture(0)
_, frame = cam.read()

frame.shape
frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
Image.fromarray(frame).save("test.jpg")

image = ImageOps.fit(
        Image.fromarray(frame),
        (224, 224),
        Image.ANTIALIAS,
    )

image.save("test.jpg")