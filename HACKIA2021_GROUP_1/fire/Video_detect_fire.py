import keras
import cv2
import numpy as np

from mtcnn import MTCNN
from PIL import Image

classes = ['fire', 'no_fire', 'start_fire']
input_shape = (224, 224)

#
# # detect and return face locations
# def detect(rgb_frame, emotion_model):
#     im = Image.fromarray(rgb_frame, 'RGB')
#     im = im.resize(input_shape)
#     img_array = np.array(im)
#     img_array = np.expand_dims(img_array, axis=0)
#     predictions = emotion_model.predict(img_array)
#     return classes[np.argmax(predictions)]


def detect(frame, backSub, model):
    global nb_det
    display_text = ''

    fgmask = backSub.apply(frame)
    (contours, hierarchy) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < 500:
            continue

        # get bounding box from countour
        (x, y, w, h) = cv2.boundingRect(c)

        # convert to RGB
        rgb_frame = frame[:, :, ::-1]

        center = [x + (w / 2), y + (h / 2)]
        max_border = max(w, h)
        top = max(int(center[1] - (max_border / 2)), 0)
        right = max(int(center[0] + (max_border / 2)), 0)
        bottom = max(int(center[1] + (max_border / 2)), 0)
        left = max(int(center[0] - (max_border / 2)), 0)
        img_array = rgb_frame[top:top + max_border, left:left + max_border, :]
        img_array = np.array(Image.fromarray(img_array).resize(input_shape))
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)
        label = classes[np.argmax(predictions)]
        if label == 'fire':
            nb_det += 1
            print("nb_det = {}".format(nb_det))
            # draw bounding box
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if nb_det > 0:            
                display_text = 'fire detected'
                break;
        else:
            nb_det = 0

    # add text
    cv2.putText(frame, display_text, (125, 125), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

    return frame


# ****************************************************************************************************
# ------------------------------------------------------------
# Main line
# ------------------------------------------------------------
if __name__ == "__main__":

    backSub = cv2.createBackgroundSubtractorMOG2()

    fire_model = keras.models.load_model("/home/ilia/HackIA21_Input/fire/Video_detect_fire_model.h5")
    video_capture = cv2.VideoCapture(0)
    
    nb_det = 0
    
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        frame_detect = detect(frame, backSub, fire_model)

        cv2.imshow('Video', frame_detect)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
