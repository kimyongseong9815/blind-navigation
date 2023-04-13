import cv2
import numpy as np
import tensorflow as tf
from enum import Enum
import threading
import time
from gtts import gTTS
import pygame
import os
import asyncio


class Label(Enum):
    KERV = 0
    GREEN_LIGHT = 1
    RED_LIGHT = 2
    CROSS_WALKS = 3


class Coordinate:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y


def play_sound(previous_filename, filename):
    def inner_playsound():
        sound = pygame.mixer.Sound(filename)
        while pygame.mixer.get_busy():
            pass
        channel.play(sound)
        if previous_filename:
            asyncio.run(delete_file(previous_filename))

    threading.Thread(target=inner_playsound, daemon=True).start()

async def delete_file(filename):
    try:
        os.remove(filename)
    except FileNotFoundError:
        print(f'{filename} not found')
    else:
        print(f'{filename} deleted')


interpreter = tf.lite.Interpreter(model_path='best-fp16.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture(0)

class_ids = set()

cross_walk_coordinates = []

last_detection_time = 0

pygame.mixer.init()
channel = pygame.mixer.Channel(0)

previous_filename = ''
filename = ''

while True:
    script = ''

    ret, frame = cap.read()
    frame_resized = cv2.resize(frame, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    input_data = np.expand_dims(frame_resized, axis=0)
    input_data = (np.float32(input_data) - 127.5) / 127.5

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    for detection in output_data[0]:
        confidence = detection[4]
        if confidence > 0.25:
            x_min = int((detection[0] - detection[2]/2) * frame.shape[1])
            y_min = int((detection[1] - detection[3]/2) * frame.shape[0])

            x_max = int((detection[0] + detection[2]/2) * frame.shape[1])
            y_max = int((detection[1] + detection[3]/2) * frame.shape[0])

            class_scores = detection[5:9]
            class_id = np.argmax(class_scores)
            class_ids.add(class_id)
            label = f'class {Label(class_id).name}'

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if class_id == 3:
                cross_walk_coordinates.append(Coordinate(detection[0] * frame.shape[1], detection[1] * frame.shape[0]))

    x_list = []
    y_list = []
    if 3 in class_ids:
        current_time = time.time()

        for coordinate in cross_walk_coordinates:
            x_list.append(coordinate.x)
            y_list.append(coordinate.y)
        
        x_mean = sum(x_list) / len(x_list)
        y_mean = sum(y_list) / len(y_list)

        if y_mean > frame.shape[0] - (frame.shape[0] * 2/3):
            if current_time - last_detection_time >= 30:
                script += '전방에 횡단보도 있습니다.'

            if 2 in class_ids:
                script += '빨간불 입니다.'

            elif 1 in class_ids:
                script += '파란불 입니다.'
                if x_mean < frame.shape[1] * 0.35:
                    script += '횡단보도가 왼쪽에 있습니다.'
                elif x_mean > frame.shape[1] * 0.65:
                    script += '횡단보도가 오른쪽에 있습니다.'

            last_detection_time = current_time

    if script:
        tts = gTTS(text=script, lang='ko')
        timestamp = int(time.time())
        previous_filename = filename
        filename = f'C:/Users/Kimyongseong/Desktop/blind-navigation/{timestamp}.mp3'
        tts.save(filename)
        play_sound(previous_filename, filename)
        
    cv2.imshow('frame', frame)
    
    class_ids.clear()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.5)

cap.release()
cv2.destroyAllWindows()
