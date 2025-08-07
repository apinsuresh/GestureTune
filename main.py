import cv2
import numpy as np
import mediapipe as mp
import pygame
import os
import tkinter as tk
from tkinter import filedialog
import time

# Initialize Pygame Mixer
pygame.mixer.init()

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Audio state
audio_files = []
current_audio_index = 0
is_playing = False

# Swipe detection
prev_hand_center_x = None
swipe_threshold = 0.15
last_swipe_time = 0
swipe_cooldown = 1.0  # seconds
gesture_cooldown = 1.0
last_gesture_time = 0

# Select audio files
def select_audio_files():
    root = tk.Tk()
    root.withdraw()
    files = filedialog.askopenfilenames(filetypes=[("Audio Files", "*.mp3 *.wav *.ogg")])
    return list(files)

# Load audio files
def load_audio_files():
    global audio_files
    audio_files = select_audio_files()
    if audio_files:
        print(f"Loaded {len(audio_files)} files")

# Play audio
def play_current_audio():
    global is_playing
    if audio_files and 0 <= current_audio_index < len(audio_files):
        pygame.mixer.music.load(audio_files[current_audio_index])
        pygame.mixer.music.play()
        is_playing = True
        print(f"Playing: {os.path.basename(audio_files[current_audio_index])}")

# Stop audio
def stop_audio():
    global is_playing
    pygame.mixer.music.stop()
    is_playing = False
    print("Stopped playback")

# Detect thumbs up gesture (Right hand)
def is_thumb_up(landmarks):
    tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    ip = landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    mcp = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    return tip.y < ip.y and tip.y < mcp.y

# Detect open palm (Left hand)
def is_palm_open(landmarks):
    fingers = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP,
    ]
    wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]
    return all(landmarks.landmark[f].y < wrist.y for f in fingers)

# Load music
load_audio_files()

# Main loop
while cap.isOpened() and audio_files:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    current_time = time.time()

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label  # 'Left' or 'Right'
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            hand_center_x = wrist.x

            # RIGHT HAND: Play music with thumbs up
            if label == "Right":
                if is_thumb_up(hand_landmarks) and not is_playing and current_time - last_gesture_time > gesture_cooldown:
                    play_current_audio()
                    last_gesture_time = current_time

                # Swipe detection (change tracks)
                if prev_hand_center_x is not None:
                    dx = hand_center_x - prev_hand_center_x
                    if abs(dx) > swipe_threshold and current_time - last_swipe_time > swipe_cooldown:
                        if dx > 0:
                            current_audio_index = (current_audio_index + 1) % len(audio_files)
                            print(f"Next track: {os.path.basename(audio_files[current_audio_index])}")
                        else:
                            current_audio_index = (current_audio_index - 1) % len(audio_files)
                            print(f"Previous track: {os.path.basename(audio_files[current_audio_index])}")
                        last_swipe_time = current_time
                        if is_playing:
                            play_current_audio()
                prev_hand_center_x = hand_center_x

            # LEFT HAND: Stop music with open palm
            elif label == "Left":
                if is_palm_open(hand_landmarks) and is_playing and current_time - last_gesture_time > gesture_cooldown:
                    stop_audio()
                    last_gesture_time = current_time

    # UI Display
    current_file = os.path.basename(audio_files[current_audio_index])
    status = "Playing" if is_playing else "Stopped"

    cv2.putText(frame, "Right Thumb Up -> Play", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, "Left Palm Open -> Stop", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, "Swipe Right Hand -> Next/Prev", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"Track: {current_file}", (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    cv2.putText(frame, f"Status: {status}", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    cv2.imshow("Gesture Music Player", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
pygame.mixer.quit()
cap.release()
cv2.destroyAllWindows()
