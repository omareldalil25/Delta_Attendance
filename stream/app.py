import cv2
import pandas as pd
import face_recognition
import numpy as np
import base64
from datetime import datetime
import os
import streamlit as st

# Load the Viola-Jones face detection classifier
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# Load images and student names
images = [
    {"name": "Diab", "image_path": "images/Diab.jfif"},
    {"name": "Mostafa Hagag", "image_path": "images/Mostafa Hagag.jfif"},
    {"name": "Ahmed Sheaba", "image_path": "images/Ahmed Sheaba.jfif"},
    {"name": "Dr.Amany Sarhan", "image_path": "images/Dr.Amany.jpg"},
    {"name": "Basel Darwish", "image_path": "images/Basel.jpg"},
    {"name": "Hussein El-Sabagh", "image_path": "images/Hussien.jpg"},
]

attendance_file = "attendance_log.csv"

if not os.path.exists(attendance_file):
    pd.DataFrame(columns=["Name", "Time"]).to_csv(attendance_file, index=False)

known_faces = []
known_names = []

for student in images:
    image = face_recognition.load_image_file(student["image_path"])
    face_encoding = face_recognition.face_encodings(image)[0]
    known_faces.append(face_encoding)
    known_names.append(student["name"])

def recognize_faces_in_image(image_path):
    uploaded_image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(uploaded_image)
    face_encodings = face_recognition.face_encodings(uploaded_image, face_locations)
    
    result_image = cv2.imread(image_path)
    
    total_faces = 0
    correct_recognitions = 0
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"
        
        if True in matches:
            name = known_names[matches.index(True)]
            correct_recognitions += 1
        
        cv2.rectangle(result_image, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(result_image, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
        
        total_faces += 1
    
    accuracy = correct_recognitions / total_faces if total_faces > 0 else 0
    formatted_accuracy = "{:.2%}".format(accuracy)
    print(f"Accuracy: {formatted_accuracy}")

    return result_image, accuracy

def log_attendance(recognized_names):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        attendance_df = pd.read_csv(attendance_file)
    except pd.errors.EmptyDataError:
        attendance_df = pd.DataFrame(columns=["Name", "Time"])
    
    for name in recognized_names:
        attendance_df = attendance_df.append({"Name": name, "Time": current_time}, ignore_index=True)
    
    attendance_df.to_csv(attendance_file, index=False)

def get_recognized_names(result_image, temp_path):
    uploaded_image = face_recognition.load_image_file(temp_path)
    face_locations = face_recognition.face_locations(uploaded_image)
    face_encodings = face_recognition.face_encodings(uploaded_image, face_locations)
    
    recognized_names = []
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"
        
        if True in matches:
            name = known_names[matches.index(True)]
        
        recognized_names.append(name)
    
    return recognized_names

def generate_excel_file():
    try:
        attendance_df = pd.read_csv(attendance_file)
    except pd.errors.EmptyDataError:
        attendance_df = pd.DataFrame(columns=["Name", "Time"])
    
    attendance_df.to_csv(attendance_file, index=False)

# Streamlit App
def main():
    st.title("Face Recognition Attendance System")

    uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        temp_path = 'temp_image.jpg'
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        # Call your existing face recognition function
        result_image, accuracy = recognize_faces_in_image(temp_path)

        # Get the recognized names
        recognized_names = get_recognized_names(result_image, temp_path)

        # Log the attendance
        log_attendance(recognized_names)

        # Convert result image to base64-encoded string
        result_image_base64 = base64.b64encode(cv2.imencode('.jpg', result_image)[1]).decode()

        # Display the result and accuracy
        st.image(result_image, caption=f"Recognition Accuracy: {accuracy:.2%}", use_column_width=True)
        st.success("Attendance recorded successfully!")

if __name__ == "__main__":
    main()
