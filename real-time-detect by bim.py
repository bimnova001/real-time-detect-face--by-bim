import tkinter as tk
from tkinter import *
from tkinter import ttk, Listbox
from PIL import Image, ImageTk
import cv2
import numpy as np
from keras.models import load_model
import csv


try:
    model = load_model("keras_Model.h5", compile=False)
except Exception as e:
    print("Error loading the model:", e)
    exit()


try:
    with open("labels.txt", "r") as f:
        class_names = f.read().splitlines()
except Exception as e:
    print("Error loading the labels:", e)
    exit()

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("Error loading the face cascade classifier.")
    exit()

# Initialize variables
face_data = []
cap = None  # Initialize cap variable for camera

# Function to preprocess the image for model prediction
def preprocess_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    return normalized_image_array

# Function to save face data to a CSV file
def save_to_csv(face_data, filename='face_data.csv'):
    try:
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['nothing',  'me' ,   "%"])
            for idx, (class_name, confidence_score) in enumerate(face_data):
                writer.writerow([class_name,f'{idx+1}'])
        print(f"Face data saved to {filename}")
        list_log.insert(tk.END, f"Face data saved to {filename}")
    except Exception as e:
        print("Error saving face data to CSV:", e)

# Function to perform real-time face detection and classification
def detect_and_classify_faces(label, list_log):
    global face_data, cap  # Access the face_data and cap variables from the global scope
    cap = cv2.VideoCapture(0)  # Open the default camera (typically webcam)
    if not cap.isOpened():
        print("Error opening camera.")
        return

    def update():
        ret, frame = cap.read()  # Read a frame from the camera
        if not ret:
            print("Error reading frame from camera.")
            return
        
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Clear the list log
        list_log.delete(0, tk.END)
        
        # Clear the face data
      
        
        for (x, y, w, h) in faces:
            # Extract the face region from the frame
            face_region = frame[y:y+h, x:x+w]
            
            # Preprocess the face image for model prediction
            face_image = Image.fromarray(face_region)
            normalized_face_image = preprocess_image(face_image)
            
            # Expand dimensions to match model input shape
            data = np.expand_dims(normalized_face_image, axis=0)
            
            # Predict the class of the face
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]
            list_log.insert(tk.END, f"{class_name}: {confidence_score:.2f}%")
            #list_log.insert(tk.END, f"% {confidence_score}")
            save_to_csv(face_data)
            
            # Display the class name and confidence score on the frame
            cv2.putText(frame, f'{class_name}: {confidence_score:.2f}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Add the face data to the list log
            list_log.insert(tk.END, f'{class_name}: {confidence_score:.2f}')
            
            # Append the face data to the face_data list
            face_data.append((class_name, confidence_score))

        # Convert the frame to RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert the frame to ImageTk format
        img = Image.fromarray(rgb_frame)
        img_tk = ImageTk.PhotoImage(image=img)
        
        # Update the label with the new image
        label.imgtk = img_tk
        label.configure(image=img_tk)
        
        # Schedule the next update
        label.after(10, update)

    # Start the update loop
    update()

# Function to toggle the camera
def toggle_camera(button):
    global cap  # Access the cap variable from the global scope
    if cap is None or not cap.isOpened():
        button.config(text="Turn Off Camera")
        detect_and_classify_faces(label, list_log)
    else:
        button.config(text="Turn On Camera")
        cap.release()

# Create the main window
root = tk.Tk()
root.title("Real-Time Face Detection & Classification by Bim XD555")

save_to_csv(face_data)
# Create a label to display the video feed
label = tk.Label(root)
label.pack()

# Create a listbox to display the log
list_log = Listbox(root, width=85, height=15)
list_log.pack()

# Create a button to toggle the camera
toggle_button = ttk.Button(root, text="Turn On Camera", command=lambda: toggle_camera(toggle_button))
toggle_button.pack()

# Run the Tkinter event loop
root.mainloop()
