import tkinter as tk
from tkinter import filedialog, messagebox, Label, Button
from PIL import Image, ImageTk
import numpy as np
import cv2
from tensorflow.keras.models import model_from_json

class EmotionDetectorApp:
    def __init__(self, master):
        self.master = master
        self.master.geometry('800x600')
        self.master.title('Emotion Detector')
        self.master.configure(background='#F0F0F0')

        self.model = self.load_model("model_a1.json", "model_weights1.h5")
        self.EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

        self.label1 = Label(master, text='Upload an image to detect emotion', font=('Arial', 14), bg='#F0F0F0', fg='#333333')
        self.label1.pack(pady=20)

        self.upload_button = Button(master, text="Upload Image", command=self.upload_image, bg='#4CAF50', fg='white', font=('Arial', 12), relief=tk.FLAT)
        self.upload_button.pack(pady=10)

        self.sign_image = Label(master, bg='#F0F0F0')
        self.sign_image.pack()

        self.detect_button = Button(master, text="Detect Emotion", command=self.detect_emotion, bg='#2196F3', fg='white', font=('Arial', 12), relief=tk.FLAT)
        self.detect_button.pack(pady=10)

    def load_model(self, json_file, weights_file):
        with open(json_file, "r") as file:
            loaded_model_json = file.read()
            model = model_from_json(loaded_model_json)
        model.load_weights(weights_file)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def detect_emotion(self):
        try:
            image_path = self.image_path
            image = cv2.imread(image_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)

            for (x, y, w, h) in faces:
                roi_gray = gray_image[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                pred = self.EMOTIONS_LIST[np.argmax(self.model.predict(roi_gray[np.newaxis, :, :, np.newaxis]))]
                self.label1.config(text=f"Predicted Emotion: {pred}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def upload_image(self):
        try:
            file_path = filedialog.askopenfilename()
            self.image_path = file_path
            uploaded = Image.open(file_path)
            uploaded.thumbnail((400, 400))
            im = ImageTk.PhotoImage(uploaded)

            self.sign_image.configure(image=im)
            self.sign_image.image = im
            self.label1.config(text='Click "Detect Emotion" to analyze')
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

def main():
    root = tk.Tk()
    app = EmotionDetectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
