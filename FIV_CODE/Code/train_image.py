import os
import time
import cv2
import numpy as np
from PIL import Image


def getImagesAndLabels(path):
    # Retrieve all image file paths from the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    faces = []
    Ids = []
    
    for imagePath in imagePaths:
        try:
            pilImage = Image.open(imagePath).convert('L')  # Convert image to grayscale
            imageNp = np.array(pilImage, 'uint8')
            Id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces.append(imageNp)
            Ids.append(Id)
        except Exception as e:
            print(f"Error processing image {imagePath}: {e}")
    
    return faces, Ids


def counter_img(num_images):
    for imgcounter in range(1, num_images + 1):
        print(f"{imgcounter} Images Trained", end="\r")
        time.sleep(0.1)  # Adjust the delay as needed


def TrainImages():
    # Initialize the face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer.create()
    harcascadePath = "haarcascade_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)

    # Retrieve faces and IDs
    faces, Ids = getImagesAndLabels("TrainingImage")
    
    # Check if any faces were found
    if len(faces) == 0:
        print("No training images found. Please add images to the 'TrainingImage' directory.")
        return
    
    # Train the recognizer
    recognizer.train(faces, np.array(Ids))
    
    # Save the trained model
    recognizer.save("TrainingImageLabel" + os.sep + "Trainner.yml")

    # Start the image counter
    num_images = len(faces)  # Count the number of images trained
    counter_img(num_images)
    print("\nTraining complete. Model saved as Trainner.yml")
