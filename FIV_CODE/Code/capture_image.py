import csv
import cv2
import os

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def take_images():
    os.makedirs("TrainingImage", exist_ok=True)
    os.makedirs("StudentDetails", exist_ok=True)

    student_id = input("Enter Your Id: ")
    name = input("Enter Your Name: ")

    if is_number(student_id) and name.isalpha():
        cam = cv2.VideoCapture(0)
        harcascade_path = "haarcascade_default.xml"
        detector = cv2.CascadeClassifier(harcascade_path)
        sample_num = 0

        while True:
            ret, img = cam.read()
            if not ret:
                print("Failed to capture image. Check the camera connection.")
                break
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (10, 159, 255), 2)
                cv2.imshow('frame', img)

                # Capture image on spacebar press
                if cv2.waitKey(1) & 0xFF == ord(' '):  
                    sample_num += 1
                    image_filename = f"TrainingImage{os.sep}{name}.{student_id}.{sample_num}.jpg"
                    cv2.imwrite(image_filename, gray[y:y + h, x:x + w])
                    print(f"Captured image {sample_num}")

            if sample_num >= 10:  # Stop after 10 captures
                break

        cam.release()
        cv2.destroyAllWindows()

        csv_file_path = "StudentDetails" + os.sep + "StudentDetails.csv"
        header = not os.path.isfile(csv_file_path)
        with open(csv_file_path, 'a+', newline='') as csv_file:
            writer = csv.writer(csv_file)
            if header:
                writer.writerow(['Id', 'Name'])  # Write header only if file is new
            writer.writerow([student_id, name])  # Write student data

        print(f"Images saved for ID: {student_id}, Name: {name}")
    else:
        if is_number(student_id):
            print("Enter Alphabetical Name")
        if name.isalpha():
            print("Enter Numeric ID")
