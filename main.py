import cv2
import dlib
import numpy as np

# Load pre-trained models for age and gender prediction
age_net = cv2.dnn.readNetFromCaffe('models/age_deploy.prototxt', 'models/age_net.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe('models/gender_deploy.prototxt', 'models/gender_net.caffemodel')

# Initialize dlib's face detector
detector = dlib.get_frontal_face_detector()

# Pre-defined labels
age_labels = ['(0-5)', '(6-10)', '(11-15)', '(16-20)', '(21-27)', '(28-35)', '(36-45)', '(46-60)''(60-65)', '(66-90)']
gender_labels = ['Male', 'Female']


def predict_age_gender(face_image):
    blob = cv2.dnn.blobFromImage(face_image, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746),
                                 swapRB=False)

    # Predict gender
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_labels[gender_preds[0].argmax()]

    # Predict age
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = age_labels[age_preds[0].argmax()]

    return age, gender


def detect_and_predict(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_image = image[y:y + h, x:x + w]
        face_image = cv2.resize(face_image, (227, 227))  # Resize to match model input size

        age, gender = predict_age_gender(face_image)

        label = f"{gender}, {age}"

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image


if __name__ == "__main__":
    image_path = r"C:\Users\DISPATCHER\Desktop\BCE 211\images\1.jpg"
    output_image = detect_and_predict(image_path)
    cv2.imshow("Output", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
