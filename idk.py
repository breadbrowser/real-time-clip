import cv2
import torch
import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import time

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN50", device=device)

# Set up webcam
cap = cv2.VideoCapture(0)

# Define the list of classes to recognize
classes = ["human","not human"]

# Preprocess the image from the webcam
def preprocess_webcam_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = Image.fromarray(image)
    image_input = preprocess(image).unsqueeze(0).to(device)
    return image_input

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Preprocess the image and get the features
    image_input = preprocess_webcam_image(frame)
    with torch.no_grad():
        image_features = model.encode_image(image_input)


    # Get the probabilities for each class
    text_inputs = clip.tokenize(classes).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    logits_per_image, logits_per_text = model(image_input, text_inputs)
    probs = logits_per_image.softmax(dim=-1).cpu().detach().numpy()


    # Display the resulting frame with the predicted class
    predicted_class = classes[probs.argmax()]
    cv2.putText(frame, predicted_class, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('frame', frame)
    time.sleep(2)
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
