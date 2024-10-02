import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('sign_language_model.h5')

# Define the labels
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Space', 
          'Delete', 'Nothing']

# Initialize the webcam
cap = cv2.VideoCapture('0')     # You can use a video file path here

# Initialize background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Flip the frame horizontally (mirror effect)
    frame = cv2.flip(frame, 1)

    # Apply the background subtraction to get the foreground mask
    fg_mask = bg_subtractor.apply(frame)

    # Use morphological operations to remove noise from the foreground mask
    kernel = np.ones((5, 5), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours and len(contours) > 0:
        # Find the largest contour (which should be the hand)
        largest_contour = max(contours, key=cv2.contourArea)

        # Create a bounding box around the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Draw the bounding box on the original frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Extract the region of interest (ROI) for gesture recognition
        roi = frame[y:y+h, x:x+w]

        # Preprocess the ROI (convert to grayscale, resize, normalize)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(roi, (64, 64))
        roi = roi / 255.0
        roi = np.reshape(roi, (1, 64, 64, 1))

        # Make a prediction using the model
        prediction = model.predict(roi)
        max_index = np.argmax(prediction[0])
        predicted_label = labels[max_index]

        # Display the prediction label on the frame
        cv2.putText(frame, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Show the original frame with bounding box and prediction
    cv2.imshow('Sign Language Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
