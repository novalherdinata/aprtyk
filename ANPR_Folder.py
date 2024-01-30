import streamlit as st
import cv2
import torch
import easyocr
import re
from io import BytesIO

@st.cache_resource()
class ANPR_Folder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using Device:", self.device)
        self.model = None

    def load_model(self, model_path):
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

    def Prep_Hitam_Putih(self,cropped_image, label):
        if label == 'NP-Hitam':
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            cropped_image = cv2.convertScaleAbs(cropped_image, alpha=0.85, beta=0.1)  # decrease contrast
            cropped_image = cv2.GaussianBlur(cropped_image,(2,2)) # apply blur to image
            cropped_image = cv2.bilateralFilter(cropped_image, d=4, sigmaColor=8, sigmaSpace=12)  # Bilateral filter for noise reduction
            cropped_image = cv2.threshold(cropped_image, 10, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] # apply binary treshold inverse
            cropped_image = cv2.bitwise_not(cropped_image)  # Invert colors
            cropped_image = cropped_image[0:cropped_image.shape[0] - 10, 0:cropped_image.shape[1]] #menghapus tanggal pajak
        elif label == 'NP-Putih':
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            cropped_image = cv2.convertScaleAbs(cropped_image, alpha=1.15, beta=0)  # Increase contrast
            cropped_image = cv2.GaussianBlur(cropped_image,(2,2)) #apply blur to image
            cropped_image = cv2.bilateralFilter(cropped_image, d=4, sigmaColor=8, sigmaSpace=12)  # Bilateral filter for noise reduction
            cropped_image = cv2.threshold(cropped_image, 10, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] # apply binary treshold inverse
            cropped_image = cropped_image[0:cropped_image.shape[0] - 10, 0:cropped_image.shape[1]] #menghapus tanggal pajak
        else:
            print("Label Tidak Diketahui")
            return None

        # Display the resulting cropped image
        return cropped_image
    def read_text(self, result_image):
        reader = easyocr.Reader(['id'], gpu=True)
        results = reader.readtext(result_image)
        return results

    # Function to sort OCR results based on X-axis coordinates
    def sort_by_x_axis(self,results):
        return sorted(results, key=lambda x: x[0][0][0])

    # Sort OCR results by X-axis coordinates
    def filter_results(self, results):
        sorted_results = self.sort_by_x_axis(results)
        plate_number = ''
        
        for result in sorted_results:
            text = re.sub(r'[^A-Za-z0-9]', '', result[1]).upper()  # Remove special characters
            
            # Condition to include shorter alpha sequences (<= 2 characters)
            if text.isalpha() and len(text.replace(" ", "")) <= 2:
                filter = ''.join([char if char.isalnum() or char == ' ' else '' for char in text]) + ' '
                plate_number += filter
            # Conditions for other types of sequences (digits, longer alpha sequences, etc.)
            elif len(text.replace(" ", "")) >= 4:
                filter = ''.join([char if char.isalnum() or char == ' ' else '' for char in text]) + ' '
                plate_number += filter
            elif text.isdigit() and len(plate_number.replace(" ", "")) <= 4:
                filter = ''.join([char if char.isalnum() or char == ' ' else '' for char in text]) + ' '
                plate_number += filter
            elif text.isalpha() and len(text.replace(" ", "")) <= 3:
                filter = ''.join([char if char.isalnum() or char == ' ' else '' for char in text]) + ' '
                plate_number += filter
            else:
                continue
        
        plate_number = plate_number.upper().strip().replace(" ", "")
        plate_number = re.sub(r"([A-Z])([0-9])", r"\1 \2", plate_number)
        plate_number = re.sub(r"([0-9])([A-Z])", r"\1 \2", plate_number)
        
        return plate_number

    def predict_image(self, image, model_path=None):
        if model_path:
            self.load_model(model_path)

        if self.model is None:
            raise ValueError("Model not loaded. Please provide a model path.")

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(img, (720, 720))

        with torch.no_grad():
            results = self.model(resized_img)

        predictions = results.pandas().xyxy[0]
        best_confidence = 0.0
        best_prediction = None

        for _, prediction in predictions.iterrows():
            x1, y1, x2, y2 = int(prediction['xmin']), int(prediction['ymin']), int(prediction['xmax']), int(prediction['ymax'])
            class_label = prediction['name']
            confidence = prediction['confidence']

            if confidence > best_confidence:
                best_confidence = confidence
                best_prediction = (x1, y1, x2, y2, class_label, confidence)

        if best_prediction:
            x1, y1, x2, y2, class_label, confidence = best_prediction

            color = (0, 255, 0)  # Default color for high confidence
            if confidence >= 0.60 and confidence <= 0.80:
                color = (255, 0, 0)  # Red color for moderate confidence
            elif confidence < 0.60:
                # Skip displaying this object if confidence is below 0.60
                best_prediction = None

        if best_prediction:
            # Draw bounding box on the image
            cv2.rectangle(resized_img, (x1, y1), (x2, y2), color, 2)
            text = f"{class_label}: {confidence:.2f}"
            cv2.putText(resized_img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
        # Crop the image using the bounding box coordinates
        cropped_image = resized_img[y1:y2, x1:x2]
        
        # Perform preprocessing on the cropped image
        result_image = self.Prep_Hitam_Putih(cropped_image, class_label)
        
        
        # Perform OCR on the cropped image
        read_text = self.read_text(result_image)
        result_ocr = self.filter_results(read_text)
        
        # Display text
        cv2.putText(resized_img, f"{result_ocr}", (x1 + 25, y2 + 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
        
        return cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR), result_ocr, class_label

