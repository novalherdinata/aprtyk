import streamlit as st
import cv2
import torch
import numpy as np
import easyocr
import re
import pytesseract
from pathlib import Path
from ANPR_Db import *

@st.cache_resource()

class ANPR:
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
            sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpen = cv2.filter2D(cropped_image, -1, sharpen_kernel)
            st.image(cropped_image, caption='Grayscale Image', use_column_width=True)
            st.write("Hasil konversi gambar dari RGB ke Grayscale.")
            cropped_image = cv2.convertScaleAbs(cropped_image, alpha=0.85, beta=0.1)  # decrease contrast
            st.image(cropped_image, caption='Decrease Contrast Image', use_column_width=True)
            st.write("Hasil penurunan kontras dari gambar yang telah dikonversi ke Grayscale.")
            cropped_image = cv2.GaussianBlur(cropped_image,(1,1), 0) # apply blur to image
            st.image(cropped_image, caption='Blur Image', use_column_width=True)
            st.write("Hasil blur dari gambar yang telah dikonversi ke Grayscale dan diturunkan kontrasnya.")
            cropped_image = cv2.bilateralFilter(cropped_image, d=4, sigmaColor=8, sigmaSpace=12)  # Bilateral filter for noise reduction
            st.image(cropped_image, caption='Bilateral Filter Image', use_column_width=True)
            st.write("Hasil bilateral filter dari gambar yang telah dikonversi ke Grayscale, diturunkan kontrasnya, dan di blur.")
            cropped_image = cv2.threshold(cropped_image, 10, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] # apply binary treshold inverse
            st.image(cropped_image, caption='Binary Treshold Inverse Image', use_column_width=True)
            st.write("Hasil binary treshold inverse dari gambar yang telah dikonversi ke Grayscale, diturunkan kontrasnya, di blur, dan di bilateral filter.")
            cropped_image = cv2.bitwise_not(cropped_image)  # Invert colors
            st.image(cropped_image, caption='Invert Colors Image', use_column_width=True)
            st.write("Hasil invert colors dari gambar yang telah dikonversi ke Grayscale, diturunkan kontrasnya, di blur, di bilateral filter, dan di binary treshold inverse.")
            cropped_image = cropped_image[0:cropped_image.shape[0] - 10, 0:cropped_image.shape[1]] #menghapus tanggal pajak
        elif label == 'NP-Putih':
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpen = cv2.filter2D(cropped_image, -1, sharpen_kernel)
            st.image(cropped_image, caption='Grayscale Image', use_column_width=True)
            st.write("Hasil konversi gambar dari RGB ke Grayscale.")
            cropped_image = cv2.convertScaleAbs(cropped_image, alpha=1.15, beta=0)  # Increase contrast
            st.image(cropped_image, caption='Increase Contrast Image', use_column_width=True)
            st.write("Hasil peningkatan kontras dari gambar yang telah dikonversi ke Grayscale.")
            cropped_image = cv2.GaussianBlur(cropped_image,(1,1), 0) #apply blur to image
            st.image(cropped_image, caption='Blur Image', use_column_width=True)
            st.write("Hasil blur dari gambar yang telah dikonversi ke Grayscale dan ditingkatkan kontrasnya.")
            cropped_image = cv2.bilateralFilter(cropped_image, d=4, sigmaColor=8, sigmaSpace=12)  # Bilateral filter for noise reduction
            st.image(cropped_image, caption='Bilateral Filter Image', use_column_width=True)
            st.write("Hasil bilateral filter dari gambar yang telah dikonversi ke Grayscale, ditingkatkan kontrasnya, dan di blur.")
            cropped_image = cv2.threshold(cropped_image, 10, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] # apply binary treshold inverse
            st.image(cropped_image, caption='Binary Treshold Image', use_column_width=True)
            st.write("Hasil binary treshold dari gambar yang telah dikonversi ke Grayscale, ditingkatkan kontrasnya, di blur, dan di bilateral filter.")
            cropped_image = cropped_image[0:cropped_image.shape[0] - 10, 0:cropped_image.shape[1]] #menghapus tanggal pajak
            
        else:
            print("Label Tidak Diketahui")
            return None

        # Display the resulting cropped image
        return cropped_image
    def read_text(self, result_image):
        reader = easyocr.Reader(['id'], gpu=True)
        results = reader.readtext(result_image)
        # results = pytesseract.image_to_string(result_image)
        pytesseract.image_to_string(result_image, lang='eng', config='--psm 6')
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

    def predict_image(self, image, path, model_path=None):
        
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
            cv2.putText(resized_img, "Plat Nomor", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # Display the resulting image & Description
        
        db = connect_db()
        checkid = db.read_check_id(path)
                    # st.text_input("Hasil OCR Plat Nomor", value=result_anpr)
        # print(checkid)
        if checkid[0] == 29:
            st.subheader("Hasil Deteksi Plat Nomor: Tidak Terdeteksi")
            return resized_img, cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR), "1"
        else:       
            st.subheader("Hasil Deteksi Plat Nomor")
            st.write("kotak berwarna hijau merupakan bounding box (BBox) merupakan prediksi terbaik dari model yang telah dilatih YoloV8 dan memiliki confidence tertinggi. BBox tersebut selanjutnya akan dimasking & crop untuk proses OCR menggunakan Tesseract OCR.")
            col1, col2 = st.columns(2)
            with col1:
                st.text_input("Hasil Prediksi Class", "Plat Nomor")
            with col2:
                st.text_input("Hasil Prediksi Confidence", confidence)

            st.image(resized_img, caption='Annotated Image', use_column_width=True) 
            # Create a mask
            mask = np.zeros(resized_img.shape[:2], dtype=np.uint8)
            mask[y1:y2, x1:x2] = 100  # Set the region inside the bounding box to 255
            # Apply the mask to the image
            masked_image = cv2.bitwise_and(resized_img, resized_img, mask=mask)
            # Display the resulting masked image
            st.subheader("Hasil Masking")
            st.write("Gambar dibawah ini merupakan hasil masking dari gambar yang telah di deteksi.")
            st.image(masked_image, caption='Masked Image', use_column_width=True)
                    
            # Crop the image using the bounding box coordinates
            cropped_image = resized_img[y1:y2, x1:x2]

            # Display the resulting cropped image
            st.subheader("Hasil Cropping")
            st.write("Gambar dibawah ini merupakan hasil cropping dari gambar yang telah di masking.")    
            st.image(cropped_image, caption='Cropped Image', use_column_width=True)        
            # Perform preprocessing on the cropped image
            st.subheader("Pre-processing")
            st.write("Gambar dibawah ini merupakan proses pre-processing dari cropping gambar. Proses pre-processing yang dilakukan adalah konversi gambar dari RGB ke Grayscale, blur, penurunan kontras, binary treshold inverse, bilateral filter dan invert colors.")
            result_image = self.Prep_Hitam_Putih(cropped_image, class_label)
            st.subheader("Hasil Pre-processing")
            st.write("Gambar dibawah ini merupakan hasil pre-processing dari gambar yang telah dicrop.")
            st.image(result_image, caption='Preprocessed Image', use_column_width=True)
            
            # Perform OCR on the cropped image
            read_text = self.read_text(result_image)
            result_ocr = self.filter_results(read_text)
            st.subheader("Hasil Pengenalan Karakter")
        
                #compare data
            statust = db.read_yolo_state(path)
            # statust2 =''.join(statust)
            print(statust)

            if statust[0] == 0:
                st.write("Hasil pengenalan karakter dari YOLO:")
                compare1 = db.read_compare_data(path)
                    # st.text_input("Hasil OCR Plat Nomor", value=result_anpr)
                textCompare1 =''.join(compare1)
                # Display text
                cv2.putText(resized_img, f"{textCompare1}", (x1 + 25, y2 + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
                
                st.image(resized_img, caption='Final Image YOLO', use_column_width=True)
                return resized_img, cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR), result_ocr
            else:
                st.write("Hasil pengenalan karakter dari YOLO")
                compare = db.read_compare_data_yolo(path)
                    # st.text_input("Hasil OCR Plat Nomor", value=result_anpr)
                textCompare =''.join(compare)
                # Display text
                cv2.putText(resized_img, f"{textCompare}", (x1 + 25, y2 + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
                st.image(resized_img, caption='Final Image YOLO', use_column_width=True)
                st.write("Hasil pengenalan karakter dari YOLO dan Tesseract OCR:")
                compare1 = db.read_compare_data(path)
                    # st.text_input("Hasil OCR Plat Nomor", value=result_anpr)
                textCompare1 =''.join(compare1)
                # Display text
                cv2.putText(resized_img, f"{textCompare1}", (x1 + 25, y2 + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
                
                st.image(resized_img, caption='Final Image YOLO dan Tesseract OCR', use_column_width=True)
            # db.__del__()
            
                return resized_img, cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR), result_ocr


