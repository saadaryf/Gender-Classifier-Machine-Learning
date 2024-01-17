from flask import render_template, request, Response
import os
import cv2
from app.face_recognition import faceRecognitionPipeline
import matplotlib.image as matimg


UPLOAD_FOLDER = 'static/uploads'


def index():
    return render_template("index.html")

def app():
    return render_template("app.html")

def gender_app():
    if request.method == 'POST':
        f = request.files['image-name']
        filename = f.filename
        path = os.path.join(UPLOAD_FOLDER,filename)
        f.save(path) #saves img to uploads folder
        #getting predictions
        pred_img, predictions = faceRecognitionPipeline(path)
        pred_filename = 'prediction_image.jpg'
        cv2.imwrite(f'./static/predict/{pred_filename}',pred_img)
        #generating report
        report = []
        for i, obj in enumerate(predictions):
            gray_image = obj['roi'] # grayscale image (array)
            eigen_image = obj['eig_img'].reshape(100,100) #eigne image (array)
            gender_name = obj['prediction_name'] #name
            score = round(obj['score']*100,2) #probability score
            #saving grayscale and eigne img to predict folder
            gray_image_name = f'roi_{i}.jpg'
            eigen_image_name = f'eigen_{i}.jpg'
            matimg.imsave(f'./static/predict/{gray_image_name}',gray_image,cmap='gray')
            matimg.imsave(f'./static/predict/{eigen_image_name}',eigen_image,cmap='gray')
            
            #saving report
            report.append([gray_image_name,eigen_image_name,gender_name,score])
        
        return render_template("gender.html",fileUpload=True,report=report) #Post Request

            
    return render_template("gender.html",fileUpload=False) #Get Request

def video():
    cap = cv2.VideoCapture(0)
    while True:
        ret , frame = cap.read()
        if ret == False:
            break
        pred_img, pred_dict = faceRecognitionPipeline(frame,path=False)
        cv2.imshow('prediction', pred_img)
        if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == ord('Q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return render_template("gender.html")