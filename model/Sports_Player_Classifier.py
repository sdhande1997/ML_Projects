import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt
import os
import shutil
import pywt               #py wavelet transformation
import cv2
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import joblib


############################ Loading a image to test ################################################

img = cv2.imread("./test_images/kobe_bryant2.jpg",1)
print("Image Shape: ",img.shape)
# cv2.imshow("Image1",img)

face_cascade = cv2.CascadeClassifier("./opencv/haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("./opencv/haarcascades/haarcascade_eye.xml")

############################ Getting Faces ################################################


def get_cropped_face_img(image_path):
    color_img = cv2.imread(image_path)
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = color_img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >=2:
            return roi_color


cropped_image = get_cropped_face_img("./test_images/kobe_bryant1.jpg")
# cv2.imshow("cropped_image",cropped_image)

path_to_data = "./dataset/"
path_to_crp_data = "./processed/"

img_dirs = []                                        #./dataset/kobe_bryant
for entry in os.scandir(path_to_data):            #get the directories in dataset folder
    if entry.is_dir():
        img_dirs.append(entry.path)
print("Directories:",img_dirs)

if os.path.exists(path_to_crp_data):            #make a new folder "processed" if it already exists remove and remake
    shutil.rmtree(path_to_crp_data)
os.mkdir(path_to_crp_data)

cropped_img_dirs = []
sports_person_file_name_dict = {}

################## Function to loop through all images in directory ################################################


for img_dir in img_dirs:
    count = 1
    sports_person = img_dir.split('/')[-1]
    sports_person_file_name_dict[sports_person] = []

    for entry in os.scandir(img_dir):
        roi_color = get_cropped_face_img(entry.path)
        if roi_color is not None:
            cropped_folder = path_to_crp_data + sports_person   #processed/ + kobe_bryant
            if not os.path.exists(cropped_folder):
                os.mkdir(cropped_folder)
                cropped_img_dirs.append(cropped_folder)
                print("Generating cropped images in folder: ", cropped_folder)

            cropped_file_name = sports_person + str(count) + ".png"   #kobe_bryant + count +.png
            print(cropped_file_name)
            cropped_file_path = cropped_folder + "/" + cropped_file_name  #processed/ +kobe_bryant + / + kobe_bryant + count +.png

            cv2.imwrite(cropped_file_path, roi_color)
            sports_person_file_name_dict[sports_person].append(cropped_file_path)
            count += 1
print("Processing Finish")

############################ Wavelet Transformation ################################################


def w2d(img, mode = "haar", level=1):   #haar= square shape functions which together form a wavelet family or basis.
    imArray = img
             # DataType Conversion
             # convert to grayscale
    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
             # convert to float
    imArray = np.float32(imArray)       #np.float32 reduces the decimal values to just single digit
    # print("imArray: ",imArray)
    imArray /= 255
    #compute the coefficients
    coeffs = pywt.wavedec2(imArray, mode, level = level)        #2d multilevel decomposition using wavedec2

    #Process Coefficients
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0

    #reconstruction
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)

    return imArray_H


im_har = w2d(cropped_image, 'db1', 5)
# cv2.imshow("im_har", im_har)

############################ Creating a dictionary of sports_person ################################################
print("Dictionary", sports_person_file_name_dict)
class_dict = {}
count = 0
for sports_celebrity in sports_person_file_name_dict.keys():
    class_dict[sports_celebrity] = count
    count = count + 1
print("Class_Dict",class_dict)


############################ Getting data for X and Y train, test set ################################################

X = []
Y = []
for sports_celebrity, training_files in sports_person_file_name_dict.items():
    for training_image in training_files:
        img = cv2.imread(training_image)
        if img is None:
            continue
        scaled_raw_img = cv2.resize(img, (32,32))
        img_har = w2d(img, 'db1', 5)
        scaled_img_har = cv2.resize(img_har, (32,32))
        combined_img = np.vstack((scaled_raw_img.reshape(32*32*3,1), scaled_img_har.reshape(32*32, 1)))
        X.append(combined_img)
        Y.append(class_dict[sports_celebrity])

print("Length of X", len(X[0]))
X= np.array(X).reshape(len(X), 4096).astype(float)
print("X shape", X.shape)

############################ Machine Learning Model ################################################

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel = 'rbf', C = 10))])
pipe.fit(X_train, Y_train)
score = pipe.score(X_test, Y_test)
print("Score: ", score)

model_params = {
    'svm': {
        'model': svm.SVC(gamma = 'auto', probability=True),
        'params': {
            'svc__C':[1,10,100,1000],
            'svc__kernel': ['rbf','linear']
        }
    },
    'random_forest':{
        'model': RandomForestClassifier(),
        'params': {
            'randomforestclassifier__n_estimators':[1,5,10]
        }
    },
    'logistic_regression':{
        'model':LogisticRegression(solver='liblinear', multi_class='auto'),
        'params':{
            'logisticregression__C': [1,5,10]
        }
    }
}
############################ Choosing Best Model ################################################
scores = []
best_estimators = {}
for algo, mp in model_params.items():
    pipe = make_pipeline(StandardScaler(),mp['model'])
    clf = GridSearchCV(pipe, mp['params'], cv=5, return_train_score=False)
    clf.fit(X_train, Y_train)
    scores.append({
        'model': algo,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    best_estimators[algo] = clf.best_estimator_

df_best_score = pd.DataFrame(scores, columns=['model', 'best_Score'])
df_best_params = pd.DataFrame(scores, columns=['model', 'best_params'])
print(df_best_score)
print(df_best_params)

print("SVM accuracy:",best_estimators['svm'].score(X_test, Y_test))
print("Random_Forest accuracy:",best_estimators['random_forest'].score(X_test, Y_test))
print("Logistic_Regression accuracy:",best_estimators['logistic_regression'].score(X_test, Y_test))

best_clf = best_estimators['svm']
cm = confusion_matrix(Y_test, best_clf.predict(X_test))

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()
############################ Saving the best model ################################################
joblib.dump(best_clf, 'saved_model.pkl')

cv2.waitKey(0)
cv2.destroyAllWindows()