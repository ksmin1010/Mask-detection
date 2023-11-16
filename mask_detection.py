from tensorflow.keras.applications.mobilenet_v2 import preprocess_input 
from tensorflow.keras.preprocessing.image import img_to_array 
from tensorflow.keras.preprocessing.image import load_img 
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 

#한 부분의 성능을 높이면 다른 부분의 성능이 낮아지는  트레이드-오프(trade-off) 문제을 어느정도 해결하도록 MobileNetV2를 사용 
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input 
from tensorflow.keras.layers import Conv2D 
from tensorflow.keras.layers import AveragePooling2D 
from tensorflow.keras.layers import Flatten 
from tensorflow.keras.layers import Dense 
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.models import Model 
from sklearn.preprocessing import LabelBinarizer 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report 

from imutils import paths 
import matplotlib.pyplot as plt 
import numpy as np 
import os 
dataset = './dataset'
plot = 'result_plot.jpg' 

#오픈소스를 변형시켜 만든 face Mask Detector 모델을 학습을 완료시킨 상태로 가져와 구동
model_name = 'mask_detector.h5'  
init_learning_rate = 1e-4
epochs = 20
batch_size = 32 

print("[이미지(dataset) 로딩]")
image_paths = list(paths.list_images(dataset)) -데이터셋을 읽어들임 
data = [] 
labels = [] 



#이미지 전처리를 실행.
for image_path in image_paths:
    
    label = image_path.split(os.path.sep)[-2]

    image = load_img(image_path, target_size=(224, 224)) 
    image = img_to_array(image) 
    image = preprocess_input(image) 

    data.append(image)
    labels.append(label)


data = np.array(data, dtype="float32")
labels = np.array(labels)

#마스크 착용여부 메시지를 레이블 이진화를 통해 출력하도록 한다. 
label_binarizer = LabelBinarizer(
labels = label_binarizer.fit_transform(labels) 
labels = to_categorical(labels) 

#학습데이터는 80%, 테스트데이터는  dataset 20% 로 분리시켜 데이터 분할한다. 

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

image_data_generator = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest")


output_model = Conv2D(32, (5, 5), padding="same", activation="relu")(output_model) # Convolution Layer

output_model = AveragePooling2D(pool_size=(5, 5), strides=1, padding="same")(output_model) # Average Pooling Layer
output_model = Flatten(name="flatten")(output_model) # 2차원 데이터로 이루어진 추출된 특징을 Dense Layer 에서 학습하기 위해 1차원 데이터로 변경

output_model = Dense(32, activation="relu")(output_model) # 출력 뉴런이 32 개 이고 활성화 함수가 relu
output_model = Dense(64, activation="relu")(output_model) # 출력 뉴런이 64 개 이고 활성화 함수가 relu
output_model = Dropout(0.5)(output_model) # 50% Dropout 적용(dropout을 적용할 비율)
output_model = Dense(32, activation="relu")(output_model) # 출력 뉴런이 32 개 이고 활성화 함수가 relu
output_model = Dense(2, activation="softmax")(output_model) # [Output Layer] 출력 뉴런이 2 개 이고 활성화 함수가 softmax

model = Model(inputs=input_model.input, outputs=output_model)


for layer in input_model.layers:
    layer.trainable = False

print("[모델 컴파일]")

print("[모델 학습]")
train = model.fit(
        image_data_generator.flow(trainX, trainY, batch_size=batch_size),
        steps_per_epoch=len(trainX) // batch_size,
        validation_data=(testX, testY),
        validation_steps=len(testX) // batch_size,
        epochs=epochs)


print("[모델 평가]")

predict_index = np.argmax(predict, axis=1)

print(classification_report(testY.argmax(axis=1), predict_index, target_names=label_binarizer.classes_))

print("[모델 저장]")

model.save(model_name, save_format="h5")

n = epochs
plt.style.use("ggplot") # 스타일 적용
plt.figure() # 새로운 figure 생성

plt.plot(np.arange(0, n), train.history["loss"], label="train_loss")
plt.plot(np.arange(0, n), train.history["val_loss"], label="val_loss") 
plt.plot(np.arange(0, n), train.history["acc"], label="train_accuracy") 
plt.plot(np.arange(0, n), train.history["val_acc"], label="val_accuracy") 
plt.title("Training Loss and Accuracy") 
plt.xlabel("epoch") 
plt.ylabel("Loss / Accuracy") 
plt.legend(loc="lower left") 
plt.savefig(plot) 
