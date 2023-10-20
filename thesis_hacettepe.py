# ÖNCEDEN YÜKLENMİŞ MODÜLLERİN İMPORT EDİLMESİ

import mediapipe as mp  # MediaPipe, canlı ve akışlı medya için platformlar arası, özelleştirilebilir machine learning çözümleri sunar.
# özellikle yüz, vücut pozu ve ellerin algılanmasını otomatik olarak gerçekleştirmektedir.
# internet sitesi: https://pypi.org/project/mediapipe/
import cv2  #
import numpy as np  # özellikle farklı datasetleri yapılandırmada ihtiyaç duyulacaktır.
import os  # klasör ve path'larla çalışmayı kolaylaştırmaktadır.
from matplotlib import pyplot as plt  # özellikle imshow forksiyonu sayesinde resimleri görmeyi kolaylaştıracaktır.
import \
    time  # kullanacağımız görüntülerde belli zmaan ararlıklarında fotoğraf almamızı ve dataset oluşturmamızı sağlayacaktırfvt 5.

# MEDIAPIPE KULLANARAK ANAHTAR NOKTALARIN BELİRLENMESİ
mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Çizim hizmeti için kullanılmaktadır.


def mediapipe_detection(image, model):
    # image webcam'den alınacaktır.
    # model ise mediapipe holistic modelidir
    # mediapipe'ta bir görüntüyü yakalamak için önce opencv modülü kullanılarak (cv2) RGB'den BGR'a dönüştürmek lazım.
    # daha sonra unwritable yaparak memory tasarrufu sağlarız çünkü webcam'den alınacak bir video akışında sürekli veri akışı olacaktır (https://github.com/google/mediapipe/issues/736)
    # Daha sonra mediapipe'a daha rahat işlemek için bir değişkene atanacaktır (results).
    # Daha sonra tekrar yazılabilir duruma getirilecektir.
    # Sonrasında ise ekranda sunulacak görüntünün gerçeğe uygun olması için tekrar renk dönüşümü yapılacaktır.

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB'DEN BGR'YE RENK DÖNÜŞÜMÜ
    image.flags.writeable = False  # Görüntü yazılamaz hale getirildi
    results = model.process(image)  # Opencv'den görüntüyü mediapipe'a işlemek için results değişkenine set ettim.
    image.flags.writeable = True  # resmi yazılabilir hale getirmek
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # RGB'DEN BGR'YE RENK DÖNÜŞÜMÜ
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # Sol el bağlantılarını kurmak
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # Sağ el bağlantılarını kurmak
    mp_drawing.draw_landmarks(image, results.pose_landmarks,
                              mp_holistic.POSE_CONNECTIONS)  # Poz bağlantılarını kurmak



def draw_styled_landmarks(image, results):
    # yüz bağlarını belirlemek
    # burada görüntü ve mediapipe modülünden aldığımız değerleri argüman olarak fonksiyona dahil edeceğiz.

    # sol el bağlarını belirlemek

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(10, 220, 60), thickness=2, circle_radius=4),
                              # poz sınırının çizgi ya da noktaların renk, kalınlık ve yarıçap bilgileri.
                              mp_drawing.DrawingSpec(color=(80, 10, 21), thickness=2, circle_radius=2))
                              # poz birleşim noktalarının çizgi ya da noktaların renk, kalınlık ve yarıçap bilgileri.

    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              # sol el sınırının çizgi ya da noktaların renk, kalınlık ve yarıçap bilgileri.
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              # sol el birleşim noktalarının çizgi ya da noktaların renk, kalınlık ve yarıçap bilgileri.
                              )
    # sağ el bağlarını belirlemek
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              # sağ el sınırının çizgi ya da noktaların renk, kalınlık ve yarıçap bilgileri.
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              # sağ el birleşim noktalarının çizgi ya da noktaların renk, kalınlık ve yarıçap bilgileri.
                              )

def extract_keypoints(results):

    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, lh, rh])




# def folder_count(directory_path):
#
#     count = 0
#
#     for item in os.listdir(directory_path):
#             item_path = os.path.join(directory_path, item)
#             if os.path.isdir(item_path):
#                 count += 1
#     return count

def file_count(directory_path):



    for item in os.listdir(directory_path):
        count = 0
        for j in os.listdir(directory_path + "\{}".format(item)):
            item_path = directory_path + "\{}\{}".format(item, j)
            if os.path.isfile(item_path):
                count += 1
        return count

DATA_PATH = r"C:\Users\MonsterPC\Desktop\AUTSL_parameter"

# file_count(my_file_path)

# def frames(url):
#     cap = cv2.VideoCapture(url)
#     length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     return length


# my_file_path = r"C:\Users\MonsterPC\Desktop\HedefMutlu"
#
# print(file_count(my_file_path))
#
# DATA_PATH = os.path.join('50labels')


actions = np.array(os.listdir(r"C:\Users\MonsterPC\Desktop\AUTSL_parameter"))

# count = 0
# for action in actions:
#     try:
#         os.makedirs(DATA_PATH + "\{}".format(action)),
#     except:
#         pass
#         for sequence in os.listdir(my_file_path):
#             try:
#                 os.makedirs(DATA_PATH + "\{}".format(sequence))
#             except:
#                 pass
#                 for m in os.listdir(my_file_path + "\{}".format(sequence)):
#                     try:
#                         os.makedirs(DATA_PATH + "\{}\{}".format(sequence, m.replace(".mp4", "")))
#                     except:
#                         pass

# m.replace(".mp4", "")))

#
# def frames(url):
#     cap = cv2.VideoCapture(url)
#     length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     return length
#
# import pandas as pd
# for i in os.listdir(my_file_path):
#         for j in range(len(os.listdir(my_file_path+"\{}".format(i)))):
#             my_newest_file_path = my_file_path+r"\{}\{}.mp4".format(i,j)
#             try:
#                 cap = cv2.VideoCapture(my_newest_file_path)
#                 with mp_holistic.Holistic(min_detection_confidence=0.5,
#                                           min_tracking_confidence=0.5) as holistic:  # burada mediapipe önce key point'leri detection ile yakalıyor ve tracking ile takip ediyor.
#
#                             my_frames = frames(my_newest_file_path)
#                             for t in range(my_frames):
#
#                                 # mediapipe modelini ayarlamak
#                                 ret, frame = cap.read()  # Video karesinin buraya döndürüldüğü resim. Hiçbir çerçeve yakalanmamışsa, görüntü boş olacaktır.
#
#                                 # Algılama yapma
#                                 image, results = mediapipe_detection(frame, holistic)
#                                 # el ve yüz için sınırları çizmek
#                                 draw_styled_landmarks(image, results)
#
#                                 keypoints = extract_keypoints(results)
#                                     # print(keypoints)
#
#                                 # Display the resulting frame
#
#
#
#                                 npy_path = os.path.join(DATA_PATH, i, str(j), str(t))
#
#                                 np.save(npy_path, keypoints)
#                             if cv2.waitKey(10) & 0xFF == ord('q'):
#                                 break
#
#                 cap.release()
#                 cv2.destroyAllWindows()
#             except:
#                 pass


        # GÖRÜNTÜDEN ANAHTAR NOKTALARI ALMAK


# result_test = extract_keypoints(results)
#
# np.save("0", result_test)
# np.load("0.npy")
#
# DATA_PATH = os.path.join('26082023X')





# # VERİSETİ OLUŞTURMAK İÇİN KLASÖRLERİN AYARLANMASI
# # Dışa aktarılan veriler için yol, numpy array'ları
# DATA_PATH = os.path.join('26082023')
# my_file_path = r"C:\Users\MonsterPC\Desktop\Videolar"
# # tanınmasını istediğimiz ifadeler
# actions = np.array(my_file_path)

# # 30 video değerinde veri
# no_sequences = 30
#
# # Videos are going to be 30 frames in length
sequence_length = 38
#
# for action in actions:
#     for sequence in range(no_sequences):
#         try:
#             os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
#         except:
#             pass
#
#
# cap = cv2.VideoCapture(my_file_path)
# # Set mediapipe model
# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#     # NEW LOOP
#     # Loop through actions
#     for action in actions:
#         # Loop through sequences aka videos
#         for sequence in range(no_sequences):
#             # Loop through video length aka sequence length
#             for frame_num in range(sequence_length):

#                 # Read feed
#                 ret, frame = cap.read()
#
#                 # Make detections
#                 image, results = mediapipe_detection(frame, holistic)
#                 #                 print(results)
#
#                 # Draw landmarks
#                 draw_styled_landmarks(image, results)
#
#                 # NEW Export keypoints
#                 keypoints = extract_keypoints(results)
#                 npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
#                 np.save(npy_path, keypoints)
#
#                 # Break gracefully
#                 if cv2.waitKey(10) & 0xFF == ord('q'):
#                     break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#
#
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

label_map = {label: num for num, label in enumerate(actions)}
                            #Temel olarak, farklı eylemlerimizin her birini temsil etmek için bir label array veya bir label dictionary oluşturacağız.
                            #az önce gidip yaptığımız şey bir dictionary yaratmaktır. dictionary'mizde sadece label kelimesi var
                            #bizim örneğimizde "merhaba" 0. index'le label'landı.
                            #Aynı şekilde "özür dilerim" 1. index ile ve "teşekkürler" kelimesi ise 2. index ile label'landı.


print(label_map)

sequences, labels = [], []  #burada sequence3s ve labels olmak üzere 2 tane boş array tanımladık.
                            #sequences feature data ya da X datamızı, labels ise label'larımızı ya da y datamızı temsil edecek.
                            #burada feature'larımızı kullanacağız ve label'larımız arasındaki  ilişkiyi
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, str(action)))).astype(int):
                            #her sequence 30 video anlamına gelecektir.

        window = []
                            #Window listesi belirli dizi için sahip olduğumuz tüm farklı frame'leri temsil etmektedir.
        for frame_num in range(sequence_length):
                            #burada her frame'in içerisinde döngü yapıyoruz.

            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                            #daha sonra bu frame'i yüklemek için np.load'ı kullanıyoruz.
                            #Bunu yapmak için, farklı numpy dizilerimize giden tam yoldan geçiyoruz.
            window.append(res)
                            #burası genel olarak şu anlama geliyor: 0 indeksli frame'i yakala, window listesine ekle, 1 indeksli frame'i yakala, window listesine ekle

        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# LSTM Neural Network Oluşturma ve Eğitimi
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)


model = Sequential()


model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(38, 258)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

print(X.shape)

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=5, callbacks=[tb_callback])
print(model.summary())
print(model.compile(metrics=['categorical_accuracy']))

# # # TAHMİN  YAPMA
# res = model.predict(X_test)
# print(actions[np.argmax(res[4])])
# print(actions[np.argmax(y_test[4])])

# # # AĞIRLIKLARI KAYDETMEK
# model.save('action.h5')
#
# from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
#
# yhat = model.predict(X_test)
# ytrue = np.argmax(y_test, axis=1).tolist()
# yhat = np.argmax(yhat, axis=1).tolist()
# print(multilabel_confusion_matrix(ytrue, yhat))
#
# # TEST IN REALTIME
#
# colors = [(20,117,116), (20,117,116), (20,117,116), (20,117,116), (20,117,116), (20,117,116), (20,117,116), (20,117,116), (20,117,116), (20,117,116)]
#
# # 1. New detection variables
# sequence = []
# sentence = []
# predictions = []
# threshold = 0.5
#
# def prob_viz(res, actions, input_frame, colors):
#     output_frame = input_frame.copy()
#     for num, prob in enumerate(res):
#         cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
#         cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
#
#     return output_frame
#
# cap = cv2.VideoCapture(0)
# # Set mediapipe model
# with mp_holistic.Holistic(min_detection_confidence=0.8, min_tracking_confidence=0.8) as holistic:
#     while cap.isOpened():
#
#         # Read feed
#         ret, frame = cap.read()
#
#         # Make detections
#         image, results = mediapipe_detection(frame, holistic)
#         print(results)
#
#         # Draw landmarks
#         draw_styled_landmarks(image, results)
#
#         # 2. Prediction logic
#         keypoints = extract_keypoints(results)
#         sequence.append(keypoints)
#         sequence = sequence[-30:]
#
#         if len(sequence) == 30:
#             res = model.predict(np.expand_dims(sequence, axis=0))[0]
#             print(actions[np.argmax(res)])
#             predictions.append(np.argmax(res))
#
#             # 3. Viz logic
#             if np.unique(predictions[-10:])[0] == np.argmax(res):
#                 if res[np.argmax(res)] > threshold:
#
#                     if len(sentence) > 0:
#                         if actions[np.argmax(res)] != sentence[-1]:
#                             sentence.append(actions[np.argmax(res)])
#                     else:
#                         sentence.append(actions[np.argmax(res)])
#
#             if len(sentence) > 5:
#                 sentence = sentence[-5:]
#
#             # Viz probabilities
#             image = prob_viz(res, actions, image, colors)
#
#         cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
#         cv2.putText(image, ' '.join(sentence), (3, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#
#         # Show to screen
#         cv2.imshow('OpenCV Feed', image)
#
#         # Break gracefully
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()
#
# plt.figure(figsize=(18, 18))
# plt.imshow(prob_viz(res, actions, image, colors))
