import cv2
import numpy as np
import os

#~ Tạo menu chọn video của mỗi người để nhận diện
while True:
    print('Menu:')
    print('[1]  -  Thiện')
    print('[2]  -  Khoa')
    print('[3]  -  Tú')
    print('[4]  -  Tâm')
    print('[5]  -  Linh')
    chon = input('Chọn video: ')
    if chon == '1':
        path = 'thien'
        id = chon
        break
    elif chon == '2':
        path = 'khoa'
        id = chon
        break
    elif chon == '3':
        path = 'tu'
        id = chon
        break
    elif chon == '4':
        path = 'tam'
        id = chon
        break
    elif chon == '5':
        path = 'linh'
        id = chon
        break
    else:
        print('Bạn nhập sai yêu cầu!')

#~ Tạo ra một đối tượng nhận dạng khuôn mặt sử dụng thuật toán LBPHFaceRecognizer trong OpenCV để nhận dạng khuôn mặt.
recognizer = cv2.face.LBPHFaceRecognizer_create()
#~ Đọc dữ liệu đã train trước đó trong trainer
recognizer.read('trainers/trainer{}.yml'.format(chon))
cascadePath = 'haarcascades/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascadePath)

#~ Thiết lập font chữ
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0

#~ Tên của các ID 
ten = ['0','Thien','Khoa','Tu','Tam','Linh']

#~ Khởi tạo camera
cam = cv2.VideoCapture(path+'.mp4')
if not cam.isOpened():
    print("Không thể mở video: {}.mp4".format(path))
    exit()

while True:
    ret, img = cam.read()
    if not ret or img is None:
        print("Video đã chạy hết ")
        break  # THOÁT khỏi vòng lặp ngay khi hết video

    img = cv2.flip(img, 1)
    
    # Chỉ chuyển sang grayscale nếu img hợp lệ
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.3,
        minNeighbors = 5,
    )
    #~ Lặp qua từng khuôn mặt được phát hiện trong ảnh và thực hiện các bước xử lý ảnh để nhận diện khuôn mặt và hiển thị kết quả lên ảnh.
    for (x, y, w, h) in faces:

        #~ Được sử dụng để vẽ một hình chữ nhật quanh khuôn mặt được phát hiện
        cv2.rectangle(img , (x, y), (x+w, y+h), (0, 255, 0), 2)

        #~ Nhận diện khuôn mặt dựa trên dữ liệu đã được train trước đó và trả về ID và độ chính xác.
        id, do_chinh_xac = recognizer.predict(gray[y+2:y+h-2, x+2:x+w-2])

        if (do_chinh_xac < 100):
            id = ten[id]
            do_chinh_xac = ' {0}%'.format(round(100 - do_chinh_xac ))
        else:
            id = 'Không thể nhận diện'
            do_chinh_xac = ' {0}%'.format(round(100 - do_chinh_xac ))

        cv2.putText(img, str(id), (x+5, y-5), font, 1, (0, 0, 255), 2)
        cv2.putText(img, str(do_chinh_xac), (x+5, y+h-5), font, 1, (0, 0, 255), 2)

    cv2.imshow('Nhận diện khuôn mặt ', img)

    k = cv2.waitKey(10) & 0xff #Bấm ESC để thoát
    if k == 27:
        break

print('Thoát chương trình')
cam.release()
cv2.destroyAllWindows()

