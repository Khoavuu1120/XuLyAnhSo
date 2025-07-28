import cv2
import numpy as np
from PIL import Image
import os

#~ Vòng lặp vô hạn được sử dụng để liên tục hiển thị menu cho đến khi người dùng chọn đúng thư mục.
#~ Mỗi lần người dùng chọn một số tương ứng với tên của một thư mục, biến `folder` được gán bằng tên đường dẫn của thư mục đó.
while True:
	print('Menu thư mục:')
	print('[1]  -  Thiện')
	print('[2]  -  Khoa')
	print('[3]  -  Tú')
	print('[4]  -  Tâm')
	print('[5]  -  Linh')
	folder = input('Chọn thư mục (1, 2, 3, 4, 5): ')
	if folder == '1':
			path = 'hinh_anh/thien'
			break
	elif folder == '2':
			path = 'hinh_anh/khoa'
			break
	elif folder == '3':
			path = 'hinh_anh/tu'
			break
	elif folder == '4':
			path = 'hinh_anh/tam'
			break
	elif folder == '5':
			path = 'hinh_anh/linh'
			break
	else:
			print('Bạn nhập sai yêu cầu!')

#~ Tạo một đối tượng nhận dạng khuôn mặt dựa trên thuật toán Local Binary Patterns Histograms (LBPH) bằng cách sử dụng lớp `cv2.face.LBPHFaceRecognizer_create()` của thư viện OpenCV
#~ Đối tượng này được sử dụng để train mô hình 
recognizer = cv2.face.LBPHFaceRecognizer_create()

#~ Tạo một đối tượng `CascadeClassifier` để phát hiện khuôn mặt trong ảnh bằng cách sử dụng mô hình phân loại phân cấp (cascade classifier). 
#~ Đối tượng này sẽ được sử dụng để phát hiện khuôn mặt trong các khung hình được lấy từ camera hoặc từ video.
#~ Tệp `haarcascade_frontalface_default.xml` chứa các thông số để phát hiện khuôn mặt được huấn luyện trước, nó được cung cấp bởi OpenCV.
detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

#~ Định nghĩa hàm `layhinhanh(path)` để trích xuất khuôn mặt và ID của các khuôn mặt từ tập hình ảnh đã cho
def layhinhanh(path):
	#~ Lấy danh sách các đường dẫn của tất cả các tệp hình ảnh trong thư mục `path`.
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    #~ Khởi tạo các danh sách để lưu trữ khuôn mặt và ID của các khuôn mặt.
    faceSamples = []
    ids = []

    #~ Lặp qua tất cả các đường dẫn của các tệp hình ảnh.
    for imagePath in imagePaths:
        #~ Mở tệp ảnh tương ứng và chuyển đổi sang ảnh độ xám bằng thư viện Pillow.
        PIL_img = Image.open(imagePath).convert('L')
	
        #~ Chuyển đổi ảnh độ xám sang mảng numpy với kiểu dữ liệu uint8.
        img_numpy = np.array(PIL_img, 'uint8')

        #~ Lấy ID của khuôn mặt bằng cách tách tên tệp hình ảnh.
        id = os.path.split(imagePath)[-1].split(".")[1]
	
        #~ Phát hiện khuôn mặt trong ảnh độ xám bằng thuật toán detectMultiScale của OpenCV.
        faces = detector.detectMultiScale(img_numpy)

		#~ Lặp qua tất cả các khuôn mặt đã phát hiện được trong ảnh.
        for (x, y, w, h) in faces:
			#~ Thêm khuôn mặt đã phát hiện được vào danh sách `faceSamples`.
            faceSamples.append(img_numpy[y+2: y+h-2, x+2: x+w-2])
	    
	    	#~ Thêm ID của khuôn mặt vào danh sách `ids`.
            ids.append(id)
	    
	#~ Hàm trả về hai giá trị là `faceSamples` và `ids`
    return faceSamples,ids

faces, ids = layhinhanh(path)

#~ Sử dụng hàm train của mô hình nhận dạng khuôn mặt để huấn luyện với các khuôn mặt và ID được trích xuất từ hàm layhinhanh(path)
recognizer.train(faces, np.array(ids, dtype=np.int32))


recognizer.write('trainers/trainer{}.yml'.format(folder))
    
print('Hoàn tất training')