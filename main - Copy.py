import cv2

#~ Khởi tạo một bộ phân loại đối tượng Cascade để phát hiện khuôn mặt trong hình ảnh.
#~ Bộ phân loại Cascade được dùng để phát hiện đối tượng trong hình ảnh bằn cách tìm kiếm các đặc trưng được xác định trước.
#~ File `haarcascade_frontalface_default.xml` chứa thông tin về các đặc trưng cần thiết để phát hiện khuôn mặt.
face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

#~ Tạo menu chọn video của mỗi người để lấy mẫu
#~ Người dùng sẽ được yêu cầu nhập số tương ứng với người mà họ muốn nhận diện khuôn mặt.
#~ Khi nhập số hợp lệ, giá trị sẽ được gán cho biến `path` để chỉ đường dẫn đến thư mục chứa ảnh của người đó.
#~ Đồng thời gán giá trị cho biến `id` để phân biệt 
while True:
    print('Menu:')
    print('[1]  -  Thiện')
    print('[2]  -  Khoa')
    print('[3]  -  Tú')
    print('[4]  -  Tâm')
    print('[5]  -  Linh')
    chon = input('Chọn video (1, 2, 3, 4, 5): ')
    if chon == '1':
        path = 'thien'
        id = chon
        break
    elif chon == '2':
        path = 'Khoa'
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

#~ Sử dụng thư viện OpenCV để mở file video từ đường dẫn lấy từ biến `path` và trả về một đối tượng.
#~ `cv2.VideoCapture()` là hàm để khởi tạo đối tượng `VideoCapture`.
#~ Trong đoạn code này đường dẫn lấy từ biến `path` và kết hợp với phần mở rộng `.mp4` để tạo thành file video cần mở. 
camera = cv2.VideoCapture(path+'.mp4') #~ (nếu đối số là 0 thì sử dụng camera của máy)

#~ Biến `id` được gán cho biến `face_id` 
#~ Biến `id` đã được xác định từ trước và sử dụng để định danh người trong ảnh hoặc video.
#~ Biến `face_id` này sẽ được sử dụng để dịnh danh khuôn mặt trong quá trình train
face_id = id

#~ Tạo biến đếm số lượng hình ảnh 
count = 0

#~ Sử dụng vòng lặp vô hạn để liên tục đọc khung hình từ camera
while True:

    #~ Biến `ret` trả về kết quả đọc khung hình có thành công hay không
    #~ BIến `img` chứa khung hình đọc được và lật theo chiều ngang (giống như gương)
    ret, img = camera.read()
    if not ret or img is None:
        print(" Không đọc được frame từ video. Kiểm tra file video có tồn tại không.")
        break
    img = cv2.flip(img, 1)
    
    #~ Hình ảnh được chuyển đổi thành ảnh độ xám để tiện cho việc xử lý. (giảm bit -> tăng tốc độ xử lý)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #~ Phát hiện khuôn mặt trong ảnh bằng hàm `face_detector.detectMultiScale` và lưu các khuôn mặt được phát hiện vào biến faces.
    faces = face_detector.detectMultiScale(img, 1.3, 5) 

    #~ Với mỗi khuôn mặt được phát hiện trong biến faces,lưu vào thư mục của người đó với tên file là mã số của người đó và số thứ tự của khuôn mặt (biến count)
    for (x, y, w, h) in faces:

        #~ Vẽ hình chữ nhật bằng tọa độ Descartes, màu xanh lá quanh khuôn mặt (làm việc trên ảnh màu)
        face_img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)  #~ (tọa độ trên bên trái), (tọa độ dưới bên phải(x + chiều rộng), (y + chiều cao), (màu), (độ dày))
        
        #~ Tăng biến đếm thêm 1
        count += 1
    
        #~ Để lưu khuôn mặt, trước hết ta sẽ cắt khung hình tương ứng với khuôn mặt đó
        face_img = gray[y: y+h, x: x+w]

        #~ Sau đó, lưu hình ảnh cắt được vào thư mục của người đó với tên file là "{tên người}/{tên người}.{mã số khuôn mặt}.{số thứ tự khuôn mặt}.jpg"
        cv2.imwrite('hinh_anh/{}/{}.{}.{}.jpg'.format(path, path, face_id, count), face_img)

        cv2.imshow('Hình', img)  
          
    #~ hiển thị khung hình và sử dụng cv2.waitKey để đợi 50ms cho đến khi người dùng bấm phím ESC để thoát
    k = cv2.waitKey(50) & 0xff #~ Bấm ESC để thoát
    if k == 27:
        break
    # elif count == 100:
    #     break
print('Hoàn tất việc lấy mẫu!')
camera.release()
cv2.destroyAllWindows()


