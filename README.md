# Domain_Classify
- File New_Domain.txt dùng để ghi những domain mới cần được phân loại.
- File Project_Domain.json dùng để lưu trữ các domain đã được phân loại tên app.
- Chạy file train_model.py để học máy:
    - Các trọng số học máy được lưu trên file model.sav
    - Các tên app sẽ được mapping với các số thứ tự ánh xạ đến nó được lưu trong file label_encoder.sav
    - Các dữ liệu sẽ được biến đổi thành vector và được lưu trữ trong vectorizer.sav

- Chạy file Predict_App.py để phân tích domain mới được ghi trong file New_Domain.txt:
    - Mỗi domain được ghi trên một dòng.
    - Nếu domain chưa từng tồn tại trong DB sẽ được hỏi có thêm dự đoán đó vào lại DB không.
