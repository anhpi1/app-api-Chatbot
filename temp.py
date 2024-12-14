import requests
import json

# URL của API
url = 'http://127.0.0.1:5000/du_doan_co_lich_su'

def questionn(i):
    # Chuỗi ký tự muốn gửi
    data = {
        "message": "{}".format(i),
        "true_label" : []  # Giả sử true_label là một danh sách trống
    }
    # Gửi yêu cầu POST với chuỗi ký tự và lấy phản hồi dưới dạng JSON
    response = requests.post(url, json=data).json()

    # In ra thông điệp trong phản hồi
    print(response.get("message", "No message in response"))

# Gọi hàm với câu hỏi
questionn("what is comparator")
