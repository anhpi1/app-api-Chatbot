from flask import Flask
from app.routes import api_routes  # Đảm bảo api_routes đã được import đúng cách

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Đăng ký Blueprint cho các route
app.register_blueprint(api_routes)

# In thông báo khi server đã sẵn sàng
if __name__ == '__main__':
    print("Server is ready to accept requests...")
    # Chạy ứng dụng Flask, chỉ gọi app.run() 1 lần
    app.run(host='0.0.0.0', port=5000, debug=True)
