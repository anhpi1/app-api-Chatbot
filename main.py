from flask import Flask
from app.routes import api_routes

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Đăng ký Blueprint cho các route
app.register_blueprint(api_routes)

# In thông báo khi server đã sẵn sàng
if __name__ == '__main__':
    print("Server is ready to accept requests...")
    # Sử dụng use_reloader=False để không đóng màn hình sau khi chạy
    app.run(debug=True, use_reloader=False)

