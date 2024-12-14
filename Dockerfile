# Bắt đầu từ image Python 3.10
FROM python:3.10-slim

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Sao chép tệp yêu cầu và mã nguồn vào container
COPY requirements.txt /app/requirements.txt
COPY main.py /app/main.py
COPY temp.py /app/temp.py
COPY test.py /app/test.py
COPY app /app/app
COPY chatbot.html /app/chatbot.html
COPY main.spec /app/main.spec

# Cài đặt các thư viện phụ thuộc từ requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Mở cổng mà ứng dụng sẽ chạy (giả sử ứng dụng sử dụng cổng 5000)
EXPOSE 5000

# Lệnh để chạy ứng dụng (Giả sử bạn đang sử dụng Flask)
CMD ["python", "main.py"]
