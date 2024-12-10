import os

def print_directory_structure(path):
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)  # Đếm số cấp thư mục
        indent = ' ' * 4 * level  # Indentation (lùi dòng)
        print(f"{indent}[DIR] {os.path.basename(root)}")  # In thư mục
        subindent = ' ' * 4 * (level + 1)  # Lùi 1 mức nữa cho các tệp tin
        for file in files:
            print(f"{subindent}[FILE] {file}")  # In tệp tin

# Gọi hàm với đường dẫn thư mục
print_directory_structure('C:\\Users\\k\\Documents\\GitHub\\app api Chatbot')
