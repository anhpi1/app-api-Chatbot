import requests
import json
# URL của API
url = 'http://127.0.0.1:5000/du_doan_co_lich_su'



    
def questionn(i,L):
    # Chuỗi ký tự muốn gửi
    data = {
        "message": "{}".format(i),
        "true_label" : L
        
        }
    with open('temp\\data_test.txt','a', encoding='utf-8') as file:
        # Gửi yêu cầu POST với chuỗi ký tự
        response = requests.post(url, json=data)
        file.write("question:{}\n".format(i))
        # In kết quả trả về từ API
        file.write(str(response.json()))
        file.write("\n")
        file.write("\n")

temp=None

with open('temp\\data_test.json', 'r', encoding='utf-8') as data_file:
    x =json.load(data_file)
    for row in x:
        
        questionn(row["question"],temp)
        temp=[row["intent"],row["parameter"],row["structure"],row["operation"],row["components"],row["applications"],row["comparison"],row["techniques"],row["simulation"]]

        
