from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import app.models as sp
import app.utils as tkj
import numpy as np
import json
import os
import tensorflow as tf
from config.settings import file_word_list ,num_words_list,number_of_input,tables,weight_model
from tensorflow.keras.models import load_model
import config.settings as ts
tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
tf.config.threading.set_inter_op_parallelism_threads(os.cpu_count())



def du_doan_tong(input,model):
    with open(file_word_list, 'r') as json_file:
            word_index = json.load(json_file)
    tokenizer = Tokenizer(num_words=num_words_list, oov_token="<OOV>")
    tokenizer.word_index = word_index
    sequence = tokenizer.texts_to_sequences([input])
    padded_sequence = pad_sequences(sequence, maxlen=number_of_input)
    Ut = tf.constant(np.array(padded_sequence))
    predictions = model.predict(Ut, verbose=0)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return predicted_class


def du_doan(cau_noi,models):
    predict=[]
    temp=[]
    for model,name_mode in zip(models, tables):
        du_doan_temp = du_doan_tong(cau_noi,model)
        #print(tkj.search_content_lable(table_name=name_mode, id=du_doan_temp))
        #print(name_mode+"_"+tkj.search_content_lable(table_name=name_mode, id=du_doan_temp)[0])
        is_true_model = sp.load_model_true_false(name_mode+"_"+tkj.search_content_lable(table_name=name_mode, id=du_doan_temp)[0])
        temp.append(du_doan_temp)
        if(du_doan_tong(cau_noi,is_true_model)):
            #print (du_doan_temp)
            predict.append(du_doan_temp)
        else: 
             #print(0)
             predict.append(0)
        del is_true_model
    return predict,temp
#print(du_doan("what is the speed of comparator?",models))

def creater_report(models,name_report):
    question=[]
    with open('app\\data\\json\\data_test.json', 'r', encoding='utf-8') as file:
        datas = json.load(file)
        for data in datas:
            question.append(data["question"])  

    #print(question)
    

    matrix_du_doan=[]

    for row in question:
        du_doan_tempp,trash= du_doan(row,models)
        matrix_du_doan.append(du_doan_tempp)
        print(du_doan_tempp)
            
    matrix_du_doan_T=sp.transpose_matrix(matrix_du_doan) 
    data_file="app\\data\\json\\data_du_doan.json"
    matrix_du_doan_T_list=[[int(x) for x in y]for y in matrix_du_doan_T]
    print(matrix_du_doan_T)
    with open(data_file, 'w', encoding='utf-8') as output_file:
        json.dump(matrix_du_doan_T_list, output_file, indent=4, ensure_ascii=False)


    matrix_true_label=[]
    with open('app\\data\\json\\data_test.json', 'r', encoding='utf-8') as file:
        datas = json.load(file)
        for data in datas:
            temp=[]
            for tabel in tables:
                temp.append(data[tabel])  
            matrix_true_label.append(temp)
    matrix_true_label_T=sp.transpose_matrix(matrix_true_label)
    for row,row_true,tabel in zip(matrix_du_doan_T,matrix_true_label_T,tables):
        tkj.tao_report(tabel, row,row_true,name_report)



   
def repair_train(question_false,false_answer,false_answer_no_include_true_false,true_answer): 
    c=0                                  
    for dd,nb,t, table in zip(false_answer,false_answer_no_include_true_false,true_answer,tables):
        
        ddd=nb==t
        dnb=dd==(t>0)
        dt=(t>0)
        if(dnb and (not ddd) and dt):
            print ("update chill")
            label=tkj.search_content_lable(table_name=table, id=nb)[0]
            new_model_true_false = sp.load_model_true_false(table+"_"+label) 
            sp.update_weights_on_incorrect_prediction( new_model_true_false, question_false, 1)
            for t in tkj.creater_random_3_question(label):
                sp.update_weights_on_incorrect_prediction( new_model_true_false, question_false, 0)
            #print (4)
            new_model_true_false.save_weights(sp.replace_space_with_underscore(weight_model.format(table+"_"+label)))
            del model,new_model_true_false
        if((not dnb) and ddd and dt):
            model = sp.load_model(table)
            new_model_true_false = sp.load_model_true_false(table+"_"+label) 
            # print(question_false)
            print("update parent model{} and chill".format(c)) 
            
            #print(t)
            sp.update_weights_on_incorrect_prediction( new_model_true_false, question_false, 0)
            sp.update_weights_on_incorrect_prediction( model, question_false, t)
            model.save_weights(sp.replace_space_with_underscore(weight_model.format(table)))
            new_model_true_false.save_weights(sp.replace_space_with_underscore(weight_model.format(table+"_"+label)))
            del model,new_model_true_false
        if((not dnb)and (not ddd) and dt):
            print("update parent model{}".format(c))
            model = sp.load_model(table)
            sp.update_weights_on_incorrect_prediction( model, question_false, t)
            model.save_weights(sp.replace_space_with_underscore(weight_model.format(table)))
            del model,new_model_true_false
        c=c+1

class ModelManager:
    def __init__(self):
        self.question_last=None
        self.model_du_doan = None
        self.model_du_doan_khong_gom_true_false = None

    def final_du_doan(self, question, models, true_answer=None):
        temp1=[]
        temp2=[]
        print("question: {}".format(question))
        if true_answer is not None:
            print("du doan    :{}".format(self.model_du_doan))
            print("true_answer:{}".format(true_answer))
            #print(self.model_du_doan_khong_gom_true_false)
            repair_train(self.question_last,self.model_du_doan,self.model_du_doan_khong_gom_true_false,true_answer)
        
        self.question_last=question
        self.model_du_doan, self.model_du_doan_khong_gom_true_false = du_doan(question, models)
        if(ts.co_gom_true_false):
            answer = sp.replace_positive(self.model_du_doan)
        else:
            print(0)
            answer = sp.replace_positive(self.model_du_doan_khong_gom_true_false) #
        final_answer = []

        # Tìm kiếm trong SQL Server
        con = sp.search_with_conditions_sqlserver(self.model_du_doan)
        if con:
            temp1.append("{}".format(self.model_du_doan))
            temp2.append("{}".format(translate_label(self.model_du_doan)))
            final_answer.append(con[0])
        for row in answer:
            con = sp.search_with_conditions_sqlserver(row)
            if con:
                temp1.append("{}".format(row))
                temp2.append("{}".format(translate_label(row)))
                final_answer.append(con[0])
        file_path="data_user.json"
        new_data={
            "cau_hoi":question,
            "nhan_so":temp1,
            "nhan_chu":temp2,
            "cau_tra_loi":final_answer
        }
        WriteUserQuestion(file_path, new_data)
        return final_answer

import json

def WriteUserQuestion(file_path, new_data):
    try:
        # Mở file và đọc nội dung hiện tại
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Kiểm tra nếu file chứa một danh sách
        if isinstance(data, list):
            data.append(new_data)  # Thêm dữ liệu mới vào danh sách
        else:
            raise ValueError("Dữ liệu trong file JSON không phải là danh sách.")
    
    except FileNotFoundError:
        # Nếu file không tồn tại, tạo một danh sách mới
        data = [new_data]
    
    # Ghi lại dữ liệu vào file
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)



def translate_label(rows):
    trans=[]
    for row,table in zip(rows,tables):
        temp=tkj.search_content_lable(table,row)[0]
        if(temp is not None):
            trans.append(temp)
            continue
        else:
            trans.append("")
    return trans


def check_model_loading(model_list):
    
    failed_models = []

    for model_name in model_list:
        try:
            #print(f"Đang kiểm tra model: {model_name}")
            # Tạo đường dẫn đến trọng số
            
            # Giả sử trọng số lưu trong thư mục "weights"
            
            # Khởi tạo mô hình (cần thay thế bằng hàm khởi tạo đúng của bạn)
            model = sp.load_model_true_false(model_name)  # Thay thế bằng hàm của bạn
            
            # Cố gắng tải trọng số
            
           
        except Exception as e:
            # Nếu lỗi, ghi nhận model vào danh sách lỗi
            print(f"Lỗi khi tải model {model_name}")
            failed_models.append((model_name, str(e)))

    return failed_models

# def ghi_cau_tra_loi(i,model_manager):

#     with open('model\data\cau_tra_loi_co_test_khong_true_false.txt','a', encoding='utf-8') as file:
#         print(i)
#         file.write("question:{}\n".format(i))
#         answer = model_manager.final_du_doan(i, models)
#         file.write("mess: {}".format(answer))
#         file.write("\n")
#         file.write("\n")

# #Tải danh sách các mô hình
# model_manager = ModelManager()
# models = []
# for name_mode in tables:  # Đảm bpyảo `tables` đã được định nghĩam
#     new_model = sp.load_model(name_mode)
#     models.append(new_model)

# # Thực hiện dự đoán
# answer = model_manager.final_du_doan("what is comparator ?", models)
# print(answer)
# answer = model_manager.final_du_doan("when is comparator ?", models)
# print(answer)
# answer = model_manager.final_du_doan("how we use comparator ?", models)
# print(answer)