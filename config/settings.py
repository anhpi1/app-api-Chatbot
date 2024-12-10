number_of_input = 32
number_of_model = 1
number_of_copies_model = 1
weight_model = 'app\\data\\weights_model\\model_{}.weights.h5'

report_train = 'app\\data\\model_{}_{}.log'
file_word_list = 'app\\data\\json\\word_list.json'

num_words_list = 1000
topics = ['question_type','question_intent','concept1','concept2','structure','operation','performance_metric','design_techniques','applications','components','testing_simulations_tools']
tables=["intent","parameter","structure","operation","components","applications","comparison","techniques","simulation"]


server = 'DESKTOP-1MU0IU3\SQLEXPRESS'
database = 'comparator'
username = ''
password = ''
command_connect_sever = 'DRIVER={{SQL Server}};SERVER={};DATABASE={};UID={};PWD={}'


command_sever_get_input = 'SELECT content FROM dbo.question;'
command_sever_get_output = 'SELECT {} FROM dbo.question;'
command_sever_get_output_train = 'SELECT {}_id FROM dbo.question;'
co_train_du_lieu_test=1
so_mau_train=4
so_mau_false_x_20 = 3
so_lan_loss_k_thay_doi=20
co_gom_true_false=1