exp_no = 1
gru_size = 256
grad_clip = 10
embed_dim = 128
n_layers = 1
batch_size = 32
drop_out = 0.1
init_learning_rate = 0.0005

data_size = 'all_data'
data_source = 'stanford'
data = '/share/data/lang/users/xiaoan/data/lambada/' + data_size + '/' + data_source + '.data.transformer'

epoch = 1
cnt = 10
directory = '/share/data/lang/users/xiaoan/exp/lstm-lambada/'

fp = open(directory + 'bash/' + 'series_' + str(exp_no) + '.sh', "w")
for i in range(cnt):
    fp.write('#!/bin/sh \n sbatch -p gpu -J lstm_' + str(exp_no) + ' -d singleton ' +
             directory + 'bash/' + str(exp_no) + '/' + 't_' + str(i)+'.sh\n')
fp.close()

for i in range(cnt):
    fp = open(directory + 'bash/' + str(exp_no) + '/' + 't_' + str(i) + '.sh', 'w')
    if i == 0:
        fp.write('#!/bin/sh\n' + 'python ' + directory + 'train.py -print_every 20000 ' +
                 ' -gru_size ' + str(gru_size) + ' -grad_clip ' +  str(grad_clip) + 
                 ' -embed_dim ' + str(embed_dim) + ' -n_layers ' + str(n_layers) +
                 ' -batch_size ' + str(batch_size) + ' -drop_out ' + str(drop_out) + 
                 ' -init_learning_rate ' + str(init_learning_rate) + ' -epoch ' + str(epoch) +
                 ' -data  ' + data + ' -use_board True ' +
                 ' -save_mode series ' + 
                 ' -save_model ' + directory + 'checkpoint/' + str(exp_no) + '/' + str(i) +
                 ' -log_file ' + directory + 'logs/' + str(exp_no) + '/' + str(i) + '.log \n')
    else:
    	fp.write('#!/bin/sh\n' + 'python ' + directory + 'train.py -print_every 20000 ' +
                 ' -gru_size ' + str(gru_size) + ' -grad_clip ' +  str(grad_clip) + 
                 ' -embed_dim ' + str(embed_dim) + ' -n_layers ' + str(n_layers) +
                 ' -batch_size ' + str(batch_size) + ' -drop_out ' + str(drop_out) + 
                 ' -init_learning_rate ' + str(init_learning_rate) + ' -epoch ' + str(epoch) +
                 ' -data  ' + data + ' -use_board True ' +
                 ' -save_mode series ' + 
                 ' -restore_model ' + directory + 'checkpoint/' + str(exp_no) + '/' + str(i-1) +
                 ' -save_model ' + directory + 'checkpoint/' + str(exp_no) + '/' + str(i) +
                 ' -log_file ' + directory + 'logs/' + str(exp_no) + '/' + str(i) + '.log \n')
    fp.close()
