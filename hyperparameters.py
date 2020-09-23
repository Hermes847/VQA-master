#------------model-------------
num_layers = 6
d_model = 768
dff = 2048
num_heads = 8
dropout_rate = 0.1
max_qst_len = 20
max_ans_len = 5
#------------training-------------
steps = 1000000
steps_per_save = 1000,
steps_per_chunk = 10000,
steps_per_report = 10
batch_size = 16
num_chunks = 16
