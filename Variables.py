path_undamaged = ["E:/대학원/인공지능 창의연구/Bookshelf Frame Structure - DSS 2000 (1)/Book_2000_udam1/udam1/",
       "E:/대학원/인공지능 창의연구/Bookshelf Frame Structure - DSS 2000 (1)/Book_2000_udam2/udam2/",
       "E:/대학원/인공지능 창의연구/Bookshelf Frame Structure - DSS 2000 (1)/Book_2000_udam3/udam3/",
       "E:/대학원/인공지능 창의연구/Bookshelf Frame Structure - DSS 2000 (1)/Book_2000_udam4/udam4/",
       "E:/대학원/인공지능 창의연구/Bookshelf Frame Structure - DSS 2000 (1)/Book_2000_udam5/udam5/"]
path_damaged = ["E:/대학원/인공지능 창의연구/Bookshelf Frame Structure - DSS 2000 (1)/Book_2000_Dam_L1C_A/Dam1/",
               "E:/대학원/인공지능 창의연구/Bookshelf Frame Structure - DSS 2000 (1)/Book_2000_Dam_L1C_B/Dam4/",
               "E:/대학원/인공지능 창의연구/Bookshelf Frame Structure - DSS 2000 (1)/Book_2000_Dam_L3A/Dam2/",
               "E:/대학원/인공지능 창의연구/Bookshelf Frame Structure - DSS 2000 (1)/Book_2000_Dam_L13/Dam3/"]     # path where the data files saved


BATCH_SIZE = 150
rep_dim = 4
lr = 0.001
weight_decay = 1e-6
epochs = 200
num_class = 3
data_saved = True
pretrained = True
threshold_quantile = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]