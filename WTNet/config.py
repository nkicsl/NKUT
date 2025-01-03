class config:
    queue_length = 300
    samples_per_volume = 30
    patch_size = 64, 64, 64
    epoch = 5
    epochs_per_val = 1
    input_channel = 1
    num_classes = 4
    batch_size = 2
    learning_rate = 0.001
    # crop_or_pad_size = 512, 512, 32
    input_train_image_dir = 'D:/TOOTH/Datasets/new/patches/256_256_16/Image'
    input_train_label_dir = 'D:/TOOTH/Datasets/new/patches/256_256_16/label'
    input_val_image_dir = 'C:/Users/zhouzhenhuan/Desktop/DATA/Val/Image'
    input_val_label_dir = 'C:/Users/zhouzhenhuan/Desktop/DATA/Val/Label'
    # input_test_image_dir = ''
    # input_test_label_dir = ''
    output_logs_dir = 'E:/PycharmProjects/NKUT_Tooth/logs'
    devices = [0, 1]
    step_size = 10
    gamma = 0.8
    latest_output_dir = 'E:/PycharmProjects/NKUT_Tooth/result/latest_output_dir/latest_result.pt'
    latest_checkpoint_file = 'E:/PycharmProjects/NKUT_Tooth/result/latest_checkpoint_dir/latest_checkpoint.pt'
    best_model_path = 'E:/PycharmProjects/NKUT_Tooth/result/best_model/best_model.pt'
    epochs_per_checkpoint = 10
