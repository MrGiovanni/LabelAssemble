device = 'cuda'

COVIDXConfig = dict(
    train_img_path = '../../data/COVIDX/train',
    val_img_path = None,
    test_img_path = '../../data/COVIDX/test',
    train_file_path = '../../data/COVIDX/train.txt',
    val_file_path = None,
    test_file_path = '../../data/COVIDX/test.txt',
    class_num = 1,
    class_filter = [],
    using_num = 30000
)
        
ChestXray14Config = dict(
    train_img_path = '../../data/chestXray14/images',
    val_img_path = '../../data/chestXray14/images',
    test_img_path = '../../data/chestXray14/images',
    train_file_path = '../../data/chestXray14/train_official.txt',
    val_file_path = '../../data/chestXray14/val_official.txt',
    test_file_path = '../../data/chestXray14/test_official.txt',
    class_num = 14,
    class_filter = [],
    using_num = 110000
)

CustomConfig = dict(
    train_img_path = None,
    val_img_path = None,
    test_img_path = None,
    train_file_path = None,
    val_file_path = None,
    test_file_path = None,
    class_num = 14,
    class_filter = [],
    using_num = 110000
)