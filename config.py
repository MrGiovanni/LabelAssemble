
assemble_datasets = ['covidx', 'chestxray14']
class_interests = ['CovidPositive']
target_source = [0, ]
class_mapping = {
    'Atelectasis': 0, 
    'Cardiomegaly': 1, 
    'Effusion': 2, 
    'Infiltration': 3, 
    'Mass': 4, 
    'Nodule': 5,                   
    'Pneumonia': 6, 
    'Pneumothorax': 7, 
    'Consolidation': 8, 
    'Edema': 9,                   
    'Emphysema': 10, 
    'Fibrosis': 11, 
    'Pleural_Thickening': 12, 
    'Hernia': 13,
    'CovidPositive': 14,
    'health': -1,
}

COVIDXConfig = dict(
    train_img_path = '../../data/COVIDX/train',
    val_img_path = None,
    test_img_path = '../../data/COVIDX/test',
    train_file_path = '../../data/COVIDX/train.txt',
    val_file_path = None,
    test_file_path = '../../data/COVIDX/test.txt',
    class_num = 1,
    class_filter = ['CovidPositive'],
    using_num = 1000
)
        
ChestXray14Config = dict(
    train_img_path = '../../data/chestXray14/images',
    val_img_path = '../../data/chestXray14/images',
    test_img_path = '../../data/chestXray14/images',
    train_file_path = '../../data/chestXray14/train_official.txt',
    val_file_path = '../../data/chestXray14/val_official.txt',
    test_file_path = '../../data/chestXray14/test_official.txt',
    class_num = 1,
    class_filter = ['Pneumonia', ],
    using_num = 1000
)

CustomConfig = dict(
    train_img_path = None,
    val_img_path = None,
    test_img_path = None,
    train_file_path = None,
    val_file_path = None,
    test_file_path = None,
    class_num = None,
    class_filter = None,
    using_num = None
)