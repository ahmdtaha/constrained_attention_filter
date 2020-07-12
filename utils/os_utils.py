import os
import pickle


def get_last_part(path):
    return os.path.basename(os.path.normpath(path))

# import configuration as file_const

def dataset_tuples(dataset_path):
    return dataset_path + '_tuples_class'


def get_dirs(base_path):
    return sorted([f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))])


def get_files(base_path,extension,append_base=False):
    if (append_base):
        files =[os.path.join(base_path,f) for f in os.listdir(base_path) if (f.endswith(extension) and not f.startswith('.'))]
    else:
        files = [f for f in os.listdir(base_path) if (f.endswith(extension) and not f.startswith('.'))]
    return sorted(files)


def txt_read(path):
    with open(path) as f:
        content = f.readlines()
    lines = [x.strip() for x in content]
    return lines

def txt_write(path,lines):
    out_file = open(path, "w")
    for line in lines:
        out_file.write('{}'.format(line))
        out_file.write('\n')
    out_file.close()

def pkl_write(path,data):
    pickle.dump(data, open(path, "wb"))


def hot_one_vector(y, max):
    import numpy as np
    labels_hot_vector = np.zeros((y.shape[0], max),dtype=np.int32)
    labels_hot_vector[np.arange(y.shape[0]), y] = 1
    return labels_hot_vector

def pkl_read(path):
    if(not os.path.exists(path)):
        return None

    data = pickle.load(open(path, 'rb'))
    return data

def touch_dir(path):
    if(not os.path.exists(path)):
        os.makedirs(path)

def last_tuple_idx(path):
    files =[f for f in os.listdir(path) if (f.endswith('.jpg') and not f.startswith('.'))]
    return len(files)

def get_file_name_ext(inputFilepath):
    filename_w_ext = os.path.basename(inputFilepath)
    filename, file_extension = os.path.splitext(filename_w_ext)
    return filename, file_extension

def get_latest_file(path,extension=''):
    files = get_files(path,extension=extension,append_base=True)
    return max(files, key=os.path.getctime)

def dir_empty(path):
    if os.listdir(path) == []:
        return True
    else:
        return False

def chkpt_exists(path):
    files = [f for f in os.listdir(path) if (f.find('.ckpt') > 0 and not f.startswith('.'))]
    if len(files):
        return True
    return False