import os
from shutil import copy, copytree
from subprocess import call
import glob


from os.path import expanduser
home = expanduser("~")


def setup_api_environment():
    if not os.path.exists(f'{home}/.kaggle/kaggle.json'):
        if not os.path.exists(f'{home}/.kaggle'):
            os.mkdir(f'{home}/.kaggle')
        copy(os.path.dirname(os.path.realpath(__file__)) + '/../configs/kaggle.json', f'{home}/.kaggle/kaggle.json')


def upload_path_content_to_dataset(path, dataset_name='steamedsheep/leaf-experiment-record',
                                   message='new experiment result'):
    if os.path.exists('upload'):
        os.rmdir('upload')
    os.mkdir('upload')
    call(f'kaggle datasets download -p upload {dataset_name}', shell=True)
    # decompress
    zf = glob.glob(f'upload/*.zip')[0]
    print(zf)
    call(f'unzip {zf} -d upload', shell=True)
    copytree(path, f'upload/{path.split("/")[-1]}')
    call(f'kaggle datasets metadata -p upload {dataset_name}', shell=True)
    call(f'kaggle datasets version -p upload -m "{message}"  --dir-mode zip', shell=True)


if __name__ == '__main__':
    setup_api_environment()