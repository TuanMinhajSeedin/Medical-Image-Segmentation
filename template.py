import os
from pathlib import Path   #This library for accept the '/' 
import logging

#logging String
logging.basicConfig(level=logging.INFO,format='[%(asctime)s]: %(message)s')

project_name='LiverTumorSegmentation'

list_of_files=[
    "artifacts/",
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/data/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/models/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    f"src/{project_name}/training/__init__.py",
    "train.py",
    "eval.py",
    "infer.py",
    "run_experiments.py",
    "configs/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb",
    "templates/index.html"
]

for filepath in list_of_files:
    filepath=Path(filepath)
    filedir,filename=os.path.split(filepath)

    if filedir !="":
        os.makedirs(filedir,exist_ok=True)    #This create folders if there is not exists
        logging.info(f"Creating  directory: {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):    #This create file if there is not exists in corresponding folder or if python file has has no size. it is not replacing if there are code in py code
        with open(filepath,"w") as f:
            pass
            logging.info(f"creating empty file: {filepath}")
    else:
        logging.info(f"{filename} is already exists")