import os
from pathlib import Path
import logging
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_path=os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(log_path,exist_ok=True)

LOG_FILE_PATH=os.path.join(log_path,LOG_FILE)


logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
     level=logging.INFO,
)

while True:
    project_name=input("Enter Your Project Name : ")
    if project_name!="":
        break

logging.info(f"Creating project by name :{project_name}")

list_of_files=[
    f"src/__init__.py",
    f"src/logger.py",
    f"src/exception.py"
    f"notebook/trial.ipynb",
    "requirements.txt",
    "setup.py",
    "README.md",
    'artifacts/data',
    'app.py'
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)


    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} is already exists")