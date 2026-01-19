#This file gives a complete logging history in logs folder log file

import os
import sys
import logging

logging_str="[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_dir='logs'
log_filepath=os.path.join(log_dir,"running_logs.log")
os.makedirs(log_dir,exist_ok=True)   #Make a log folder and inside that store all logs

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(log_filepath),  #Get the log file
        logging.StreamHandler(sys.stdout)   #see in terminal
    ]
)

logger=logging.getLogger('LiverTumorSegmentationLogger')