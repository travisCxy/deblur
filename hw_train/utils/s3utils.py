import configparser
import os

import random

def load_s3_config():
    config = configparser.ConfigParser()
    config.read("s3.cfg")
    endpoints = config["DEFAULT"]["S3_ENDPOINT"].split(",")
    random.shuffle(endpoints)
    os.environ["S3_ENDPOINT"] = endpoints[0]
    os.environ["S3_USE_HTTPS"] = config["DEFAULT"]["S3_USE_HTTPS"]
    os.environ["AWS_ACCESS_KEY_ID"] = config["DEFAULT"]["AWS_ACCESS_KEY_ID"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = config["DEFAULT"]["AWS_SECRET_ACCESS_KEY"]
    os.environ["AWS_LOG_LEVEL"] = config["DEFAULT"]["AWS_LOG_LEVEL"]
