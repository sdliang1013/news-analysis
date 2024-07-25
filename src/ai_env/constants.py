import platform

DATA_DIR = "D:/data"
if platform.system() != "Windows":
    DATA_DIR = "/home/apps/data"
