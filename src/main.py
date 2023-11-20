from common import *
from build_dl import main as build_dl_main
from train import main as train_main

if BUILD_DL:
    build_dl_main()
else:
    train_main()
