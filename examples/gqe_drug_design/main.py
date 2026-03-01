from training.train_gptqe import train_gptqe

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

if __name__ == "__main__":
    train_gptqe()
