import os

hazy = sorted(os.listdir('datasets/reside6k/train/hazy'))
gt = sorted(os.listdir('datasets/reside6k/train/GT'))

for i in range(10):
    print(hazy[i], " | ", gt[i])