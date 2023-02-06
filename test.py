import torch
import numpy
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import os
# import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import warnings
import eval
import bleu
import utils
import string
import copy
import argparse
from torchinfo import summary
from models import *

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import os
# import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import warnings
import eval
import bleu
import utils
import string
import copy
import argparse

from models import *

img_dir = './dataset/Flickr8k_Dataset/'
ann_dir = './dataset/Flickr8k_text/Flickr8k.token.txt'
train_dir = './dataset/Flickr8k_text/Flickr_8k.trainImages.txt'
val_dir = './dataset/Flickr8k_text/Flickr_8k.devImages.txt'
test_dir = './dataset/Flickr8k_text/Flickr_8k.testImages.txt'
vocab_file = './vocab.txt'

model = torch.load("./model_saves/resnet18/model_final_lyr_3_hds_1_1670520647.7878547.pt")

def predict(model, device, image_name):
    vocab = []
    with open(vocab_file, "r") as vocab_f:
        for line in vocab_f:
            vocab.append(line.strip())
    image_path = os.path.join(img_dir, image_name)
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224,224)),
#                 transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    hypotheses = eval.get_output_sentence(model, device, image, vocab)

    for i in range(len(hypotheses)):
        hypotheses[i] = [vocab[token - 1] for token in hypotheses[i]]
        hypotheses[i] = " ".join(hypotheses[i])

    return hypotheses


device = 'cuda' if torch.cuda.is_available() else 'cpu'


images = []
with open(test_dir, "r") as test:
    i = 0
    for line in test:
        images.append(line.strip())
        i+=1
        if i == 30:
            break

for indice in range(i):

    a = predict(model, device, images[indice])
    
    img = plt.imread(img_dir + images[indice])
    plt.title(a[0])
    plt.imshow(img)

    plt.show()
    