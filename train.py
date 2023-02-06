import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import os
from torch.utils.tensorboard import SummaryWriter
# import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import warnings
import eval
import utils
import time
import copy
import argparse

from models import *


img_dir = './dataset/Flickr8k_Dataset/'
ann_dir = './dataset/Flickr8k_text/Flickr8k.token.txt'
train_dir = './dataset/Flickr8k_text/Flickr_8k.trainImages.txt'
val_dir = './dataset/Flickr8k_text/Flickr_8k.devImages.txt'
test_dir = './dataset/Flickr8k_text/Flickr_8k.testImages.txt'

vocab_file = './vocab.txt'
_t = time.time()

SEED = 123
torch.manual_seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Flickr8kDataset(Dataset):
    """Flickr8k dataset."""
    
    def __init__(self, img_dir, split_dir, ann_dir, vocab_file, transform=None):
        """
        Args:
            img_dir (string): Directory with all the images.
            ann_dir (string): Directory with all the tokens
            split_dir (string): Directory with all the file names which belong to a certain split(train/dev/test)
            vocab_file (string): File which has the entire vocabulary of the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.split_dir = split_dir
        self.SOS = self.EOS = None
        self.word_2_token = None
        self.vocab_size = None
        self.image_file_names, self.captions, self.tokenized_captions= self.tokenizer(self.split_dir, self.ann_dir)
        
        if(transform == None):
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),

                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def tokenizer(self, split_dir, ann_dir):
        image_file_names = []
        captions = []
        tokenized_captions = []
        
        with open(split_dir, "r") as split_f:
            sub_lines = split_f.readlines()
        
        with open(ann_dir, "r") as ann_f:
            for line in ann_f:
                if line.split("#")[0] + "\n" in sub_lines:
                    caption = utils.clean_description(line.replace("-", " ").split()[1:])
                    image_file_names.append(line.split()[0])
                    captions.append(caption)


        vocab = []


        with open(vocab_file, "r") as vocab_f:
            for line in vocab_f:
                vocab.append(line.strip())
        
        self.vocab_size = len(vocab) + 2 #The +2 is to accomodate for the SOS and EOS
        self.SOS = 0
        self.EOS = self.vocab_size - 1
        
        
        self.word_2_token = dict(zip(vocab, list(range(1, self.vocab_size - 1))))

        for caption in captions:
            temp = []
            for word in caption:
                temp.append(self.word_2_token[word])
            temp.insert(0, self.SOS)
            temp.append(self.EOS)
            tokenized_captions.append(temp)
            
        assert(len(image_file_names) == len(captions))
            
        return image_file_names, captions, tokenized_captions
        

    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name, cap_tok, caption = self.image_file_names[idx], self.tokenized_captions[idx], self.captions[idx]
        img_name, instance = img_name.split('#')
        img_name = os.path.join(self.img_dir,
                                img_name)
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        cap_tok = torch.tensor(cap_tok)
        sample = {'image': image, 'cap_tok': cap_tok, 'caption': caption}

        return sample




def collater(batch):
    '''This functions pads the cpations and makes them equal length
    '''
    
    cap_lens = torch.tensor([len(item['cap_tok']) for item in batch]) #Includes SOS and EOS as part of the length
    caption_list = [item['cap_tok'] for item in batch]
#     padded_captions = pad_sequence(caption_list, padding_value=9631)
    images = torch.stack([item['image'] for item in batch])

    return images, caption_list, cap_lens


def display_sample(sample):
    image = sample['image']
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255]
    )
    image = inv_normalize(image)
    caption = ' '.join(sample['caption'])
    cap_tok = sample['cap_tok']
    plt.figure()
    plt.imshow(image.permute(1,2,0))
    print("Caption: ", caption)
    print("Tokenized Caption: ", cap_tok)
    plt.show()

def predict(model, device, image_name):
    vocab = []
    with open(vocab_file, "r") as vocab_f:
        for line in vocab_f:
            vocab.append(line.strip())
    image_path = os.path.join(img_dir, image_name)
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224,224)),
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


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)






def train_for_epoch(model, dataloader, optimizer, device, n_iter, args):
    '''Train an EncoderDecoder for an epoch

    Returns
    -------
    avg_loss : float
        The total loss divided by the total numer of sequence
    '''
    
    criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')
    total_loss = 0 
    total_num = 0
    for data in tqdm(dataloader):
        images, captions, cap_lens = data
        captions = pad_sequence(captions, padding_value=model.target_eos) #(seq_len, batch_size)
        images, captions, cap_lens = images.to(device), captions.to(device), cap_lens.to(device)
        optimizer.zero_grad()

        logits = model(images, captions).permute(1, 0, 2)

        captions = captions[1:]
        mask = model.get_target_padding_mask(captions)
        captions = captions.masked_fill(mask,-1)
        loss = criterion(torch.flatten(logits, 0, 1), torch.flatten(captions))


        total_loss += loss.item()
        total_num += len(cap_lens)

        loss.backward()
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)
        optimizer.step()
        

        n_iter += 1
        torch.cuda.empty_cache()
    return total_loss/total_num, n_iter


parser = argparse.ArgumentParser(description='Training Script for Encoder+Transformer decoder')
parser.add_argument('--lr', type=float, help='learning rate', default=0.0001)
parser.add_argument('--batch-size', type=int, help='batch size', default=64)
parser.add_argument('--batch-size-val', type=int, help='batch size validation', default=64)
parser.add_argument('--encoder-type', choices=['resnet18', 'resnet50', 'resnet101', 'efficientnet_v2_s'], default='resnet18',
                    help='Network to use in the encoder (default: resnet18)')
parser.add_argument('--fine-tune', type=int, choices=[0,1], default=0)
parser.add_argument('--decoder-type', choices=['transformer'], default='transformer')
parser.add_argument('--beam-width', type=int, default=4)
parser.add_argument('--num-epochs', type=int, default=20)
parser.add_argument('--experiment-name', type=str, default="autobestmodel")
parser.add_argument('--num-tf-layers', help="Number of transformer layers", type=int, default=3)
parser.add_argument('--num-heads', help="Number of heads", type=int, default=2)
parser.add_argument('--beta1', help="Beta1 for Adam", type=float, default=0.9)
parser.add_argument('--beta2', help="Beta2 for Adam", type=float, default=0.999)
parser.add_argument('--dropout-trans', help="Dropout_Trans", type=float, default=0.1)
parser.add_argument('--use-checkpoint', help="Use checkpoint or start from beginning", type=int, default=0)


args = parser.parse_args()

encoder_type = args.encoder_type
decoder_type = args.decoder_type #transformer
warmup_steps = 4000
n_iter = 1

if encoder_type == 'resnet18':
    CNN_channels = 512 # 2048 for resnet101
elif encoder_type == 'efficientnet_v2_s':
    CNN_channels = 1280
else:
    CNN_channels = 2048

max_epochs = args.num_epochs
# max_epochs = 20
beam_width = args.beam_width

print("Epochs are read correctly: ", max_epochs)
print("Encoder type is read correctly: ", encoder_type)
print("Number of CNN channels being used: ", CNN_channels)
print("Fine tune setting is set to: ", bool(args.fine_tune))


word_embedding_size = 512
attention_dim = 512
model_save_path = './models_saves2/'+encoder_type+'/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lamda = 1.

learning_rate = 0.00004

decoder_hidden_size = CNN_channels
dropout = args.dropout_trans

batch_size = args.batch_size
batch_size_val = args.batch_size_val
grad_clip = 5.
transformer_layers = args.num_tf_layers
heads = args.num_heads
beta1 = args.beta1
beta2 = args.beta2

mode = 'train'

if not os.path.isdir(model_save_path):
    os.mkdir(model_save_path)

train_data = Flickr8kDataset(img_dir, train_dir, ann_dir, vocab_file)
train_data_to_eval = eval.TestDataset(img_dir, train_dir, ann_dir, vocab_file)
val_data = eval.TestDataset(img_dir, val_dir, ann_dir, vocab_file)
test_data = eval.TestDataset(img_dir, test_dir, ann_dir, vocab_file)



train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collater)
train_dataloader_to_eval = DataLoader(train_data_to_eval, batch_size=batch_size_val, shuffle=False, collate_fn=eval.collater)
val_dataloader = DataLoader(val_data, batch_size=batch_size_val, shuffle=False, collate_fn=eval.collater)
test_dataloader = DataLoader(test_data, batch_size=batch_size_val, shuffle=False, collate_fn=eval.collater)

encoder_class = Encoder

decoder_class = TransformerDecoder

model = EncoderDecoder(encoder_class, decoder_class, train_data.vocab_size, target_sos=train_data.SOS, 
                      target_eos=train_data.EOS, fine_tune=bool(args.fine_tune), encoder_type=args.encoder_type, encoder_hidden_size=CNN_channels, 
                       decoder_hidden_size=decoder_hidden_size, 
                       word_embedding_size=word_embedding_size, attention_dim=attention_dim, decoder_type=decoder_type, beam_width=beam_width, dropout=dropout,
                       transformer_layers=transformer_layers, num_heads=heads)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2)) 


fixed_image = "2090545563_a4e66ec76b.jpg"
writer = SummaryWriter("RESULTS_EFFICIENT_h1_ly3")

if mode == "train":
    
    best_bleu4 = 0.
    poor_iters = 0
    epoch = 1
    num_iters_change_lr = 4
    max_poor_iters = 10
    best_model = None
    best_optimizer = None
    best_loss = None
    best_epoch = None
    best_metrics = None


    model.to(device)
    print("Ground Truth captions: ", [" ".join(caption) for caption in val_data.all_captions[fixed_image]])


    while epoch <= max_epochs:
        print(epoch)
        model.train()
        loss, n_iter = train_for_epoch(model, train_dataloader, optimizer, device, n_iter, args)
        

        model.eval()
        print(f'Epoch {epoch}: loss={loss}')

        metrics_val = eval.get_pycoco_metrics(model, device, val_data, val_dataloader)
        metrics_train = eval.get_pycoco_metrics(model, device, train_data_to_eval, train_dataloader_to_eval)

        print(metrics_val)
        is_epoch_better = metrics_val['Bleu_4'] > best_bleu4
        if is_epoch_better:
        
            best_bleu4 = metrics_val['Bleu_4']
            best_model = copy.deepcopy(model)
            best_epoch = copy.deepcopy(epoch)
            best_optimizer = copy.deepcopy(optimizer)
            best_loss = copy.deepcopy(loss)
            best_metrics = copy.deepcopy(metrics_val)
        print("Predicted caption: ",predict(model, device, fixed_image))
        

        if epoch % 5 == 0:
            torch.save(best_model, model_save_path+ 'model_checkpoints_epoch'+str(epoch)+'_lyr_'+str(transformer_layers)+'_hds_'+str(heads)+'_'+str(_t)+'.pt')


        writer.add_scalar('Loss/train', loss, epoch)
        writer.add_scalar('Accuracy/BLEU/train', metrics_train['Bleu_4'], epoch)
        writer.add_scalar('Accuracy/CIDER/train', metrics_train['CIDEr'], epoch)
        writer.add_scalar('Accuracy/BLEU/val', metrics_val['Bleu_4'], epoch)
        writer.add_scalar('Accuracy/CIDER/val', metrics_val['CIDEr'], epoch)
        
        epoch += 1
        if epoch > max_epochs:
            torch.save(best_model, model_save_path+ 'model_final'+'_lyr_'+str(transformer_layers)+'_hds_'+str(heads)+'_'+str(_t)+'.pt')
            test_metrics = eval.get_pycoco_metrics(best_model, device, test_data, test_dataloader)
            test_metrics = 0
            #utils.save_model_and_result(model_save_path, args.experiment_name, best_model, decoder_type, best_optimizer, best_epoch, best_bleu4, best_loss, best_metrics, test_metrics)
            print(f'Finished {max_epochs} epochs')
        torch.cuda.empty_cache()

    writer.close()



