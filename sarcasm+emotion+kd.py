# -*- coding: utf-8 -*-

"""
DEPENDENCIES:

!pip install NRCLex
!python -m textblob.download_corpora
!pip install ftfy regex tqdm
!pip install git+https://github.com/openai/CLIP.git
!git clone https://github.com/FreddeFrallan/Multilingual-CLIP
!bash Multilingual-CLIP/get-weights.sh
!pip install transformers

"""

#IMPORTS:
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
import clip
import numpy as np




#from google.colab import drive
#drive.mount('/content/drive')

data_path = ""
#put the dataset path in csv to the data_path variable
data = pd.read_csv(data_path)







device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


#!cp -r /content/data /content/Multilingual-CLIP/
# %cd Multilingual-CLIP
from src import multilingual_clip
text_model = multilingual_clip.load_model('M-BERT-Distil-40')
# %cd ..




device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_model, compose = clip.load('RN50x4', device = device)

text_model = text_model.cpu()

def process(idx_val,arr):
  if idx_val=='0':
    arr.append(0)
  else:
    arr.append(1)

def get_data(data):
  #data = pd.read_csv(dataset_path)
  text = list(data['text'])
  img_path = list(data['Name'])
  label = list(data['Level1'])
  valence = list(data['Valence'])
  valence = list(map(lambda x: x - 1 , valence))
  arousal = list(data['Arousal'])
  arousal = list(map(lambda x: x - 1 , arousal))
  Fear = list(data['Fear'])
  Neglect	= list(data['Neglect'])
  irritation=list(data['irritation'])	
  Rage	= list(data['Rage'])
  Disgust	= list(data['Disgust'])
  Nervousness	= list(data['Nervousness'])
  Shame	= list(data['Shame'])
  Disappointment = list(data['Disappointment'])	
  Envy	= list(data['Envy'])
  Suffering	= list(data['Suffering'])
  Sadness	= list(data['Sadness'])
  Joy	= list(data['Joy'])
  Pride = list(data['Pride'])
  Sarcasm = list(data['Sarcasm'])
  Humor = list(data['Humor'])
  Intensity = list(data['Level2'])
  Target_1 = list(data['Level10_1'])
  Target_2 = list(data['Level10_2'])
  Target_3 = list(data['Level10_3'])
  Target_4 = list(data['Level10_4'])
  Target_5 = list(data['Level10_5'])
  Target_6 = list(data['Level10_6'])
  Target_7 = list(data['Level10_7'])
  #optimize memory for features
  name, text_features,image_features,l,a,v, fe, neg, ir, ra, disg, ner, sh, disa, en, su, sa, jo, pr, sar, hum, inten, t1,t2,t3,t4,t5,t6,t7= [],[],[],[],[], \
  [],[],[],[],[],[],[],[],[],[],[],[],[], [],[],[],[],[],[],[],[],[],[],[]
  for txt,img_name,L,A,V,fear,neglect,irritation,rage,disgust,nervousness, shame,disappointment,envy,suffering,sadness,joy,pride, \
  sarcasm,humor,intensity,target1,target2, \
  target3, target4, target5, target6, target7 \
   in tqdm(zip(text,img_path,label,arousal,valence, Fear,	Neglect,	irritation,	Rage,	Disgust,	\
              Nervousness,	Shame,	Disappointment,	Envy,	Suffering,	Sadness, Joy,	Pride, Sarcasm, Humor, \
               Intensity, Target_1, Target_2, Target_3, Target_4, Target_5, Target_6, Target_7)):
    
    try:
      #img = preprocess(Image.open('/content/drive/.shortcut-targets-by-id/1Z57L19m3ZpJ6bEPdyaIMYuI00Tc2RT1I/memes_our_dataset_hindi/my_meme_data/'+img)).unsqueeze(0).to(device)
      #name.append(img)
      img = Image.open('/content/drive/.shortcut-targets-by-id/1Z57L19m3ZpJ6bEPdyaIMYuI00Tc2RT1I/memes_our_dataset_hindi/my_meme_data/'+img_name)
    except Exception as e:
      print(e)
      continue

    #name.append(img_name)
    img = torch.stack([compose(img).to(device)])
    l.append(L)
    a.append(A)
    v.append(V)
    
    #print(fear,fe)
    process(fear,fe)
    process(neglect,neg)
    process(irritation,ir)
    process(rage,ra)
    process(disgust,disg)
    process(nervousness,ner)
    process(shame,sh)
    process(disappointment,disa)
    process(envy,en)
    process(suffering,su)
    process(sadness,sa)
    process(joy,jo)
    process(pride,pr)
    sar.append(sarcasm)
    hum.append(humor)
    inten.append(intensity)
    t1.append(target1)
    t2.append(target2)
    t3.append(target3)
    t4.append(target4)
    t5.append(target5)
    t6.append(target6)
    t7.append(target7)
  
    #txt = torch.as_tensor(txt)
    with torch.no_grad():
      temp_txt = text_model([txt]).detach().cpu().numpy()
      text_features.append(temp_txt)
      temp_img = clip_model.encode_image(img).detach().cpu().numpy()
      image_features.append(temp_img)

      del temp_txt
      del temp_img
      
      torch.cuda.empty_cache()
    
    del img
    #del txt
    torch.cuda.empty_cache()
  return text_features,image_features,l,v,a, fe, neg, ir, ra, disg, ner, sh, disa, en, su, sa, jo, pr, sar,hum, inten, t1, t2, t3, t4, t5 ,t6, t7


class HatefulDataset(Dataset):

  def __init__(self,data):
    
    self.t_f,self.i_f,self.label,self.v,self.a,self.fe, self.neg, self.ir, self.ra, \
    self.disg, self.ner, self.sh, self.disa, self.en, self.su, self.sa, self.jo, self.pr, \
    self.sar, self.hum, self.inten, self.t1, self.t2, self.t3, \
    self.t4, self.t5, self.t6, self.t7 = get_data(data)
    self.t_f = np.squeeze(np.asarray(self.t_f),axis=1)
    self.i_f = np.squeeze(np.asarray(self.i_f),axis=1)

    
    
  def __len__(self):
    return len(self.a)

  def __getitem__(self,idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    #print(idx)
    



    label = self.label[idx]
    T = self.t_f[idx,:]
    I = self.i_f[idx,:]
    v = self.v[idx]
    a = self.a[idx]
    fe = self.fe[idx]
    neg = self.neg[idx]
    ir = self.ir[idx]
    ra = self.ra[idx]
    disg = self.disg[idx]
    ner = self.ner[idx]
    sh = self.sh[idx]
    disa = self.disa[idx]
    en = self.en[idx]
    su = self.su[idx] 
    sa = self.sa[idx]
    jo = self.jo[idx]
    pr = self.pr[idx]
    sar = self.sar[idx]
    hum = self.hum[idx]
    inten = self.inten[idx]
    t1 = self.t1[idx]
    t2 = self.t2[idx]
    t3 = self.t3[idx]
    t4 = self.t4[idx]
    t5 = self.t5[idx]
    t6 = self.t6[idx] 
    t7 = self.t7[idx]
    #name = self.name[idx]

    sample = {'label':label,'processed_txt':T,'processed_img':I,'valence':v,'arousal':a , 'fear': fe, 'neglect': neg, \
              'irritation':ir, 'rage':ra, 'disgust':disg, 'nervousness':ner, 'shame':sh, 'disappointment':disa, \
              'envy':en, 'suffering':su, 'sadness':sa, 'joy':jo, 'pride':pr, 'sarcasm':sar, 'humor': hum, \
              'inten': inten, 't1':t1, 't2':t2, 't3':t3, 't4':t4, 't5':t5, 't6':t6, 't7': t7}
    return sample

outliers = []
for names in tqdm(list(data['Name'])):
  if not os.path.exists('/content/drive/.shortcut-targets-by-id/1Z57L19m3ZpJ6bEPdyaIMYuI00Tc2RT1I/memes_our_dataset_hindi/my_meme_data/'+names):
    outliers.append(names)

outliers

data = data[~data['Name'].isin(outliers)]

len(data)

class HatefulDatasetFinal(Dataset):

  def __init__(self,data,dataset,outliers):
    
    self.name, self.t_f,self.i_f,self.label,self.v,self.a,self.fe, self.neg, self.ir, self.ra, \
    self.disg, self.ner, self.sh, self.disa, self.en, self.su, self.sa, self.jo, self.pr, \
    self.sar, self.hum, self.inten, self.t1, self.t2, self.t3, \
    self.t4, self.t5, self.t6, self.t7 = \
    list(data['Name']), \
    [i['processed_txt'] for i in dataset], \
    [i['processed_img'] for i in dataset], \
    [i['label'] for i in dataset], \
    [i['valence'] for i in dataset], \
    [i['arousal'] for i in dataset], \
    [i['fear'] for i in dataset], \
    [i['neglect'] for i in dataset], \
    [i['irritation'] for i in dataset], \
    [i['rage'] for i in dataset], \
    [i['disgust'] for i in dataset], \
    [i['nervousness'] for i in dataset], \
    [i['shame'] for i in dataset], \
    [i['disappointment'] for i in dataset], \
    [i['envy'] for i in dataset], \
    [i['suffering'] for i in dataset], \
    [i['sadness'] for i in dataset], \
    [i['joy'] for i in dataset], \
    [i['pride'] for i in dataset], \
    [i['sarcasm'] for i in dataset], \
    [i['humor'] for i in dataset], \
    [i['inten'] for i in dataset], \
    [i['t1'] for i in dataset], \
    [i['t2'] for i in dataset], \
    [i['t3'] for i in dataset], \
    [i['t4'] for i in dataset], \
    [i['t5'] for i in dataset], \
    [i['t6'] for i in dataset], \
    [i['t7'] for i in dataset]
    
    self.t_f = np.asarray(self.t_f)
    self.i_f = np.asarray(self.i_f)
   
    
  def __len__(self):
    return len(self.a)

  def __getitem__(self,idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    #print(idx)
    


    #print(idx)
    label = self.label[idx]
    T = self.t_f[idx,:]
    I = self.i_f[idx,:]
    v = self.v[idx]
    a = self.a[idx]
    fe = self.fe[idx]
    neg = self.neg[idx]
    ir = self.ir[idx]
    ra = self.ra[idx]
    disg = self.disg[idx]
    ner = self.ner[idx]
    sh = self.sh[idx]
    disa = self.disa[idx]
    en = self.en[idx]
    su = self.su[idx] 
    sa = self.sa[idx]
    jo = self.jo[idx]
    pr = self.pr[idx]
    sar = self.sar[idx]
    hum = self.hum[idx]
    inten = self.inten[idx]
    t1 = self.t1[idx]
    t2 = self.t2[idx]
    t3 = self.t3[idx]
    t4 = self.t4[idx]
    t5 = self.t5[idx]
    t6 = self.t6[idx] 
    t7 = self.t7[idx]
    name = self.name[idx]

    sample = {'name':name, 'label':label,'processed_txt':T,'processed_img':I,'valence':v,'arousal':a , 'fear': fe, 'neglect': neg, \
              'irritation':ir, 'rage':ra, 'disgust':disg, 'nervousness':ner, 'shame':sh, 'disappointment':disa, \
              'envy':en, 'suffering':su, 'sadness':sa, 'joy':jo, 'pride':pr, 'sarcasm':sar, 'humor': hum, \
              'inten': inten, 't1':t1, 't2':t2, 't3':t3, 't4':t4, 't5':t5, 't6':t6, 't7': t7}
    return sample



#!pip install pytorch-lightning
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
class MFB(nn.Module):
    def __init__(self,img_feat_size, ques_feat_size, is_first, MFB_K, MFB_O, DROPOUT_R):
        super(MFB, self).__init__()
        #self.__C = __C
        self.MFB_K = MFB_K
        self.MFB_O = MFB_O
        self.DROPOUT_R = DROPOUT_R

        self.is_first = is_first
        self.proj_i = nn.Linear(img_feat_size, MFB_K * MFB_O)
        self.proj_q = nn.Linear(ques_feat_size, MFB_K * MFB_O)
        
        self.dropout = nn.Dropout(DROPOUT_R)
        self.pool = nn.AvgPool1d(MFB_K, stride = MFB_K)

    def forward(self, img_feat, ques_feat, exp_in=1):
        '''
            img_feat.size() -> (N, C, img_feat_size)    C = 1 or 100
            ques_feat.size() -> (N, 1, ques_feat_size)
            z.size() -> (N, C, MFB_O)
            exp_out.size() -> (N, C, K*O)
        '''
        batch_size = img_feat.shape[0]
        img_feat = self.proj_i(img_feat)                # (N, C, K*O)
        ques_feat = self.proj_q(ques_feat)              # (N, 1, K*O)
        
        exp_out = img_feat * ques_feat             # (N, C, K*O)
        exp_out = self.dropout(exp_out) if self.is_first else self.dropout(exp_out * exp_in)     # (N, C, K*O)
        z = self.pool(exp_out) * self.MFB_K         # (N, C, O)
        z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
        z = F.normalize(z.view(batch_size, -1))         # (N, C*O)
        z = z.view(batch_size, -1, self.MFB_O)      # (N, C, O)
        return z

import pickle
hm_data_hard = HatefulDataset(data)

hm_final = hm_data_hard


hm_final = HatefulDatasetFinal(data,hm_final,outliers)



import torch



torch.manual_seed(123)
t_p,te_p = torch.utils.data.random_split(hm_final,[5908,1478])

torch.manual_seed(123)
t_p,v_p = torch.utils.data.random_split(t_p,[5022,886])


"""
!rm -rf lightning logs

!cp /content/drive/MyDrive/memotion2/Memotion_2.zip /content/

!unzip Memotion_2.zip

!unzip -qq /content/train/train_images.zip

df = pd.read_csv('/content/train/train_data.csv')

df.head(10)
"""


def get_data_memotion(dataset_path):

  k = pd.read_csv(dataset_path)
  text = list(k['ocr_text'])
  img_path = list(k['name'])
  dataframe = k.apply(LabelEncoder().fit_transform)
  sarcasm = list(dataframe['sarcastic'])
  offensive = list(dataframe['offensive'])
  sentiment = list(dataframe['overall_sentiment'])

  text_features,image_features = [],[]
  for txt,img in tqdm(zip(text,img_path)):
    #txt = clip.tokenize(txt,truncate=True).to(device)
   
    #img = preprocess(Image.open('/content/train_images/'+img)).unsqueeze(0).to(device)
    img = Image.open('/content/train_images/'+img)
    img = torch.stack([compose(img).to(device)])
    with torch.no_grad():
      
      temp_txt = text_model([txt]).detach().cpu().numpy()
      text_features.append(temp_txt)
      temp_img = clip_model.encode_image(img).detach().cpu().numpy()
      image_features.append(temp_img)

      del temp_txt
      del temp_img
      #del temp_c
      torch.cuda.empty_cache()
    del txt
    del img
    #del c
    torch.cuda.empty_cache()

  

  text_features = np.squeeze(np.asarray(text_features,dtype=np.float32),axis=1)
  
  image_features = np.squeeze(np.asarray(image_features,dtype=np.float32),axis=1)
  

  return text_features,image_features, sarcasm, offensive, sentiment

class HatefulDatasetMemotion(Dataset):

  def __init__(self,data_path):
    
    self.t_f,self.i_f,self.sarcasm,self.label,self.sentiment = get_data_memotion(data_path)
    self.t_f = np.asarray(self.t_f)
    self.i_f = np.asarray(self.i_f)
    self.sarcasm = np.asarray(self.sarcasm)
    self.label = np.asarray(self.label)
    self.sentiment = np.asarray(self.sentiment)
    
    
    
    
  def __len__(self):
    return len(list(self.label))

  def __getitem__(self,idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    



    label = self.label[idx]
    T = self.t_f[idx,:]
    I = self.i_f[idx,:]
    sarcasm = self.sarcasm[idx]
    sentiment = self.sentiment[idx]
    
    
      
    
    

    




    sample = {'label':label,'processed_txt':T,'processed_img':I,'sarcasm':sarcasm,'sentiment':sentiment}
    return sample



from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score,precision_score
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import os
import warnings
import torch
t_m = HatefulDatasetMemotion('/content/train/train_data.csv')


torch.manual_seed(123)
t_m,v_m = torch.utils.data.random_split(t_m,[7000,500])



# train on memotion 2.0

warnings.filterwarnings("ignore", category=DeprecationWarning) 
class ClassifierMemotion(pl.LightningModule):

  def __init__(self):
    super().__init__()

    self.MFB = MFB(640,640,True,256,64,0.1)
    
    self.fin = torch.nn.Linear(64,4)
    self.fin_sarcasm = torch.nn.Linear(64,4)
    self.fin_sentiment = torch.nn.Linear(64,3)
    self.dropout_op = torch.nn.Dropout(0.2)
    
    
    

  def forward(self, x,y):
      
      x_,y_ = x,y
    
      #x = torch.cat((x,nrc),1)
      #print(x,y)
      #z = self.MFB(torch.unsqueeze(y.float(),axis=1),torch.unsqueeze(x.float(),axis=1))
      z = self.MFB(torch.unsqueeze(y.float(),axis=1),torch.unsqueeze(y.float(),axis=1))

      c = self.fin(torch.squeeze(z,dim=1))
      c_sarcasm = self.fin_sarcasm(torch.squeeze(z,dim=1))
      c_sentiment = self.fin_sentiment(torch.squeeze(z,dim=1))
      # probability distribution over labels
      c = torch.log_softmax(c, dim=1)
      c_sarcasm = torch.log_softmax(c_sarcasm, dim=1)
      c_sentiment = torch.log_softmax(c_sentiment, dim=1)
      #c_emo = torch.softmax(c_emo, dim=1)
      return z,c,c_sarcasm,c_sentiment

  def cross_entropy_loss(self, logits, labels):
    return F.nll_loss(logits, labels)

  def training_step(self, train_batch, batch_idx):
      lab,txt,img,sarcasm,sentiment = train_batch
      lab = train_batch[lab]
      txt = train_batch[txt]
      img = train_batch[img]
      sarcasm = train_batch[sarcasm]
      sentiment = train_batch[sentiment]
      
      
      hidden,logits,logit_sarcasm,logit_sentiment = self.forward(txt,img)
      loss1 = self.cross_entropy_loss(logits, lab)
      loss2 = self.cross_entropy_loss(logit_sarcasm, sarcasm)
      loss3 = self.cross_entropy_loss(logit_sentiment, sentiment)
      #loss = loss1+loss2+loss3
      #loss = loss3
      loss = loss2+loss3
      self.log('train_loss', loss)
      return loss


  def validation_step(self, val_batch, batch_idx):
      lab,txt,img,sarcasm,sentiment = val_batch
      lab = val_batch[lab]
      txt = val_batch[txt]
      img = val_batch[img]
      sarcasm = val_batch[sarcasm]
      sentiment = val_batch[sentiment]
      
      
      hidden,logits,logit_sarcasm,logit_sentiment = self.forward(txt,img)
      tmp = np.argmax(logits.detach().cpu().numpy(),axis=-1)
      loss = self.cross_entropy_loss(logits, lab)
      lab = lab.detach().cpu().numpy()
      self.log('val_acc', accuracy_score(lab,tmp))
      self.log('val_acc_sarcasm', f1_score(sarcasm.detach().cpu().numpy(),np.argmax(logit_sarcasm.detach().cpu().numpy(),axis=-1),average='macro'))
      #self.log('val_roc_auc',roc_auc_score(lab,tmp))
      self.log('val_loss', loss)
      tqdm_dict = {'val_acc': accuracy_score(lab,tmp)}
      #print('Val acc {}'.format(accuracy_score(lab,tmp)))
      return {
                'progress_bar': tqdm_dict,
              'val_acc_sarcasm': f1_score(sarcasm.detach().cpu().numpy(),np.argmax(logit_sarcasm.detach().cpu().numpy(),axis=-1),average='macro'),
              'val_acc_sentiment': f1_score(sentiment.detach().cpu().numpy(),np.argmax(logit_sentiment.detach().cpu().numpy(),axis=-1),average='macro')
      }
      
  def validation_epoch_end(self, validation_step_outputs):
    outs = []
    for out in validation_step_outputs:
      outs.append(out['val_acc_sarcasm'])
    self.log('val_acc_all', sum(outs)/len(outs))
    print(f'***Acc at epoch end {sum(outs)/len(outs)}****')

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=5e-3)
    return optimizer

  def predict_step(self, batch, batch_idx: int , dataloader_idx: int = None):
      lab,txt,img,sarcasm,sentiment = batch
      lab = batch[lab]
      txt = batch[txt]
      img = batch[img]
      z,_,_,_ = self(txt,img)
      return z
class HmDataModule(pl.LightningDataModule):

  def setup(self, stage):
    
      
    
    self.hm_train = t_m
    self.hm_test = v_m
    

  def train_dataloader(self):
    return DataLoader(self.hm_train, batch_size=128)

  def val_dataloader(self):
    return DataLoader(self.hm_test, batch_size=64)

data_module = HmDataModule()
from pytorch_lightning.callbacks import ModelCheckpoint
checkpoint_callback = ModelCheckpoint(
     monitor='val_acc_all',
     dirpath='primary/ckpts/',
     filename='memotion2-ckpt-epoch{epoch:02d}-val_acc_all{val_acc_all:.2f}',
     auto_insert_metric_name=False,
     save_top_k=1,
    mode="max",
 )
# train
from pytorch_lightning import seed_everything
seed_everything(123, workers=True)
hm_model_memotion = ClassifierMemotion()
#trainer = pl.Trainer(gpus=1,deterministic=True,max_epochs=60,callbacks=[checkpoint_callback])
trainer = pl.Trainer(gpus=1,max_epochs=60,callbacks=[checkpoint_callback])

trainer.fit(hm_model_memotion, data_module)

!ls primary/ckpts

!rm -rf primary

!cp /content/primary/ckpts/memotion2-ckpt-epoch16-val_acc_all0.31.ckpt /content/drive/MyDrive/memotion2/ckpts

device = 'cuda' if torch.cuda.is_available() else 'cpu'
hm_model_emotion = hm_model_memotion.load_from_checkpoint('/content/primary/ckpts/memotion2-ckpt-epoch24-val_acc_all0.29.ckpt')
hm_model_memotion.to(device)

#!rm -rf /content/our_ds/ckpts

!rm -rf primary

final_train = {}
final_val = {}

o_p,e1_p,e2_p,e3_p,e4_p,e5_p,e6_p,e7_p,e8_p,e9_p,e10_p,e11_p,e12_p,e13_p,i_p = \
[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
o_t,e1_t,e2_t,e3_t,e4_t,e5_t,e6_t,e7_t,e8_t,e9_t,e10_t,e11_t,e12_t,e13_t,i_t = \
[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
def append_p(tba,appendee):
  for i in np.argmax(tba.detach().cpu().numpy(),axis=-1):
    appendee.append(i)
def append_gt(tba,appendee):
  for i in tba.detach().cpu().numpy():
    appendee.append(i)
N = []
val_f1,train_f1 = [],[]

import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=":16:8"

#THIS is for determining perf on our dataset (sarcasm+emotion)
pred_e = 0
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score,precision_score
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import roc_auc_score
import os

class Classifier(pl.LightningModule):

  def __init__(self):
    super().__init__()

    self.MFB = MFB(640,640,True,256,64,0.1)
    self.loss_fn_emotion=torch.nn.KLDivLoss(reduction='batchmean',log_target=True)
    #self.coupler_lstm = torch.nn.LSTMCell(input_size=64, hidden_size=64, bias=True, device='cuda', dtype=None)
    self.coupler_gru = torch.nn.GRUCell(input_size=64, hidden_size=64, bias=True, device=device, dtype=None)
    self.ffn_coupler = torch.nn.Linear(128,64)
    #self.layer_1 = torch.nn.Linear(512, 128)
    #self.layer_2 = torch.nn.Linear(512, 128)
    #self.pre_fin = torch.nn.Linear(128,64)
    self.encode_text = torch.nn.Linear(1280,64)
    self.fin = torch.nn.Linear(64,2)
    self.fin_v = torch.nn.Linear(64,4)
    self.fin_a = torch.nn.Linear(64,4)
    self.fin_sarcasm = torch.nn.Linear(64,3)
    self.fin_e1 = torch.nn.Linear(64,2)
    self.fin_e2 = torch.nn.Linear(64,2)
    self.fin_e3 = torch.nn.Linear(64,2)
    self.fin_e4 = torch.nn.Linear(64,2)
    self.fin_e5 = torch.nn.Linear(64,2)
    self.fin_e6 = torch.nn.Linear(64,2)
    self.fin_e7 = torch.nn.Linear(64,2)
    self.fin_e8 = torch.nn.Linear(64,2)
    self.fin_e9 = torch.nn.Linear(64,2)
    self.fin_e10 = torch.nn.Linear(64,2)
    self.fin_e11 = torch.nn.Linear(64,2)
    self.fin_e12 = torch.nn.Linear(64,2)
    self.fin_e13 = torch.nn.Linear(64,2)
    self.fin_inten = torch.nn.Linear(64,3)
    self.fin_target_ident = torch.nn.Linear(64,7)
    self.fin_emotion_mult = torch.nn.Linear(64,13)
    

  def forward(self, x,y, hidden_pred):
      
      x_,y_ = x,y
      x = x.float()
      y = y.float()
      
      z_ = self.MFB(torch.unsqueeze(y,axis=1),torch.unsqueeze(x,axis=1))
      z = self.coupler_gru(torch.squeeze(z_,dim=1),torch.squeeze(hidden_pred.to(device),dim=1))
      z = torch.unsqueeze(z,dim=1)
      

      c = self.fin(torch.squeeze(z,dim=1))
      c_inten = self.fin_inten(torch.squeeze(z,dim=1))
      c_v = self.fin_v(torch.squeeze(z,dim=1))
      c_a = self.fin_a(torch.squeeze(z,dim=1))
      c_e1 = self.fin_e1(torch.squeeze(z,dim=1))
      c_e2 = self.fin_e2(torch.squeeze(z,dim=1))
      c_e3 = self.fin_e3(torch.squeeze(z,dim=1))
      c_e4 = self.fin_e4(torch.squeeze(z,dim=1))
      c_e5 = self.fin_e5(torch.squeeze(z,dim=1))
      c_e6 = self.fin_e6(torch.squeeze(z,dim=1))
      c_e7 = self.fin_e7(torch.squeeze(z,dim=1))
      c_e8 = self.fin_e8(torch.squeeze(z,dim=1))
      c_e9 = self.fin_e9(torch.squeeze(z,dim=1))
      c_e10 = self.fin_e10(torch.squeeze(z,dim=1))
      c_e11 = self.fin_e11(torch.squeeze(z,dim=1))
      c_e12 = self.fin_e12(torch.squeeze(z,dim=1))
      c_e13 = self.fin_e13(torch.squeeze(z,dim=1))
      c_sarcasm = self.fin_sarcasm(torch.squeeze(z,dim=1))
      # probability distribution over labels
      c = torch.log_softmax(c, dim=1)
      c_inten = torch.log_softmax(c_inten, dim=1)
      c_a = torch.log_softmax(c_a, dim=1)
      c_v = torch.log_softmax(c_v, dim=1)
      c_e1 = torch.log_softmax(c_e1, dim=1)
      c_sarcasm = torch.log_softmax(c_sarcasm, dim=1)
      c_e2 = torch.log_softmax(c_e2, dim=1)
      c_e3 = torch.log_softmax(c_e3, dim=1)
      c_e4 = torch.log_softmax(c_e4, dim=1)
      
      c_e5 = torch.log_softmax(c_e5, dim=1)
      c_e6 = torch.log_softmax(c_e6, dim=1)
      c_e7 = torch.log_softmax(c_e7, dim=1)
      
      c_e8 = torch.log_softmax(c_e8, dim=1)
      c_e9 = torch.log_softmax(c_e9, dim=1)
      c_e10 = torch.log_softmax(c_e10, dim=1)
      
      c_e11 = torch.log_softmax(c_e11, dim=1)
      c_e12 = torch.log_softmax(c_e12, dim=1)
      c_e13 = torch.log_softmax(c_e13, dim=1)
      c_target = self.fin_target_ident(torch.squeeze(z,dim=1))
      c_emotion = self.fin_emotion_mult(torch.squeeze(z,dim=1))
      return z,c,c_a,c_v,c_e1,c_e2,c_e3,c_e4,c_e5,c_e6,c_e7,c_e8,c_e9,c_e10,c_e11,c_e12, c_e13, c_inten, c_target, c_sarcasm, c_emotion

  def cross_entropy_loss(self, logits, labels):
    return F.nll_loss(logits, labels)
  

  def training_step(self, train_batch, batch_idx):
      _,lab,txt,img,val,arou,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12,e13,sarcasm,_,intensity,t1,t2,t3,t4,t5,t6,t7 = train_batch
      
      lab = train_batch[lab]
      #print(lab)
      txt = train_batch[txt]
      #print(txt)
      
      img = train_batch[img]
      val = train_batch[val]
      arou = train_batch[arou]
      #print(a)
      e1 = train_batch[e1]
      e2 = train_batch[e2]
      e3 = train_batch[e3]
      e4 = train_batch[e4]
      e5 = train_batch[e5]
      e6 = train_batch[e6]
      e7 = train_batch[e7]
      e8 = train_batch[e8]
      e9 = train_batch[e9]
      e10 = train_batch[e10]
      e11 = train_batch[e11]
      e12 = train_batch[e12]
      e13 = train_batch[e13]
      intensity = train_batch[intensity]
      sarcasm = train_batch[sarcasm]
      with torch.no_grad():
        hidden_pred,_,_,_ = hm_model_memotion(txt.to(device),img.to(device))



      t1,t2,t3,t4,t5,t6,t7 = torch.unsqueeze(train_batch[t1],1),torch.unsqueeze(train_batch[t2],1),\
      torch.unsqueeze(train_batch[t3],1),torch.unsqueeze(train_batch[t4],1),torch.unsqueeze(train_batch[t5],1),\
      torch.unsqueeze(train_batch[t6],1),torch.unsqueeze(train_batch[t7],1)
      gt_target = torch.cat((t1,t2,t3,t4,t5,t6,t7),1) #ground truth target

      gt_emotion = torch.cat((torch.unsqueeze(e1,1),torch.unsqueeze(e2,1),torch.unsqueeze(e3,1),torch.unsqueeze(e4,1),torch.unsqueeze(e5,1),torch.unsqueeze(e6,1),\
                              torch.unsqueeze(e7,1),torch.unsqueeze(e8,1),torch.unsqueeze(e9,1),torch.unsqueeze(e10,1),torch.unsqueeze(e11,1),torch.unsqueeze(e12,1),\
                              torch.unsqueeze(e13,1)),1)

      z,logit_offen,logit_arou,logit_val, a,b,c,d,e,f,g,h,i,j,k,l,m,inten,logit_target,logit_sarcasm,logit_emotion = self.forward(txt,img,hidden_pred) # logit_target is logits of target
      
      hidden_pred = torch.squeeze(hidden_pred,dim=1)
      hidden_pred = F.log_softmax(hidden_pred / 1).float()
      z = torch.squeeze(z,dim=1)
      z = F.log_softmax(z / 1).float()
      loss_transfer = self.loss_fn_emotion(z, hidden_pred)
    
      
      loss1 = self.cross_entropy_loss(logit_offen, lab)
      #loss2 = self.cross_entropy_loss(logit_arou, arou)
      #loss3 = self.cross_entropy_loss(logit_val, val)
      loss4 = self.cross_entropy_loss(a, e1)
      loss5 = self.cross_entropy_loss(b, e2)
      loss6 = self.cross_entropy_loss(c, e3)
      loss7 = self.cross_entropy_loss(d, e4)
      loss8 = self.cross_entropy_loss(e, e5)
      loss9 = self.cross_entropy_loss(f, e6)
      loss10 = self.cross_entropy_loss(g, e7)
      loss11 = self.cross_entropy_loss(h, e8)
      loss12 = self.cross_entropy_loss(i, e9)
      loss13 = self.cross_entropy_loss(j, e10)
      loss14 = self.cross_entropy_loss(k, e11)
      loss15 = self.cross_entropy_loss(l, e12)
      loss16 = self.cross_entropy_loss(m, e13)
      loss17 = self.cross_entropy_loss(inten, intensity)
      
      loss18 = F.binary_cross_entropy_with_logits(logit_target.float(), gt_target.float())
      loss_emo_mult = F.binary_cross_entropy_with_logits(logit_emotion.float(), gt_emotion.float())
      loss_sarcasm = self.cross_entropy_loss(logit_sarcasm, sarcasm)
      
      loss = loss_sarcasm+loss_emo_mult
      
      self.log('train_loss', loss)
      f1_sarcasm = f1_score(sarcasm.detach().cpu().numpy(),np.argmax(logit_sarcasm.detach().cpu().numpy(),axis=-1),average='macro')

      #return loss
      return {"loss": loss, "f1": f1_sarcasm}
  def training_epoch_end(self, training_step_outputs):
    out_f1 = []
    for out in training_step_outputs:
      out_f1.append(out['f1'])
    train_f1.append(sum(out_f1)/len(out_f1))
    


  def validation_step(self, val_batch, batch_idx):
      _,lab,txt,img,val,arou,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12,e13,sarcasm,_,intensity,t1,t2,t3,t4,t5,t6,t7 = val_batch
      #print(val_batch)
      lab = val_batch[lab]
      txt = val_batch[txt]
      img = val_batch[img]
      val = val_batch[val]
      arou = val_batch[arou]
      e1 = val_batch[e1]
      e2 = val_batch[e2]
      e3 = val_batch[e3]
      e4 = val_batch[e4]
      e5 = val_batch[e5]
      e6 = val_batch[e6]
      e7 = val_batch[e7]
      e8 = val_batch[e8]
      e9 = val_batch[e9]
      e10 = val_batch[e10]
      e11 = val_batch[e11]
      e12 = val_batch[e12]
      e13 = val_batch[e13]
      sarcasm = val_batch[sarcasm]
      intensity = val_batch[intensity]
      t1,t2,t3,t4,t5,t6,t7 = torch.unsqueeze(val_batch[t1],1),torch.unsqueeze(val_batch[t2],1),\
      torch.unsqueeze(val_batch[t3],1),torch.unsqueeze(val_batch[t4],1),torch.unsqueeze(val_batch[t5],1),\
      torch.unsqueeze(val_batch[t6],1),torch.unsqueeze(val_batch[t7],1)
      #print(t1.size())
      gt_target = torch.cat((t1,t2,t3,t4,t5,t6,t7),1) #ground truth target
      gt_emotion = torch.cat((torch.unsqueeze(e1,1),torch.unsqueeze(e2,1),torch.unsqueeze(e3,1),torch.unsqueeze(e4,1),torch.unsqueeze(e5,1),torch.unsqueeze(e6,1),\
                              torch.unsqueeze(e7,1),torch.unsqueeze(e8,1),torch.unsqueeze(e9,1),torch.unsqueeze(e10,1),torch.unsqueeze(e11,1),torch.unsqueeze(e12,1),\
                              torch.unsqueeze(e13,1)),1)
      with torch.no_grad():
        hidden_pred,_,_,_ = hm_model_memotion(txt.to(device),img.to(device))
      _,logits,logit_arou,logit_val, a,b,c,d,e,f,g,h,i,j,k,l,m,inten,logit_target,logit_sarcasm,logit_emotion = self.forward(txt,img,hidden_pred)
      


      
      
      tmp = np.argmax(logits.detach().cpu().numpy(),axis=-1)
      loss = self.cross_entropy_loss(logits, lab)
      lab = lab.detach().cpu().numpy()
      self.log('val_acc', f1_score(lab,tmp,average='macro'))
      #self.log('val_roc_auc',roc_auc_score(lab,tmp))
      self.log('val_loss', loss)
      tqdm_dict = {'val_acc': f1_score(lab,tmp,average='macro')}
      #print('Val acc {}'.format(accuracy_score(lab,tmp)))
      return {
                'progress_bar': tqdm_dict,
              'val_loss_target': F.binary_cross_entropy_with_logits(logit_target.float(), gt_target.float()),
              'val_loss_emotion_multilabel': F.binary_cross_entropy_with_logits(logit_emotion.float(), gt_emotion.float()),
              'val_acc e1': accuracy_score(e1.detach().cpu().numpy(),np.argmax(a.detach().cpu().numpy(),axis=-1)),
      'val_acc e2': accuracy_score(e2.detach().cpu().numpy(),np.argmax(b.detach().cpu().numpy(),axis=-1)),
      'val_acc e3': accuracy_score(e3.detach().cpu().numpy(),np.argmax(c.detach().cpu().numpy(),axis=-1)),
      'val_acc e4': accuracy_score(e4.detach().cpu().numpy(),np.argmax(d.detach().cpu().numpy(),axis=-1)),
      'val_acc e5': accuracy_score(e5.detach().cpu().numpy(),np.argmax(e.detach().cpu().numpy(),axis=-1)),
      'val_acc e6': accuracy_score(e6.detach().cpu().numpy(),np.argmax(f.detach().cpu().numpy(),axis=-1)),
      'val_acc e7': accuracy_score(e7.detach().cpu().numpy(),np.argmax(g.detach().cpu().numpy(),axis=-1)),
      'val_acc e8': accuracy_score(e8.detach().cpu().numpy(),np.argmax(h.detach().cpu().numpy(),axis=-1)),
      'val_acc e9': accuracy_score(e9.detach().cpu().numpy(),np.argmax(i.detach().cpu().numpy(),axis=-1)),
      'val_acc e10': accuracy_score(e10.detach().cpu().numpy(),np.argmax(j.detach().cpu().numpy(),axis=-1)),
      'val_acc e11': accuracy_score(e11.detach().cpu().numpy(),np.argmax(k.detach().cpu().numpy(),axis=-1)),
      'val_acc e12': accuracy_score(e12.detach().cpu().numpy(),np.argmax(l.detach().cpu().numpy(),axis=-1)),
      'val_acc e13': accuracy_score(e13.detach().cpu().numpy(),np.argmax(m.detach().cpu().numpy(),axis=-1)),
      'val_acc intensity': f1_score(intensity.detach().cpu().numpy(),np.argmax(inten.detach().cpu().numpy(),axis=-1),average='macro'),
       'val_acc sarcasm': accuracy_score(sarcasm.detach().cpu().numpy(),np.argmax(logit_sarcasm.detach().cpu().numpy(),axis=-1)),
       'f1 sarcasm': f1_score(sarcasm.detach().cpu().numpy(),np.argmax(logit_sarcasm.detach().cpu().numpy(),axis=-1),average='macro')
      }
      
  def validation_epoch_end(self, validation_step_outputs):
    outs = []
    outs1,outs2,outs3,outs4,outs5,outs6,outs7,outs8,outs9,outs10,outs11,outs12,outs13,outs14,outs16,outs17 = \
    [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
    outs15 = []
    outs18 = []
    for out in validation_step_outputs:
      outs.append(out['progress_bar']['val_acc'])
      outs1.append(out['val_acc e1'])
      outs2.append(out['val_acc e2'])
      outs3.append(out['val_acc e3'])
      outs4.append(out['val_acc e4'])
      outs5.append(out['val_acc e5'])
      outs6.append(out['val_acc e6'])
      outs7.append(out['val_acc e7'])
      outs8.append(out['val_acc e8'])
      outs9.append(out['val_acc e9'])
      outs10.append(out['val_acc e10'])
      outs11.append(out['val_acc e11'])
      outs12.append(out['val_acc e12'])
      outs13.append(out['val_acc e13'])
      outs14.append(out['val_acc intensity'])
      outs15.append(out['val_loss_target'])
      outs16.append(out['val_loss_emotion_multilabel'])
      outs17.append(out['val_acc sarcasm'])
      outs18.append(out['f1 sarcasm'])
    self.log('val_acc_all', sum(outs)/len(outs))
    self.log('val_loss_target', sum(outs15)/len(outs15))
    self.log('val_acc_all e1', sum(outs1)/len(outs1))
    self.log('val_acc_all e2', sum(outs2)/len(outs2))
    self.log('val_acc_all e3', sum(outs3)/len(outs3))
    self.log('val_acc_all e4', sum(outs4)/len(outs4))
    self.log('val_acc_all e5', sum(outs5)/len(outs5))
    self.log('val_acc_all e6', sum(outs6)/len(outs6))
    self.log('val_acc_all e7', sum(outs7)/len(outs7))
    self.log('val_acc_all e8', sum(outs8)/len(outs8))
    self.log('val_acc_all e9', sum(outs9)/len(outs9))
    self.log('val_acc_all e10', sum(outs10)/len(outs10))
    self.log('val_acc_all e11', sum(outs11)/len(outs11))
    self.log('val_acc_all e12', sum(outs12)/len(outs12))
    self.log('val_acc_all e13', sum(outs13)/len(outs13))
    self.log('val_acc_all inten', sum(outs14)/len(outs14))
    self.log('val_loss_all emo', sum(outs16)/len(outs16))
    self.log('val_acc_all sarcasm', sum(outs17)/len(outs17))
    self.log('val_f1_all sarcasm', sum(outs18)/len(outs18))
    
    print(f'***f1 at epoch end {sum(outs)/len(outs)}****')
    print(f'***val acc inten at epoch end {sum(outs14)/len(outs14)}****')
    print(f'***val loss emotion at epoch end {sum(outs16)/len(outs16)}****')
    print(f'***val acc sarcasm at epoch end {sum(outs17)/len(outs17)}****')
    print(f'***val f1 sarcasm at epoch end {sum(outs18)/len(outs18)}****')
    val_f1.append(sum(outs18)/len(outs18))

  def test_step(self, batch, batch_idx):
      name,lab,txt,img,val,arou,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12,e13,sarcasm,_,intensity,t1,t2,t3,t4,t5,t6,t7 = batch
      name = batch[name]
      lab = batch[lab]
      txt = batch[txt]
      img = batch[img]
      e1 = batch[e1]
      e2 = batch[e2]
      e3 = batch[e3]
      e4 = batch[e4]
      e5 = batch[e5]
      e6 = batch[e6]
      e7 = batch[e7]
      e8 = batch[e8]
      e9 = batch[e9]
      e10 = batch[e10]
      e11 = batch[e11]
      e12 = batch[e12]
      e13 = batch[e13]
      intensity = batch[intensity]
      sarcasm = batch[sarcasm]
      t1,t2,t3,t4,t5,t6,t7 = torch.unsqueeze(batch[t1],1),torch.unsqueeze(batch[t2],1),\
      torch.unsqueeze(batch[t3],1),torch.unsqueeze(batch[t4],1),torch.unsqueeze(batch[t5],1),\
      torch.unsqueeze(batch[t6],1),torch.unsqueeze(batch[t7],1)
      gt_target = torch.cat((t1,t2,t3,t4,t5,t6,t7),1)
      gt_emotion = torch.cat((torch.unsqueeze(e1,1),torch.unsqueeze(e2,1),torch.unsqueeze(e3,1),torch.unsqueeze(e4,1),torch.unsqueeze(e5,1),torch.unsqueeze(e6,1),\
                              torch.unsqueeze(e7,1),torch.unsqueeze(e8,1),torch.unsqueeze(e9,1),torch.unsqueeze(e10,1),torch.unsqueeze(e11,1),torch.unsqueeze(e12,1),\
                              torch.unsqueeze(e13,1)),1)
      with torch.no_grad():
        hidden_pred,_,_,_ = hm_model_memotion(txt.to(device),img.to(device))
      _,logits,logit_arou,logit_val, a,b,c,d,e,f,g,h,i,j,k,l,m,inten,logit_target,logit_sarcasm,logit_emotion = self.forward(txt,img,hidden_pred)
      #self.log('val_acc 1', accuracy_score(lab.detach().cpu().numpy(),np.argmax(logits.detach().cpu().numpy(),axis=-1)))
      for n in name:
        N.append(n)
      append_gt(lab,o_t); append_gt(e1,e1_t); append_gt(e2,e2_t); append_gt(e3,e3_t); append_gt(e4,e4_t); append_gt(e5,e5_t);\
      append_gt(e6,e6_t); append_gt(e7,e7_t); append_gt(e8,e8_t); append_gt(e9,e9_t); append_gt(e10,e10_t); append_gt(e11,e11_t); \
      append_gt(e12,e12_t); append_gt(e13,e13_t); append_gt(intensity,i_t);

      append_p(logits,o_p); append_p(a,e1_p); append_p(b,e2_p); append_p(c,e3_p); append_p(d,e4_p); append_p(e,e5_p);\
      append_p(f,e6_p); append_p(g,e7_p); append_p(h,e8_p); append_p(i,e9_p); append_p(j,e10_p); append_p(k,e11_p); \
      append_p(l,e12_p); append_p(m,e13_p); append_p(inten,i_p);

      tmp = np.argmax(logits.detach().cpu().numpy(),axis=-1)
      loss = self.cross_entropy_loss(logits, lab)
      lab = lab.detach().cpu().numpy()
      self.log('test_acc', accuracy_score(lab,tmp))
      self.log('test f1',f1_score(sarcasm.detach().cpu().numpy(),np.argmax(logit_sarcasm.detach().cpu().numpy(),axis=-1),average='macro'))
      np.save('multitask_logit_emotion.npy',logit_emotion.detach().cpu().numpy())
      np.save('multitask_logit_offensive.npy',lab)
      np.save('multitask_logit_intensity.npy',inten.detach().cpu().numpy())
      #self.log('test confusion matrix',confusion_matrix(lab,tmp))
      print(f'confusion matrix {confusion_matrix(sarcasm.detach().cpu().numpy(),np.argmax(logit_sarcasm.detach().cpu().numpy(),axis=-1))}')
      print(f'confusion matrix intensity {confusion_matrix(intensity.detach().cpu().numpy(),np.argmax(inten.detach().cpu().numpy(),axis=-1))}')
      print(f'confusion matrix offensive {confusion_matrix(lab,tmp)}')

      #self.log('test_roc_auc',roc_auc_score(sarcasm.detach().cpu().numpy(),np.argmax(logit_sarcasm.detach().cpu().numpy(),axis=-1)))
      #self.log('F1',f1_score(sarcasm.detach().cpu().numpy(),np.argmax(logit_sarcasm.detach().cpu().numpy(),axis=-1)))
      #self.log('recall',recall_score(sarcasm.detach().cpu().numpy(),np.argmax(logit_sarcasm.detach().cpu().numpy(),axis=-1)))
      #self.log('precision',precision_score(sarcasm.detach().cpu().numpy(),np.argmax(logit_sarcasm.detach().cpu().numpy(),axis=-1)))
      best_threshold = np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])
      y_test = torch.nn.Sigmoid()(logit_emotion)
      y_test = y_test.detach().cpu().numpy()
      y_pred = np.array([[1 if y_test[i][j]>=best_threshold[j] else 0 for j in range(13)] for i in range(len(y_test))])
      print(y_pred)
      total_correctly_predicted = len([i for i in range(len(y_test)) if (y_test[i]==y_pred[i]).sum() == 13])
      self.log('test_loss', loss)
      print(total_correctly_predicted)
      pred_e = y_test
      return {'test_loss': loss,
              'test_loss_target': F.binary_cross_entropy_with_logits(logit_target.float(), gt_target.float()),
              'test_loss_emotion_multilabel': F.binary_cross_entropy_with_logits(logit_emotion.float(), gt_emotion.float()),
               'test_acc':f1_score(lab,tmp,average='macro'),
              'test_acc e1': accuracy_score(e1.detach().cpu().numpy(),np.argmax(a.detach().cpu().numpy(),axis=-1)),
              'test_acc e2': accuracy_score(e2.detach().cpu().numpy(),np.argmax(b.detach().cpu().numpy(),axis=-1)),
              'test_acc e3': accuracy_score(e3.detach().cpu().numpy(),np.argmax(c.detach().cpu().numpy(),axis=-1)),
              'test_acc e4': accuracy_score(e4.detach().cpu().numpy(),np.argmax(d.detach().cpu().numpy(),axis=-1)),
              'test_acc e5': accuracy_score(e5.detach().cpu().numpy(),np.argmax(e.detach().cpu().numpy(),axis=-1)),
              'test_acc e6': accuracy_score(e6.detach().cpu().numpy(),np.argmax(f.detach().cpu().numpy(),axis=-1)),
              'test_acc e7': accuracy_score(e7.detach().cpu().numpy(),np.argmax(g.detach().cpu().numpy(),axis=-1)),
              'test_acc e8': accuracy_score(e8.detach().cpu().numpy(),np.argmax(h.detach().cpu().numpy(),axis=-1)),
              'test_acc e9': accuracy_score(e9.detach().cpu().numpy(),np.argmax(i.detach().cpu().numpy(),axis=-1)),
              'test_acc e10': accuracy_score(e10.detach().cpu().numpy(),np.argmax(j.detach().cpu().numpy(),axis=-1)),
              'test_acc e11': accuracy_score(e11.detach().cpu().numpy(),np.argmax(k.detach().cpu().numpy(),axis=-1)),
              'test_acc e12': accuracy_score(e12.detach().cpu().numpy(),np.argmax(l.detach().cpu().numpy(),axis=-1)),
              'test_acc e13': accuracy_score(e13.detach().cpu().numpy(),np.argmax(m.detach().cpu().numpy(),axis=-1)),
              'test_acc inten': f1_score(intensity.detach().cpu().numpy(),np.argmax(inten.detach().cpu().numpy(),axis=-1),average='macro'),
              'test_acc sarcasm': accuracy_score(sarcasm.detach().cpu().numpy(),np.argmax(logit_sarcasm.detach().cpu().numpy(),axis=-1)),
              'f1 sarcasm': f1_score(sarcasm.detach().cpu().numpy(),np.argmax(logit_sarcasm.detach().cpu().numpy(),axis=-1),average='macro')}
  def test_epoch_end(self, outputs):
        # OPTIONAL
        outs = []
        outs1,outs2,outs3,outs4,outs5,outs6,outs7,outs8,outs9,outs10,outs11,outs12,outs13,outs14 = \
        [],[],[],[],[],[],[],[],[],[],[],[],[],[]
        outs15 = []
        outs16 = []
        outs17 = []
        outs18 = []
        for out in outputs:
          outs15.append(out['test_loss_target'])
          outs.append(out['test_acc'])
          outs1.append(out['test_acc e1'])
          outs2.append(out['test_acc e2'])
          outs3.append(out['test_acc e3'])
          outs4.append(out['test_acc e4'])
          outs5.append(out['test_acc e5'])
          outs6.append(out['test_acc e6'])
          outs7.append(out['test_acc e7'])
          outs8.append(out['test_acc e8'])
          outs9.append(out['test_acc e9'])
          outs10.append(out['test_acc e10'])
          outs11.append(out['test_acc e11'])
          outs12.append(out['test_acc e12'])
          outs13.append(out['test_acc e13'])
          outs14.append(out['test_acc inten'])
          outs16.append(out['test_acc sarcasm'])
          outs17.append(out['test_loss_emotion_multilabel'])
          outs18.append(out['f1 sarcasm'])

        #print(outs)
        self.log('final test accuracy', sum(outs)/len(outs))
        self.log('test_acc_all e1', sum(outs1)/len(outs1))
        self.log('test_acc_all e2', sum(outs2)/len(outs2))
        self.log('test_acc_all e3', sum(outs3)/len(outs3))
        self.log('test_acc_all e4', sum(outs4)/len(outs4))
        self.log('test_acc_all e5', sum(outs5)/len(outs5))
        self.log('test_acc_all e6', sum(outs6)/len(outs6))
        self.log('test_acc_all e7', sum(outs7)/len(outs7))
        self.log('test_acc_all e8', sum(outs8)/len(outs8))
        self.log('test_acc_all e9', sum(outs9)/len(outs9))
        self.log('test_acc_all e10', sum(outs10)/len(outs10))
        self.log('test_acc_all e11', sum(outs11)/len(outs11))
        self.log('test_acc_all e12', sum(outs12)/len(outs12))
        self.log('test_acc_all e13', sum(outs13)/len(outs13))
        self.log('test_acc_all inten', sum(outs14)/len(outs14))
        self.log('test_loss_all target', sum(outs15)/len(outs15))
        self.log('test_acc_all sarcasm', sum(outs16)/len(outs16))
        self.log('test_loss_all emo', sum(outs17)/len(outs17))
        self.log('test_f1_all sarcasm', sum(outs18)/len(outs18))

  """
  def predict_step(self, batch, batch_idx: int , dataloader_idx: int = None):
      name,lab,txt,img,val,arou,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12,e13,sarcasm,_,intensity,t1,t2,t3,t4,t5,t6,t7 = batch
      lab = batch[lab]
      txt = batch[txt]
      img = batch[img]
      with torch.no_grad():
        hidden_pred,_,_,_ = hm_model_memotion(txt.cuda(),img.cuda())
      logit_offen,logit_arou,logit_val, a,b,c,d,e,f,g,h,i,j,k,l,m,inten,logit_target,logit_sarcasm,logit_emotion = self(txt,img,hidden_pred)
      return logit_sarcasm
  """
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=5e-3)
    return optimizer


class HmDataModule(pl.LightningDataModule):

  def setup(self, stage):
    
      
    
    self.hm_train = t_p
    self.hm_val = v_p
    self.hm_test = te_p
    

  def train_dataloader(self):
    return DataLoader(self.hm_train, batch_size=64)

  def val_dataloader(self):
    return DataLoader(self.hm_val, batch_size=64)

  def test_dataloader(self):
    return DataLoader(self.hm_test, batch_size=128)

data_module = HmDataModule()


checkpoint_callback = ModelCheckpoint(
     monitor='val_f1_all sarcasm',
     dirpath='noemo/ckpts/',
     filename='our-ds-ckpt-epoch{epoch:02d}-val_f1_all sarcasm{val_f1_all sarcasm:.2f}',
     auto_insert_metric_name=False,
     save_top_k=1,
    mode="max",
 )
all_callbacks = []
all_callbacks.append(checkpoint_callback)

# train
from pytorch_lightning import seed_everything
seed_everything(123, workers=True)
hm_model = Classifier()
gpus = 1 if torch.cuda.is_available() else 0
trainer = pl.Trainer(gpus=gpus,max_epochs=60,callbacks=all_callbacks)
#trainer = pl.Trainer(gpus=gpus,deterministic=True,max_epochs=60,callbacks=all_callbacks)

trainer.fit(hm_model, data_module)






test_dataloader = DataLoader(dataset=te_p, batch_size=1478)

ckpt_path = '/content/noemo/ckpts/our-ds-ckpt-epoch19-val_f1_all sarcasm0.64.ckpt'

trainer.test(dataloaders=test_dataloader,ckpt_path=ckpt_path)

