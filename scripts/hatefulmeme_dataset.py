import torch
import pandas as pd
import pandas_path
from pandas_path import path
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from bert_serving.client import BertClient
bc = BertClient()
from transformers import AutoTokenizer, pipeline, AutoModelForMaskedLM
tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
model = AutoModelForMaskedLM.from_pretrained("bert-base-german-cased")
pipe = pipeline('feature-extraction', model=model,
                tokenizer=tokenizer)


def get_german_bert_vector(data_path):
    if os.path.exists(os.path.basename(data_path)+'.german_bert_vec.pkl'):
        df = pd.read_pickle(os.path.basename(data_path)+'.german_bert_vec.pkl')
    else:
        df = pd.DataFrame(columns=['id', 'encode_text'])
        samples_frame = pd.read_json(
            data_path, lines=True)
        i = 0
        for text in tqdm(samples_frame['text']):
            df = df.append({'id': samples_frame['id'][i], 'encode_text' :pipe(text, pad_to_max_length=True)[0][0]}, ignore_index=True)
            i += 1
        pd.to_pickle(df, os.path.basename(data_path)+'.german_bert_vec.pkl')
    return df



def get_bert_vector(data_path):
    if os.path.exists(os.path.basename(data_path)+'.multi_bert_vec.pkl'):
        df = pd.read_pickle(os.path.basename(data_path)+'.multi_bert_vec.pkl')
    else:
        df = pd.DataFrame(columns=['id', 'encode_text'])
        samples_frame = pd.read_json(
            data_path, lines=True)
        for i, text in enumerate(samples_frame['text']):
            df = df.append({'id': samples_frame['id'][i], 'encode_text' : bc.encode([text])[0]}, ignore_index=True)
        pd.to_pickle(df, os.path.basename(data_path)+'.multi_bert_vec.pkl')
    return df



class HatefulMemesDataset(torch.utils.data.Dataset):
    """Uses jsonl data to preprocess and serve 
    dictionary of multimodal tensors for model input.
    """

    def __init__(
        self,
        data_path,
        img_dir,
        image_transform,
        text_transform,
        bert_transform, 
        balance=False,
        dev_limit=None,
        random_state=0,
    ):

        self.samples_frame = pd.read_json(
            data_path, lines=True
        )
        #self.encoded_output = get_bert_vector(data_path)
        self.encoded_output = get_german_bert_vector(data_path)
        self.dev_limit = dev_limit
        if balance:
            neg = self.samples_frame[
                self.samples_frame.label.eq(0)
            ]
            pos = self.samples_frame[
                self.samples_frame.label.eq(1)
            ]
            self.samples_frame = pd.concat(
                [
                    neg.sample(
                        pos.shape[0], 
                        random_state=random_state
                    ), 
                    pos
                ]
            )
        if self.dev_limit:
            if self.samples_frame.shape[0] > self.dev_limit:
                self.samples_frame = self.samples_frame.sample(
                    dev_limit, random_state=random_state
                )
        self.samples_frame = self.samples_frame.reset_index(
            drop=True
        )
        self.samples_frame.img = self.samples_frame.apply(
            lambda row: (img_dir / row.img), axis=1
        )

        # https://github.com/drivendataorg/pandas-path
        if not self.samples_frame.img.path.exists().all():
            raise FileNotFoundError
        if not self.samples_frame.img.path.is_file().all():
            raise TypeError
            
        self.image_transform = image_transform
        self.text_transform = text_transform
        self.bert_transform = bert_transform

    def __len__(self):
        """This method is called when you do len(instance) 
        for an instance of this class.
        """
        return len(self.samples_frame)

    def __getitem__(self, idx):
        """This method is called when you do instance[key] 
        for an instance of this class.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        text = torch.Tensor(list(self.encoded_output[self.encoded_output['id']==self.samples_frame.loc[idx, "id"]]['encode_text'].values)[0][:768]).squeeze()
        img_id = self.samples_frame.loc[idx, "id"]

        image = Image.open(
            self.samples_frame.loc[idx, "img"]
        ).convert("RGB")
        image = self.image_transform(image)

        '''
        text = torch.Tensor(
            self.text_transform.get_sentence_vector(
                self.samples_frame.loc[idx, "text"]
            )
        ).squeeze()
        
        text = torch.Tensor(
                bc.encode([self.samples_frame.loc[idx, "text"]])[0]).squeeze()
        '''


        if "label" in self.samples_frame.columns:
            label = torch.Tensor(
                [self.samples_frame.loc[idx, "label"]]
            ).long().squeeze()
            sample = {
                "id": img_id, 
                "image": image, 
                "text": text, 
                "label": label
            }
        else:
            sample = {
                "id": img_id, 
                "image": image, 
                "text": text
            }

        return sample

