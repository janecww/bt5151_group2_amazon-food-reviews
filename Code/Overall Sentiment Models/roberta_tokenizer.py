from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")


class tokenize_text_with_labels(Dataset):
    def __init__(self, df, maxlen):
        self.df = df
        # A reset reindexes from 1 to len(df), the shuffled df frames are sparse.
        self.df.reset_index(drop=True, inplace=True)
        self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.maxlen = maxlen

    def __len__(self):
        return(len(self.df))

    def __getitem__(self, index):
        review = self.df.loc[index, 'Summary_Text']
        label = int(self.df.loc[index, 'Score']) + 1

        # Tokenize text
        tokenized_text = self.tokenizer(
            review,
            truncation=True,
            padding='max_length',
            max_length=self.maxlen,
            return_tensors="pt"
        )

        # Extract input_ids and attention_mask
        input_ids = tokenized_text['input_ids'].squeeze(0)
        attention_mask = tokenized_text['attention_mask'].squeeze(0)

        # Return as dictionary
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label)
        }
