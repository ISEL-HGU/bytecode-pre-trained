from transformers import PreTrainedTokenizer
from .byte_tok_runner import tokenize_class_file_with_gradle


class CustomTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_file, **kwargs):
        super().__init__(**kwargs)
        
        
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.vocab = [line.strip() for line in f]
        
        
        self.special_tokens = {
            'pad_token': '<pad>',
            'unk_token': '<unk>',
            'bos_token': '<bos>',
            'eos_token': '<eos>',
            'mask_token': '<mask>',
        }
        
        
        for token in self.special_tokens.values():
            if token not in self.vocab:
                self.vocab.append(token)
        
        
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}
        
        
        self.pad_token = self.special_tokens['pad_token']
        self.pad_token_id = self.token_to_id[self.pad_token]
        
        self.unk_token = self.special_tokens['unk_token']
        self.unk_token_id = self.token_to_id[self.unk_token]
        
        self.bos_token = self.special_tokens['bos_token']
        self.bos_token_id = self.token_to_id[self.bos_token]
        
        self.eos_token = self.special_tokens['eos_token']
        self.eos_token_id = self.token_to_id[self.eos_token]
        
        self.mask_token = self.special_tokens['mask_token']
        self.mask_token_id = self.token_to_id[self.mask_token]
        
        self.option = ['t']  
        self.unk_token_id_direct = self.token_to_id.get(self.unk_token, None)

    def tokenize_file(self, project_dir, file_path):
        
        tokens = tokenize_class_file_with_gradle(project_dir, file_path, self.option)
        return tokens

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, list):
            return [self.convert_tokens_to_ids(token) for token in tokens]
        else:
            return self.token_to_id.get(tokens, self.unk_token_id_direct)

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, list):
            return [self.convert_ids_to_tokens(id_) for id_ in ids]
        else:
            return self.id_to_token.get(ids, self.unk_token)

    def get_vocab(self):
        return self.token_to_id
