import torchtext
from torchtext.data import Field

class Vocabulary:
    """
    # Класс для создания словаря на основе списка из текстовых описаний изображений captions.
    #
    # tokenizer - токенайзер для разбиения предложения на токены (слова, знаки препинания и т.д.)
    # freq_threshold - минимальная частота встречания слова в корпусе captions. Если слово встречается меньше 
    #                  freq_threshold раз, то оно не включается в словарь, и при переводе текстовых токенов 
    #                  в числа заменяется на код токена "<unk>".
    #    
    """
    def __init__(self, tokenizer, freq_threshold=5):
        self.freq_threshold = freq_threshold       
        self.tokenizer = tokenizer
        # Словарь, создаваемый на основе captions
        self.vocab = {}
        self.all_lenght_captions = []

    def __len__(self):
        """Возвращает длину словаря"""
        return len(self.vocab.stoi)

    def build_vocabulary(self, captions):
        """Строит словарь на основе предложений из captions"""
        text_field = Field(
                            tokenize='basic_english', 
                            init_token = '<start>', 
                            eos_token = '<end>', 
                            fix_length=100,
                            lower=True
                          )
        preprocessed_text = []
        for five_captions in captions:
            for caption in five_captions:
                try:
                    preprocessed_text.append(text_field.preprocess(caption))
                    len_caption = len(self.tokenizer.tokenize(caption))
                    self.all_lenght_captions.append(len_caption)
                except:
                    print('five_captions =', five_captions, 'caption =', caption)
        text_field.build_vocab(preprocessed_text, min_freq=self.freq_threshold)
        self.vocab = text_field.vocab

    def numericalize(self, text):
        """Переводит текстовое предложение в его числовое представление."""
        
        tokenized_text = self.tokenizer.tokenize(text)
        result = [self.vocab.stoi[token] if token in self.vocab.stoi else self.vocab.stoi["<unk>"] for token in tokenized_text]
        
        return result