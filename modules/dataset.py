import torch
from torch.utils.data import Dataset
import nltk
import random
from modules.vocabulary import Vocabulary

class CocoDataset(Dataset):
    """
    # Класс для создания датасета, наследуемый от torch.utils.data.Dataset.
    #
    # img_vectors - векторы изображений, полученные с помощью Inception-v3.
    # captions - описания изображений; каждому изображению соответствует несколько описаний. При обучении сети 
                 для каждого изображения будем использовать одно случайным образом выбранное описание.
    # freq_threshold - минимальная частота встречания слова в корпусе captions. Если слово встречается меньше 
    #                  freq_threshold раз, то оно не включается в словарь, и при переводе текстовых токенов 
    #                  в числа заменяется на код токена "<unk>".
    # max_len_caption - максимальная длина предложения, под которую подгоняются все описания, используемые при обучении сети.
    #    
    """
    def __init__(self, img_vectors, captions, freq_threshold=5, max_len_caption=15):

        self.imgs = img_vectors
        self.captions = captions 
        self.max_len_caption = max_len_caption

        # Инициализируем и строим словарь
        tokenizer = nltk.WordPunctTokenizer()
        self.vocab = Vocabulary(tokenizer, freq_threshold)
        self.vocab.build_vocabulary(self.captions)

    def __len__(self):
        """
        # Возвращаемые значения: 
        #    количество наборов captions (равное количеству изображений).
        #
        """
        return len(self.captions)   #(self.df)

    def __getitem__(self, index):
        """
        # Извлекает из датасета вектор изображения и соответствующее изображению предобработанное описание. 
        # Предобработка описания включает:
        # 1) дополнение токенами начала и конца предложения ("<start>" и "<end>"),
        # 2) приведение к длине max_len_caption + 2, дополняяя паддингами либо обрезая,
        # 3) перевод описания в числовой вид.
        # Аргументы:
        #    index - номер изображения в датасете
        # Возвращаемые значения:
        #    тензор из вектора изображения и тензор из описания изображения 
        #
        """
        # Случайным образом выбираем одно из описаний, соответствующих изображению
        caption = self.captions[index][random.randint(0, len(self.captions[index]) - 1)]
        img_vector = self.imgs[index]
        
        # Инициализируем описание числовым представлением токена "<start>"
        numericalized_caption = [self.vocab.vocab.stoi["<start>"]]
        
        # Добавляем к описанию числовое представление самого предложения
        numericalized_caption += self.vocab.numericalize(caption)       
        len_numericalized_caption = len(numericalized_caption) - 1
        
        # Если длина описания меньше self.max_len_caption, дополняем числовыми представлениями токена "<pad>"
        if len_numericalized_caption < self.max_len_caption:
            numericalized_caption.extend([self.vocab.vocab.stoi["<pad>"]] * (self.max_len_caption-len_numericalized_caption))
        # Если длина описания больше self.max_len_caption, обрезаем описание 
        elif len_numericalized_caption > self.max_len_caption:
            numericalized_caption = numericalized_caption[:self.max_len_caption + 1]
            
        # Добавляем к описанию числовое представление токена "<end>"        
        numericalized_caption.append(self.vocab.vocab.stoi["<end>"])

        return torch.tensor(img_vector), torch.tensor(numericalized_caption)