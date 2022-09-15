import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from modules.dataset import CocoDataset

class CocoDataloader:
    """
    # Класс для создания dataloaders из датасета, с использованием torch.utils.data.DataLoader.
    #
    # img_vectors - векторы изображений, полученные с помощью Inception-v3.
    # captions - описания изображений; каждому изображению соответствует несколько описаний. При обучении сети 
    #            для каждого изображения будем использовать одно случайным образом выбранное описание.
    # freq_threshold - минимальная частота встречания слова в корпусе captions. Если слово встречается меньше 
    #                  freq_threshold раз, то оно не включается в словарь, и при переводе текстовых токенов 
    #                  в числа заменяется на код токена "<unk>".
    # batch_size - размер батча.
    # num_workers - количество подпроцессов для загрузки данных. 0 означает, что данные будут загружены в основном процессе.
    # shuffle - флаг, перетасовывать ли данные каждую эпоху.
    # pin_memory - флаг, использовать ли pin_memory.
    #    
    """

    def __init__(self, img_vectors, captions, freq_threshold=5, max_len_caption=15, 
                       batch_size=32, num_workers=0, shuffle=True, pin_memory=True):
        self.img_vectors = img_vectors 
        self.captions = captions 
        self.freq_threshold = freq_threshold
        self.max_len_caption = max_len_caption
        # Доля данных, которые будут использованы для обучения сети; остальные используем для валидации.
        self.train_part = 0.9
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory

    def collate_fn(self, batch):
        """
        # Оъединяет список экземпляров датасета, чтобы сформировать мини-батч тензоров.
        # Аргументы:
        #    batch - батч данных.
        # Возвращаемые значения: 
        #    словарь из тензоров изображений ("images"), 
        #               тензоров соответствующих изображениям описаний("captions"), 
        #
        """
        images = []
        captions_ = []
        for elem in batch:
            images.append(elem[0])
            captions_.append(elem[1])

        images = torch.stack(images)
        captions_ = torch.stack(captions_)

        return {"images": images, 
                "captions": captions_}

    def split_dataset(self, train_part):
        """
        # Создает датасет с помощью класса CocoDataset и делит его на train и validation части.
        # Аргументы:
        #    train_part - доля данных, которые будут использованы для обучения сети; остальные используем для валидации.
        # Возвращаемые значения: 
        #    train_data - тренировочный датасет.
        #    valid_data - валидационный датасет.
        #
        """
        dataset = CocoDataset(self.img_vectors, self.captions, self.freq_threshold, self.max_len_caption)
        train_size, val_size = int(train_part * len(dataset)), len(dataset) - int(train_part * len(dataset))
        train_data, valid_data = random_split(dataset, [train_size, val_size])

        print(f"Количество экземпляров в тренировочном датасете: {len(train_data)}")
        print(f"Количество экземпляров в валидационном датасете: {len(valid_data)}")
        
        return train_data, valid_data

    def get_dataloaders(self):
        """
        # Создает загрузчики данных (dataloaders).
        # Возвращаемые значения: 
        #    train_loader - dataloader для обучения сети.
        #    valid_loader - dataloader для валидации сети.
        #
        """
        train_data, valid_data = self.split_dataset(self.train_part)
        train_loader = DataLoader(train_data, 
                                  batch_size=self.batch_size, 
                                  num_workers=self.num_workers, 
                                  shuffle=self.shuffle, 
                                  pin_memory=self.pin_memory, 
                                  collate_fn=self.collate_fn)
        valid_loader = DataLoader(valid_data, 
                                  batch_size=self.batch_size, 
                                  num_workers=self.num_workers, 
                                  shuffle=self.shuffle, 
                                  pin_memory=self.pin_memory, 
                                  collate_fn=self.collate_fn)
        return train_loader, valid_loader