import torch
from torch.nn.functional import softmax
import numpy as np
from os.path import isfile, join
from os import listdir
from PIL import Image
from matplotlib import pyplot as plt
from nltk import pos_tag

# загружаем inseption, чтобы можно было прогонять через него новые картинки, 
# получать их эмбеддинги и генерировать описания с помощью нашей сети
from beheaded_inception3 import beheaded_inception_v3
inception = beheaded_inception_v3().train(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GenerateCaptions:
    """
    # Класс для генерации описаний для нескольких случайных изображений с последующим выводом изображений и их описаний.
    #
    # model - обученная модель для генерации описаний.
    # vocab_itos - список слов из словаря; сеть предсказывает номера слов в этом списке, из которых строится описание.
    # model_with_attention - флаг, используется ли модель с attention (которая генерирует разные варианты описаний) или простая,
    # которая генерирует всего один вариант описания изображения.
    # path_to_test_images - адрес папки, в которой лежат изображения для тестирования генерации описаний
    # count_images - количество случайных изображений для которых сгенерируем описания
    #    
    """
    def __init__(self, model, vocab_itos, model_with_attention, 
                       path_to_test_images='D:\\DS\\NLP_Final_Project\\coco2017\\test2017\\', count_images=5):
        self.model = model
        self.vocab_itos = vocab_itos
        self.model_with_attention = model_with_attention
        self.path_to_test_images = path_to_test_images
        self.count_images = count_images
        
    def img_to_vectors(self, img):
        """
        # Переводит изображение в np, а затем в тензор, и получает векторное представления изображения с помощью inception3.
        # Аргументы:
        #    path_to_img - путь к изображению.
        # Возвращаемые значения:
        #    vectors_neck - векторное представление изображения. 
        #
        """
        img_np = np.array(img).astype('float32') / 255.

        assert isinstance(img_np, np.ndarray) and np.max(img_np) <= 1 and np.min(img_np) >=0 and img_np.shape[-1] == 3

        image = torch.tensor(img_np.transpose([2, 0, 1]), dtype=torch.float32)
        vectors_8x8, vectors_neck, logits = inception(image[None])
        return vectors_neck

    def generate_caption(self, image_vectors, init_caption_prefix=2, max_len=17):
        """
        # Генерирует описание изображения. 
        # 
        # Аргументы:
        #    image_vectors - - векторное представление изображения.
        #    init_caption_prefix - код токена начала предложения ("<start>") в словаре.
        #    max_len - максимальная длина сгенерированного предложения.
        # Возвращаемые значения:
        #    caption_prefix - числовое представления описания (список чисел, каждое из которых номер слова в self.vocab_itos). 
        #
        """
        with torch.no_grad():
            # image_vectors =  torch.Size([1, 2048])
            caption_prefix = []
            caption_prefix.append(init_caption_prefix)

            # слово за словом генерируем описание картинки
            for _ in range(max_len):

                # 1. представляем caption_prefix в виде матрицы
                caption_prefix_for_model = torch.LongTensor(caption_prefix).unsqueeze(0)
                # на первой итерации: caption_prefix_for_model.shape = [1, 1] = [batch_size, caption_prefix_len]

                # 2. Получить из RNN-ки логиты, передав ей vectors_neck и матрицу из п.1
                logits = self.model(image_vectors.to(device), caption_prefix_for_model.to(device))
                # logits = [1, 1, 10312] = [1, 1, vocabulary_size]

                # 3. Перевести логиты RNN-ки в вероятности (например, с помощью F.softmax)
                probabilities = softmax(logits.squeeze(0)[-1], 0)
                # probabilities = torch.Size([10312]) = = [vocabulary_size]

                # 4. сэмплировать следующее слово в описании, используя полученные вероятности. Можно сэмплировать жадно 
                # (тупо слово с самой большой вероятностью), можно сэмплировать из распределения
                predicted = probabilities.argmax(0)
                # predicted =  = torch.Size([1])

                # 5. Добавляем новое слово в caption_prefix
                caption_prefix.append(predicted.item())

                # 6. Если RNN-ка сгенерила символ конца предложения, останавливаемся
                if (self.vocab_itos[predicted.item()] == "<end>") | (self.vocab_itos[predicted.item()] == "."):
#                     print('caption_prefix = ', caption_prefix)
                    return caption_prefix
#             print('caption_prefix main return = ', caption_prefix)
            return caption_prefix

    def create_captions_for_img(self, img_vectors):            
        """
        # Создает множество из одного (для простой сети) или нескольких (для сети с attention) описаний изображения.
        # Аргументы:
        #    img_vectors -  векторное представление изображения. 
        # Возвращаемые значения:
        #    few_generated_captions - множество описаний изображения. 
        #
        """
        few_generated_captions = set()

        if self.model_with_attention:
            for i in range(10):
                sentence = []
                res = self.generate_caption(img_vectors)
                while res is None:
                    res = self.generate_caption(img_vectors)
                for encoded_word in res[1:-1]:
                    word = self.vocab_itos[encoded_word]
                    if (word not in sentence) & (word != '<unk>'):
                        sentence.append(word)     
                tokens_tag = pos_tag(sentence[1:])
                only_tokens = set(token for _, token in pos_tag(sentence[1:]))
                if (tokens_tag[-1][-1] not in ('IN', 'DT', 'CC')) and ('NN' in only_tokens):
                    few_generated_captions.add((' '.join(sentence[1:])))
        else:
            sentence = []
            res = self.generate_caption(img_vectors)
            for encoded_word in res[1:-1]:
                word = self.vocab_itos[encoded_word]
                if (word not in sentence) & (word != '<unk>'):
                    sentence.append(word)     
            few_generated_captions.add((' '.join(sentence[1:])))            
        return few_generated_captions            
            

    def show_imgs_and_captions(self):
        """
        # Главная функция класса. Выводит случайным образом выбранные изображения и сгенерированные для них описания
        #
        """
        # Строим список с адресами всех тестовых изображений
        images_paths = [self.path_to_test_images + f for f in listdir(self.path_to_test_images) \
                        if (isfile(join(self.path_to_test_images, f))) & (f.endswith(('jpg', 'png')))]
        
        # Получаем count_images случайных номеров изображений, для которых будем генерировать описания
        if self.count_images < 10:
            few_random_images = list(range(self.count_images))
        else:
            few_random_images = np.random.randint(low = 0, high = len(images_paths) - 1, size=self.count_images)
        for num, img_num in enumerate(few_random_images):
            img = Image.open(images_paths[img_num])
            img = img.resize((299, 299))
            # Получаем векторы выбранных изображений
            img_vectors = self.img_to_vectors(img) #images_paths[img_num]
            # Выводим изображения и их описания
            plt.figure(figsize=(100, 80))
            plt.subplot(self.count_images, 1, num+1)
            plt.title(self.create_captions_for_img(img_vectors), fontsize=20, loc='left')
            plt.axis('off')
            plt.imshow(img) 