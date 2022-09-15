import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CaptionNet(nn.Module):
    """
    # Класс для построения модели, которая генерирует описания изображений.
    #
    # embed_size - размер эмбеддинга.
    # hidden_size - размер вектора скрытого состояния.
    # vocab_size - длина словаря, на основании которого модель строит описания.
    # num_layers - количество слоев в LSTM.
    # cnn_feature_size - длина вектора изображения (полученного с помощью предобученной inception_v3)
    #    
    """
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, cnn_feature_size=2048):
        super(self.__class__, self).__init__()

        self.hidden_size = hidden_size        
        
        # стандартная архитектура такой сети такая: 

        # 1. линейные слои для преобразования эмбеддиинга картинки в начальные состояния h0 и c0 LSTM-ки
        self.img_embed = nn.Linear(cnn_feature_size, embed_size)
        
        # 2. слой эмбеддинга
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # 3. несколько LSTM слоев (для начала не берите больше двух, чтобы долго не ждать)
        self.lstm1 = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        # 4. линейный слой для получения логитов
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, image_vectors, captions_ix):
        """
        # Обучение сети (train-режим).
        # Аргументы:
        #    image_vectors - векторы изображений; image_vectors =  torch.Size([32, 2048]) = [batch_size, cnn_feature_size]
        #    captions_ix - числовые представления описаний; captions_ix =  torch.Size([32, 17]) = [batch_size, sen_len]
        # Возвращаемые значения:
        #    outputs - логиты для последующего вычисления вероятностей, на основании которых будем выбирать слова для описания. 
        #
        """      
        # image_vectors =  torch.Size([32, 2048]) = [batch_size, embed_size]
        # captions_ix =  torch.Size([32, 17]) = [batch_size, sen_len]
        
        # 1. инициализируем LSTM state
        batch_size = image_vectors.shape[0] # features is of shape (batch_size, embed_size)
        h, c = (torch.zeros((1, batch_size, self.hidden_size), device=device), \
                       torch.zeros((1, batch_size, self.hidden_size), device=device))                    
        # h, c =  torch.Size([1, 32, 256]) = [1, batch_size, hidden_size]

        # 2. применим слой эмбеддингов к image_vectors
        img_embeddings = self.img_embed(image_vectors)  # (batch_size, embed_size)

        # Убираем токен <end>, чтобы избежать прогнозирования, когда <end> подается на вход LSTM
        captions_ix = captions_ix[:, :-1] 
        # captions_ix = torch.Size([32, 16]) = [batch_size, sen_len - 1]        
        caption_embeddings = self.embed(captions_ix)
        
        embeddings = torch.cat((img_embeddings.unsqueeze(1), caption_embeddings), dim=1) 
        # embeddings =  torch.Size([32, 17, 256]) = [batch_size, sen_len, embed_size]

        # 3. скормим LSTM captions_emb
        lstm_out, (h, c) = self.lstm1(embeddings, (h, c))
        # lstm_out =  torch.Size([32, 17, 256]) = [batch_size, sent len, hidden_size]      
        
        # 4. посчитаем логиты из выхода LSTM
        logits = self.linear(lstm_out)
        # logits =  torch.Size([32, 17, 10312]) = [batch_size, sent len, vocab_size]
        
        return logits       

class Attention(nn.Module):
    """ 
    # Класс, реализующий механизм Багданов attention (https://arxiv.org/pdf/1409.0473.pdf).
    # cnn_feature_size - длина вектора изображения (полученного с помощью предобученной inception_v3)
    # hidden_size - размер вектора скрытого состояния.
    # output_size - разрность вектора attention score.
    #
    """
    def __init__(self, cnn_feature_size, hidden_size, output_size = 1):
        super(Attention, self).__init__()
        self.cnn_feature_size = cnn_feature_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Линейный слой для преобразования image_vectors
        self.W_a = nn.Linear(self.cnn_feature_size, self.hidden_size)
        # Линейный слой для преобразования hidden state (h) - вектора скрытого состояния из LSTM.
        self.U_a = nn.Linear(self.hidden_size, self.hidden_size)
        # Линейный слой для получения скоров, на основании которых будут расчитаны веса внимания.
        self.v_a = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, image_vectors, h):
        """
        # Обучение сети (train-режим).
        # Аргументы:
        #    image_vectors - векторы изображений; image_vectors =  torch.Size([32, 2048]) = [batch_size, cnn_feature_size]
        #    h - финальный вектор скрытого состояния из LSTM.
        # Возвращаемые значения:
        #    context - линейная комбинация image_vectors, взвешенных по atten_weight, указывающая сети, 
        #    каким частям изображения следует уделить больше внимания.
        #       
        """
        # добавляем дополнительную размерность в h
        h = h.unsqueeze(0)
        atten_1 = self.W_a(image_vectors)
        atten_2 = self.U_a(h)
        # применяем тангенс к сумме результатов работы полносвязных слоев (и atten_1, и atten_2 имеют размерность hidden_size)
        atten_tan = torch.tanh(atten_1+atten_2)
        # каждый скор соответствует одному выходу image_vectors 
        atten_score = self.v_a(atten_tan)        
        atten_weight = F.softmax(atten_score, dim = 1)
        # сначала умножаем каждый вектор из image_vectors на соответствующий softmax score, 
        # а затем суммируем эти векторы чтобы получить context вектор
        context = torch.sum(atten_weight * image_vectors, dim = 1)
        return context

class CaptionNetWithAttantion2LSTM(nn.Module):
    """
    # Класс для построения модели, которая генерирует описания изображений.
    #
    # embed_size - размер эмбеддинга.
    # hidden_size - размер вектора скрытого состояния.
    # vocab_size - длина словаря, на основании которого модель строит описания.
    # num_layers - количество слоев в LSTM.
    # cnn_feature_size - длина вектора изображения (полученного с помощью предобученной inception_v3)
    #    
    """
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, cnn_feature_size=2048):
        super(self.__class__, self).__init__()
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.sample_temp = 0.5
        
        # 1. линейные слои для преобразования эмбеддиинга картинки в начальные состояния h0 и c0 LSTM-ки
        self.img_embed = nn.Linear(cnn_feature_size, hidden_size)        
                
        # 2. слой эмбедднга
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # 3. несколько LSTM слоев (для начала не берите больше двух, чтобы долго не ждать)
        self.lstm1 = nn.LSTM(embed_size + cnn_feature_size, hidden_size, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        
        # 4. линейный слой для получения логитов
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        # 5. слой attention
        self.attention = Attention(cnn_feature_size, hidden_size)
        
        # 6. слой dropout
        self.drop = nn.Dropout(p=0.5)
        
    def forward(self, image_vectors, captions_ix):
        """
        # Обучение сети (train-режим).
        # Аргументы:
        #    image_vectors - векторы изображений; image_vectors =  torch.Size([32, 2048]) = [batch_size, cnn_feature_size]
        #    captions_ix - числовые представления описаний; captions_ix =  torch.Size([32, 17]) = [batch_size, sen_len]
        # Возвращаемые значения:
        #    outputs - логиты для последующего вычисления вероятностей, на основании которых будем выбирать слова для описания. 
        #
        """      
        # 1. инициализируем LSTM state
        img_embeddings = self.img_embed(image_vectors)  
        # img_embeddings = torch.Size([32, 256]) = [batch_size, hidden_size]
        
        h, c = (img_embeddings.unsqueeze(0), img_embeddings.unsqueeze(0))
        # h, c =  torch.Size([1, 32, 256]) = [1, batch_size, hidden_size]
        
         # 2. применим слой эмбеддингов к captions_ix
        caption_embeddings = self.embed(captions_ix)
        # embeddings =  torch.Size([32, 17, 256]) = [batch_size, sen_len, embed_size]
        
        seq_len = captions_ix.size(1)
        batch_size = img_embeddings.size(0)
        
        # инициализируем тензор, в который будем записывать output из LSTM
        outputs = torch.zeros(batch_size, seq_len, self.vocab_size).to(device)
        # outputs =  torch.Size([32, 17, 10312]) = [batch_size, sen_len, vocab_size]
        
        # Создаем логиты для каждого токена из seq_len кроме первого токена(<start>).
        for t in range(seq_len):
            sample_prob = 0.0 if t == 0 else 0.5
            use_sampling = np.random.random() < sample_prob
            if use_sampling == False:
                word_embed = caption_embeddings[:,t,:]
            # word_embed = torch.Size([32, 256]) = [batch_size, embed_size]
            
            context = self.attention(image_vectors, h)
            # context = torch.Size([1, 32, 2048]) = [1, batch_size, cnn_feature_size]
            
            if len(word_embed.shape) == 3:
                word_embed = word_embed.squeeze(1)
            input_concat = torch.cat([word_embed, context.squeeze(0)], 1)
            # input_concat = torch.Size([32, 2304]) = [batch_size, embed_size + cnn_feature_size]
            
            lstm_out, (h,c) = self.lstm1(input_concat.unsqueeze(1), (h,c))
            lstm_out, (h,c) = self.lstm2(lstm_out, (h,c))
            # h, c =  torch.Size([1, 32, 256]) = [1, batch_size, hidden_size]
            lstm_out = self.drop(lstm_out)
            # lstm_out =  torch.Size([32, 1, 256]) = [batch_size, 1, hidden_size]
            output = self.linear(lstm_out)
            # output =  torch.Size([32, 1, 10312]) = [batch_size, 1, vocab_size]
            
            if use_sampling == True:
                # используем температуру для усиления значений перед применением softmax
                scaled_output = output / self.sample_temp
                scoring = F.log_softmax(scaled_output, dim=1)
                top_idx = scoring.topk(1)[1]
                word_embed = self.embed(top_idx).squeeze(1)
                # output =  torch.Size([32, 1, 10312]) = [batch_size, 1, vocab_size]
                # outputs[:, t, :] =  torch.Size([32, 10312]) = [batch_size, vocab_size]
            outputs[:, t, :] = output.squeeze(1)
        # outputs =  torch.Size([32, 17, 10312]) = [batch_size, sen_len, vocab_size]
        return outputs    