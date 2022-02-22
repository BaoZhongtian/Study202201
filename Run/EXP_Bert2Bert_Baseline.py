import os
import numpy
import torch
import transformers
from Tools import ProgressBar, SaveModel
from Loader_NCLS import NCLS_Loader

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

num_train_epochs = 100
batch_size = 1

if __name__ == '__main__':
    config = transformers.EncoderDecoderConfig.from_encoder_decoder_configs(
        transformers.BertConfig.from_pretrained('C:/PythonProject/bert-base-multilingual-cased'),
        transformers.BertConfig.from_pretrained('C:/PythonProject/bert-base-multilingual-cased'))
    model = transformers.EncoderDecoderModel(config)
    tokenizer = transformers.BertTokenizer.from_pretrained('C:/PythonProject/bert-base-multilingual-cased')

    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # model.config.decoder_start_token_id = model.config.decoder.pad_token_id
    # model.config.pad_token_id = model.config.decoder.pad_token_id
    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1E-4)

    save_path = 'E:/ProjectData/NCLS/EN2ZH-Summarization/Bert2Bert-AllParameter-1K-Another/'
    if not os.path.exists(save_path): os.makedirs(save_path)
    train_data = NCLS_Loader(sample_number=1000)
    pbar = ProgressBar(n_total=num_train_epochs * len(train_data))

    total_loss, train_counter = 0.0, 0
    for epoch in range(num_train_epochs):
        model.train()
        for batch_index in range(0, len(train_data), batch_size):
            model_inputs = tokenizer([_['Article'] for _ in train_data[batch_index:batch_index + batch_size]],
                                     max_length=512, truncation=True, return_tensors='pt')
            # labels = tokenizer([_['Summary'] for _ in train_data[batch_index:batch_index + batch_size]],
            #                    max_length=512, padding="max_length", truncation=True)['input_ids']

            with tokenizer.as_target_tokenizer():
                labels = tokenizer([_['CrossLingualSummary'] for _ in train_data[batch_index:batch_index + batch_size]],
                                   max_length=512, truncation=True, return_tensors='pt')['input_ids']
            # for indexX in range(len(labels)):
            #     for indexY in range(len(labels[indexX])):
            #         if labels[indexX][indexY] == 0: labels[indexX][indexY] = -100

            outputs = model(input_ids=model_inputs['input_ids'].cuda(), labels=labels.cuda())
            loss = outputs.loss
            loss.backward()
            pbar(epoch * len(train_data) + batch_index, {'loss': loss.item()})
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            train_counter += 1
            if train_counter % 1000 == 999:
                print('\nCurrent 1000 Total Loss = %f' % total_loss)
                total_loss = 0.0
                SaveModel(model,
                          save_path + '%08d-Parameter/' % (epoch * len(train_data) + batch_index))
