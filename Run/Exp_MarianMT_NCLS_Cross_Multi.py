import os
import numpy
import torch
import transformers
from Tools import ProgressBar, SaveModel
from Loader_NCLS import NCLS_Loader

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

num_train_epochs = 20
batch_size = 6

if __name__ == '__main__':
    model = transformers.MarianMTModel.from_pretrained("C:/PythonProject/opus-mt-en-zh")
    tokenizer = transformers.MarianTokenizer.from_pretrained("C:/PythonProject/opus-mt-en-zh")

    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5E-5)

    save_path = 'E:/ProjectData/NCLS/EN2ZH-Summarization/MarianMT-All-Multi-Lower/'
    if not os.path.exists(save_path): os.makedirs(save_path)
    train_data = NCLS_Loader(sample_number=900000)
    pbar = ProgressBar(n_total=num_train_epochs * len(train_data))

    total_loss, train_counter = 0.0, 0
    for epoch in range(num_train_epochs):
        model.train()
        for batch_index in range(0, len(train_data), batch_size):
            model_inputs = tokenizer([_['Article'].lower() for _ in train_data[batch_index:batch_index + batch_size]],
                                     max_length=512, padding='max_length', truncation=True, return_tensors='pt')
            # labels = tokenizer([_['Summary'] for _ in train_data[batch_index:batch_index + batch_size]],
            #                    max_length=512, padding="max_length", truncation=True)['input_ids']

            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    [_['CrossLingualSummary'] for _ in train_data[batch_index:batch_index + batch_size]],
                    max_length=512, padding='max_length', truncation=True, return_tensors='pt')['input_ids']
            for indexX in range(len(labels)):
                for indexY in range(len(labels[indexX])):
                    if labels[indexX][indexY] == tokenizer.pad_token_id: labels[indexX][indexY] = -100

            outputs = model(input_ids=model_inputs['input_ids'].cuda(),
                            attention_mask=model_inputs['attention_mask'].cuda(), labels=labels.cuda())
            loss = outputs.loss
            loss.backward()
            pbar(epoch * len(train_data) + batch_index, {'loss': loss.item()})
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            train_counter += 1
            if train_counter % 10000 == 9999:
                print('\nCurrent 1000 Total Loss = %f' % total_loss)
                total_loss = 0.0
                SaveModel(model,
                          save_path + '%08d-Parameter/' % (epoch * len(train_data) + batch_index))
