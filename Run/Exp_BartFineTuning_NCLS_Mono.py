import os
import json
import tqdm
import torch
from Tools import ProgressBar, SaveModel
from Loader_NCLS import NCLS_Loader
from transformers import BartForConditionalGeneration, BartTokenizer

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
batch_size = 3
learning_rate = 1E-5
episode_number = 10

if __name__ == '__main__':
    model = BartForConditionalGeneration.from_pretrained('C:/ProjectData/bart-large-cnn')
    tokenizer = BartTokenizer.from_pretrained('C:/ProjectData/bart-large-cnn')

    train_data = NCLS_Loader(sample_number=100000)
    save_path = 'E:/ProjectData/NCLS/bart-finetuning-640Length-100K/'
    if not os.path.exists(save_path): os.makedirs(save_path)

    model = model.cuda()

    if torch.cuda.device_count() > 1:  # 判断是不是有多个GPU
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    times = 0
    pbar = ProgressBar(n_total=episode_number * len(train_data))
    episode_loss = 0.0
    for episode_index in range(episode_number):
        for batch_index in range(0, len(train_data), batch_size):
            times += 1

            batch_article = [_['Article'] for _ in train_data[batch_index:batch_index + batch_size]]
            batch_summary = [_['Summary'] for _ in train_data[batch_index:batch_index + batch_size]]
            batch_article = tokenizer.batch_encode_plus(
                batch_article, return_tensors='pt', max_length=640, truncation=True, padding=True)
            batch_label = tokenizer.batch_encode_plus(
                batch_summary, return_tensors='pt', max_length=640, truncation=True, padding=True)

            # article_ids, article_label = batch_article['input_ids'], batch_article['mlm_label']
            loss = model(input_ids=batch_article['input_ids'].cuda(), labels=batch_label['input_ids'].cuda())[0]

            if torch.cuda.device_count() > 1: loss = torch.mean(loss)
            pbar(episode_index * len(train_data) + batch_index, {'loss': loss.item()})
            loss.backward()
            optimizer.step()
            model.zero_grad()

            episode_loss += loss.item()
            if times % 1000 == 999:
                print('\nTotal 1000 Loss =', episode_loss)
                episode_loss = 0.0
                SaveModel(model,
                          save_path + '%08d-Parameter/' % (episode_index * len(train_data) + batch_index))
                torch.save(
                    {'epoch': episode_index, 'optimizer': optimizer.state_dict()},
                    os.path.join(save_path, '%08d-Optimizer.pkl' % (episode_index * len(train_data) + batch_index)))
