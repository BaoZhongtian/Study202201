import os
import json
import tqdm
import torch
from Tools import ProgressBar, SaveModel
from Loader_NCLS import NCLS_Loader
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import EncoderDecoderModel, BertGenerationEncoder, BertGenerationDecoder, BertTokenizer
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
batch_size = 1
learning_rate = 1E-5
episode_number = 10

if __name__ == '__main__':
    # model = BartForConditionalGeneration.from_pretrained('C:/PythonProject/bart-large')
    # tokenizer = BartTokenizer.from_pretrained('C:/PythonProject/bart-large')
    # max_length = 1000

    # encoder = BertGenerationEncoder.from_pretrained(
    #     "C:/PythonProject/bert-base-uncased", bos_token_id=101, eos_token_id=102)
    # decoder = BertGenerationDecoder.from_pretrained(
    #     "C:/PythonProject/bert-base-uncased", add_cross_attention=True, is_decoder=True, bos_token_id=101,
    #     eos_token_id=102)
    # model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
    # model.config.decoder_start_token_id = 101
    # model.config.pad_token_id = 0
    # tokenizer = BertTokenizer.from_pretrained('C:/PythonProject/bert-base-uncased')

    model = MT5ForConditionalGeneration.from_pretrained('C:/PythonProject/mt5-base')
    tokenizer = MT5Tokenizer.from_pretrained('C:/PythonProject/mt5-base')
    max_length = 640

    train_data = NCLS_Loader(sample_number=100000)
    save_path = 'E:/ProjectData/NCLS/MT5-finetuning-%dLength-100K/' % max_length
    if not os.path.exists(save_path): os.makedirs(save_path)
    model = model.cuda()

    if torch.cuda.device_count() > 1:  # 判断是不是有多个GPU
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    times = 0
    pbar = ProgressBar(n_total=episode_number * len(train_data))
    episode_loss = 0.0
    for episode_index in range(episode_number):
        for batch_index in range(0, len(train_data), batch_size):
            times += 1

            batch_article = [_['Article'] for _ in train_data[batch_index:batch_index + batch_size]]
            batch_summary = [_['Summary'] for _ in train_data[batch_index:batch_index + batch_size]]
            batch_article = tokenizer.batch_encode_plus(
                batch_article, return_tensors='pt', max_length=max_length, truncation=True, padding=True)
            batch_summary = tokenizer.batch_encode_plus(
                batch_summary, return_tensors='pt', max_length=max_length, truncation=True, padding=True)

            # article_ids, article_label = batch_article['input_ids'], batch_article['mlm_label']
            loss = model(
                input_ids=batch_article['input_ids'].cuda(), attention_mask=batch_article['attention_mask'].cuda(),
                labels=batch_summary['input_ids'].cuda(), return_dict=True).loss

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
