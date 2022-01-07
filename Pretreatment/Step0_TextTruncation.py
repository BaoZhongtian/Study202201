import os
import json
import tqdm
from transformers import BertTokenizer

cnn_path = 'D:/ProjectData/CNNDM/cnn/stories'
dm_path = 'D:/ProjectData/CNNDM/dailymail_stories/dailymail/stories'
save_path = 'D:/ProjectData/CNNDM/Step0_TextTruncation_test'
if not os.path.exists(save_path): os.makedirs(save_path)

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print('Tokenizer Load Completed')

    train_data = json.load(open('D:/ProjectData/CNNDM/CNNDM_test.json', 'r'))
    for treat_sample in tqdm.tqdm(train_data):
        filename = treat_sample['filename'] + '.story'
        if os.path.exists(os.path.join(cnn_path, filename)):
            with open(os.path.join(cnn_path, filename), 'r', encoding='UTF-8') as file:
                raw_data = file.readlines()
        else:
            with open(os.path.join(dm_path, filename), 'r', encoding='UTF-8') as file:
                raw_data = file.readlines()

        with open(os.path.join(save_path, filename.replace('story', 'txt')), 'w', encoding='UTF-8')as file:
            current_token_len = 0
            for sample in raw_data:
                sample = sample.replace('\n', '').lower()
                if sample == '': continue
                if sample[0] == '@': break
                tokens = tokenizer.encode(sample, add_special_tokens=False)
                current_token_len += len(tokens)
                if current_token_len >= 510: break
                file.write(tokenizer.decode(tokens) + '\n')

        # exit()
