import os
import json
import numpy
import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if __name__ == '__main__':
    model = BartForConditionalGeneration.from_pretrained('C:/ProjectData/bart-large-cnn')
    tokenizer = BartTokenizer.from_pretrained('C:/ProjectData/bart-large-cnn')

    train_data = json.load(open('../Pretreatment/CNNDM_train.json', 'r'))
    save_path = 'C:/ProjectData/CNNDM/bart-large-cnn-result/train/'
    if not os.path.exists(save_path): os.makedirs(save_path)

    model = model.cuda()

    for treat_index in tqdm.trange(len(train_data)):
        treat_sample = train_data[treat_index]
        if os.path.exists(os.path.join(save_path, treat_sample['filename'] + '.json')): continue
        with open(os.path.join(save_path, treat_sample['filename'] + '.json'), 'w'):
            pass

        article = tokenizer.encode(' ' + train_data[treat_index]['article'], max_length=1000, truncation=True,
                                   return_tensors='pt').cuda()
        result = model.generate(article).squeeze().detach().cpu().numpy().tolist()
        json.dump(result, open(os.path.join(save_path, treat_sample['filename'] + '.json'), 'w'))
        # exit()
