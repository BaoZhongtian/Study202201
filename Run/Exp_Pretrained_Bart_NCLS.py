import os
import json
import tqdm
from Loader_NCLS import NCLS_Loader
from transformers import BartForConditionalGeneration, BartTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if __name__ == '__main__':
    model = BartForConditionalGeneration.from_pretrained(
        'C:/ProjectData/NCLS/bart-finetuning-640Length/00005984-Parameter')
    tokenizer = BartTokenizer.from_pretrained('C:/ProjectData/bart-large-cnn')

    train_data = NCLS_Loader(sample_number=999999, use_part='test')
    save_path = 'C:/ProjectData/NCLS/bart-finetuning-result-fewshot-5984/test/'
    if not os.path.exists(save_path): os.makedirs(save_path)

    model = model.cuda()

    for treat_index in tqdm.trange(len(train_data)):
        treat_sample = train_data[treat_index]

        if os.path.exists(os.path.join(save_path, '%08d.json' % treat_index)): continue
        with open(os.path.join(save_path, '%08d.json' % treat_index), 'w'):
            pass

        article = tokenizer.encode(' ' + train_data[treat_index]['Article'], max_length=1000, truncation=True,
                                   return_tensors='pt').cuda()
        result = model.generate(article).squeeze().detach().cpu().numpy().tolist()
        json.dump({'Predict': result, 'PredictSentence': tokenizer.decode(result, skip_special_tokens=True),
                   'Label': train_data[treat_index]['Summary']},
                  open(os.path.join(save_path, '%08d.json' % treat_index), 'w'))
        # exit()
