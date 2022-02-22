import os
import numpy
import torch
import json
import tqdm
import transformers
from Tools import ProgressBar, SaveModel
from Loader_NCLS import NCLS_Loader

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

batch_size = 10

if __name__ == '__main__':
    tokenizer = transformers.MarianTokenizer.from_pretrained("C:/PythonProject/opus-mt-en-zh")

    load_path = 'E:/ProjectData/NCLS/EN2ZH-Summarization/MarianMT-All-Multi/'
    test_data = NCLS_Loader(sample_number=5000, use_part='test')
    for epoch_name in os.listdir(load_path)[::-1]:
        print(epoch_name)
        if epoch_name.find('Parameter') == -1: continue
        if epoch_name.find('Result') != -1: continue
        save_path = os.path.join(load_path, epoch_name + '-ResultBeam')
        if os.path.exists(save_path):
            # continue
            pass
        else:
            os.makedirs(save_path)

        model = transformers.MarianMTModel.from_pretrained(os.path.join(load_path, epoch_name))
        model.eval()
        model = model.cuda()

        total_loss, train_counter = 0.0, 0
        for batch_index in tqdm.trange(0, len(test_data), batch_size):
            if os.path.exists(os.path.join(save_path, '%08d.json' % batch_index)): continue
            file = open(os.path.join(save_path, '%08d.json' % batch_index), 'w')
            model_inputs = tokenizer([_['Article'] for _ in test_data[batch_index:batch_index + batch_size]],
                                     max_length=512, padding='max_length', truncation=True, return_tensors='pt')
            labels = [_['CrossLingualSummary'] for _ in test_data[batch_index:batch_index + batch_size]]

            outputs = model.generate(input_ids=model_inputs['input_ids'].cuda(),
                                     attention_mask=model_inputs['attention_mask'].cuda(), num_beams=4)
            predict_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            json.dump(predict_sentence, file)
            file.close()
        # exit()
