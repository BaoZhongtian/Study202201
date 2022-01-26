import os
import json
import numpy

if __name__ == '__main__':
    total_data = json.load(open('CNNDM_train.json', 'r'))
    print(total_data[0])
    numpy.random.shuffle(total_data)
    print(total_data[0])

    json.dump(total_data[0:100000], open('CNNDM_train_first100K.json', 'w'))
    json.dump(total_data[100000:200000], open('CNNDM_train_second100K.json', 'w'))
    json.dump(total_data[200000:300000], open('CNNDM_train_third100K.json', 'w'))
