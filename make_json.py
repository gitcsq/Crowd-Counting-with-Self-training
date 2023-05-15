import os
import json
import random
labeled_rate = 0.05
root = '/home/caosq/datasets/ShanghaiTech_Crowd_Counting_Dataset'
part_A_train = os.path.join(root, 'part_A_final/train_data', 'images')
part_A_test = os.path.join(root, 'part_A_final/test_data', 'images')
part_B_train = os.path.join(root, 'part_B_final/train_data', 'images')
part_B_test = os.path.join(root, 'part_B_final/test_data', 'images')

part_A_train_list = [os.path.join(part_A_train, m) for m in os.listdir(part_A_train)]
# file_path = './part_A_train.json'
# with open(file_path, mode='w') as file_obj:
#     json.dump(part_A_train_list, file_obj)

file_path = './part_A_train_{}.json'.format(str(labeled_rate))
random.shuffle(part_A_train_list)
with open(file_path, mode='w') as file_obj:
    json.dump(part_A_train_list[:int(labeled_rate*len(part_A_train_list))], file_obj)
file_path = './part_A_unlabeled_{}.json'.format(str(labeled_rate))
with open(file_path, mode='w') as file_obj:
    json.dump(part_A_train_list[int(labeled_rate*len(part_A_train_list)):], file_obj)


# part_A_test_list = [os.path.join(part_A_test, m) for m in os.listdir(part_A_test)]
# file_path = './part_A_test.json'
# with open(file_path, mode='w') as file_obj:
#     json.dump(part_A_test_list, file_obj)

part_B_train_list = [os.path.join(part_B_train, m) for m in os.listdir(part_B_train)]
# file_path = './part_B_train.json'
# with open(file_path, mode='w') as file_obj:
#     json.dump(part_B_train_list, file_obj)

file_path = './part_B_train_{}.json'.format(str(labeled_rate))
random.shuffle(part_B_train_list)
with open(file_path, mode='w') as file_obj:
    json.dump(part_B_train_list[:int(labeled_rate*len(part_B_train_list))], file_obj)

file_path = './part_B_unlabeled_{}.json'.format(str(labeled_rate))
with open(file_path, mode='w') as file_obj:
    json.dump(part_B_train_list[int(labeled_rate * len(part_B_train_list)):], file_obj)

# part_B_test_list = [os.path.join(part_B_test, m) for m in os.listdir(part_B_test)]
# file_path = './part_B_test.json'
# with open(file_path, mode='w') as file_obj:
#     json.dump(part_B_test_list, file_obj)

