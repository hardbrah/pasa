import json
import random   
# 1360 neg
# 905+455 pos
def merge_files_to_json():
    # 读取q_false_neg.txt内容
    with open('q_false_neg.txt', 'r') as f1:
        data_neg = f1.read()
    
    # 读取q_false_true.txt内容
    with open('q_true_pos.txt', 'r') as f2:
        data_true = f2.read()
    
    # 合并数据（按顺序保留两个文件内容）
    merged_data = data_neg +"\n" + data_true
    
    # 写入JSON文件
    with open('selector_train.json', 'w') as f_out:
        f_out.write(merged_data)

        import random
def shuffle_text():
    # 读取文件内容
    with open('selector_train.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 打乱行顺序
    random.shuffle(lines)

    # 将打乱后的内容写入新的文件
    with open('shuffled_selector_train.txt', 'w', encoding='utf-8') as file:
        file.writelines(lines)

    print("内容已按行打乱并保存到 shuffled_1.txt")

if __name__ == "__main__":
    shuffle_text()