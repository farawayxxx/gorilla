import os
import json
import jsonlines
from tqdm import tqdm

def write_to_jsonl(data, filename):
    """
    Writes a list of instances to a .jsonl file, one instance per line.

    Parameters:
    - data (list): List of instances to write.
    - filename (str): The name of the file to write to.
    """
    with open(filename, 'w', encoding='utf-8') as file:
        for item in data:
            json_str = json.dumps(item, ensure_ascii=False)
            file.write(json_str + "\n")

def process_jsonl_files(query_folder, answer_folder):
    result_list = []
    
    # 获取query文件夹中所有jsonl文件
    query_files = [f for f in os.listdir(query_folder) if f.endswith('.json')]
    
    for file_name in tqdm(query_files):
        query_path = os.path.join(query_folder, file_name)
        answer_path = os.path.join(answer_folder, file_name)
        
        # 获取文件名(不含后缀)
        split_name = os.path.splitext(file_name)[0]
        
        # 判断是否是exec文件
        is_exec_file = 'exec' in file_name.lower()
            
        try:
            # 读取query文件的所有行
            query_data_list = []
            with jsonlines.open(query_path, 'r') as query_reader:
                for query_line in query_reader:
                    query_data_list.append(query_line)
            
            if is_exec_file:
                # 对于exec文件，直接从query文件中获取ground_truth
                for query_item in query_data_list:
                    result_dict = {
                        'id': query_item['id'],
                        'split': split_name,
                        'function': query_item.get('function', []),
                        'query': query_item.get('question', []),
                        'ground_truth': query_item.get('ground_truth', [])
                    }
                    result_list.append(result_dict)
            else:
                # 对于非exec文件，需要从answer文件中获取ground_truth
                if not os.path.exists(answer_path):
                    print(f"Warning: No matching file found in answer folder for {file_name}")
                    continue
                    
                # 读取answer文件的所有行
                answer_data_list = []    
                with jsonlines.open(answer_path, 'r') as answer_reader:
                    for answer_line in answer_reader:
                        answer_data_list.append(answer_line)
                
                # 创建answer_dict用于快速查找
                answer_dict = {item['id']: item for item in answer_data_list}
                
                # 处理每一行数据
                for query_item in query_data_list:
                    query_id = query_item['id']
                    
                    # 查找对应的answer
                    if query_id in answer_dict:
                        answer_item = answer_dict[query_id]
                        
                        result_dict = {
                            'id': query_id,
                            'split': split_name,
                            'function': query_item.get('function', []),
                            'query': query_item.get('question', []),
                            'ground_truth': answer_item.get('ground_truth', [])
                        }
                        result_list.append(result_dict)
                    else:
                        print(f"Warning: No matching answer found for query ID {query_id}")
                
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in file {file_name}: {e}")
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
    
    return result_list


query_file = "./data"
answer_file = "./data/possible_answer"

data = process_jsonl_files(query_folder=query_file, answer_folder=answer_file)
write_to_jsonl(data, "./bfcl.jsonl")