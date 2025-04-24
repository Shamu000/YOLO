import os

def check_file(file_path):
    """
    检查文件是否符合要求。
    如果文件有格式错误，返回 False 并删除该文件。
    如果文件无误，返回 True。
    """
    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, 1):
            parts = line.strip().split()
            # 检查是否恰好有13个数
            if len(parts) != 13:
                print(f"{file_path} (line {line_number}): Expected 13 numbers, found {len(parts)}")
                return False

            # 检查第一个数是否是非负数，且在[0, 26]范围内
            try:
                first_num = float(parts[0])
                if first_num < 0 or first_num > 35:
                    print(f"{file_path} (line {line_number}): First number '{parts[0]}' should be between 0 and 26")
                    return False
            except ValueError:
                print(f"{file_path} (line {line_number}): First value '{parts[0]}' is not numeric")
                return False

            # 检查其余每个数是否都是大于0小于1的数
            for part in parts[1:]:
                try:
                    num = float(part)
                    if num <= 0 or num >= 1:
                        print(f"{file_path} (line {line_number}): Number '{part}' should be between 0 and 1")
                        return False
                except ValueError:
                    print(f"{file_path} (line {line_number}): Non-numeric value '{part}'")
                    return False
    return True

def check_and_remove_files(folder_path):
    """
    遍历文件夹中的所有 .txt 文件，检查格式。
    如果文件有错误，则删除。
    """
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            if not check_file(file_path):
                print(f"Deleting file: {file_path}")
                os.remove(file_path)

# 使用函数
folder_path = '/media/zjy/TOSHIBA EXT/330_RMUL_上午/labels'  # 你的文件夹路径
check_and_remove_files(folder_path)
