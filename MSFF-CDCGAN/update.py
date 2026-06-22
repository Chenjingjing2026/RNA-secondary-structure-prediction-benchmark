import os


def delete_extra_files(folder1, folder2):
    # 获取文件夹1和文件夹2中的所有文件
    files_folder1 = set(os.listdir(folder1))  # 获取folder1中的文件列表，并转换为集合（去重）
    files_folder2 = set(os.listdir(folder2))  # 获取folder2中的文件列表，并转换为集合（去重）

    # 找出folder1中有但folder2中没有的文件
    extra_files = files_folder1 - files_folder2

    # 删除folder1中多余的文件
    for file in extra_files:
        file_path = os.path.join(folder1, file)
        try:
            if os.path.isfile(file_path):  # 确保是文件而非文件夹
                os.remove(file_path)  # 删除文件
                print(f"删除文件: {file}")
            else:
                print(f"{file} 不是一个文件，跳过删除")
        except Exception as e:
            print(f"删除文件 {file} 时发生错误: {e}")


# 示例调用
folder1 = "/home/chenjingjing/Models/MSFF-CDCGAN/picture/bprna_new/test/str_photo"
folder2 = "/home/chenjingjing/Models/MSFF-CDCGAN/picture/bprna_new/test/seq_photo"
delete_extra_files(folder1, folder2)
