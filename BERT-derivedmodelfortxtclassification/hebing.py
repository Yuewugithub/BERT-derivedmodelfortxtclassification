import os

# 文件夹路径
folder_path = 'D:/BaiduNetdiskDownload/togongda-report/togongda-report/挖矿-英/'  # 请用你的实际文件夹路径替换
output_file = 'combined.txt'  # 合并后的文件名称
with open(output_file, 'w',encoding="utf-8") as outfile:
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r',encoding="utf-8") as readfile:
                outfile.write(readfile.read())
                outfile.write("\n")