# os.chdir('D:/BaiduNetdiskDownload/togongda-report/togongda-report/APT-英-600')
#
# path = r'D:/BaiduNetdiskDownload/togongda-report/togongda-report/APT-英-600/'  #【设定pdf文件路径】
# txt_save_path = 'D:/BaiduNetdiskDownload/togongda-report/togongda-report/APT-英-600/TXT'  # 【设定保存TXT文件路径】
from pdfminer.high_level import extract_text
import os


def pdf_to_txt(input_path, output_path):
    print(input_path)
    text = extract_text(input_path)
    print(output_path)
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(text)
directory = 'D:/BaiduNetdiskDownload/togongda-report/togongda-report/挖矿-英-19/'
files = os.listdir(directory)
#print(files)
savedir = 'D:/BaiduNetdiskDownload/togongda-report/togongda-report/挖矿-英/'
for file in files:
    try:

        pdf_to_txt(directory+file, savedir+file[:-3]+"txt")
    except:
        continue
