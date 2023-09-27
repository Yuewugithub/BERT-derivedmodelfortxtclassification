import os
path = "C:/project/BertTextClassification-main/HTMLhandle/lesuo"


datanames = os.listdir(path)
for i in datanames:
    print(i[:-3])
    with open('data6261.txt', 'a',encoding="utf-8") as file:
        file.write(i[:-3]+"&"+"1"+"\n")
