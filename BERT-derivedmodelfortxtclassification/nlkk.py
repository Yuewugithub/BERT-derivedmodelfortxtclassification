with open('aug.txt', 'r',encoding="utf-8") as infile, open('output.txt', 'w',encoding="utf-8") as outfile:
    # 遍历原文件的每一行
    for line in infile:
        # 去除每一行最后的换行符（如果有的话），然后删除倒数第二个字符，最后再加上换行符
        line = line.rstrip('\n')
        if len(line) > 1:  # 如果行的长度大于1，即有至少两个字符
            outfile.write(line[:-2] + line[-1] + '\n')
        else:
            outfile.write(line + '\n')  # 如果行的长度不大于1，直接写入