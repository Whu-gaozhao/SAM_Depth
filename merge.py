import os
 
# 分块影像所在文件夹，不能有中文
tifDir = r"/home/zhao/DataSets/ISPRS_dataset/Vaihingen/crop/"
# 输出的文件夹，不能有中文，如果文件夹不存在则会被创建
outPath = r"/home/zhao/DataSets/ISPRS_dataset/Vaihingen/merge/"
 
# ----------------只需修改上面两个参数--------------------
# 当前脚本所在的目录，用于找gdal_merge文件
currentPath = os.path.dirname(os.path.abspath(__file__))
 
# 输出目录不存在则创建
if not os.path.exists(outPath):
    os.makedirs(outPath)
 
# 列出输入文件夹的tif文件
tifs = [i for i in os.listdir(tifDir) if i.endswith(".tif")]
 
print("tifs" , tifs)
# 获取目标文件数量，前缀相同的
# 分块的文件名的长度相同，将分块编号去掉
targetFile = set()
for i in tifs:
    targetFile.add(i[:-26]+i[-4:])
print("拼接后应该有 %s 个文件" % len(targetFile))
print("targetFile" , targetFile)
# 切换工作空间，到影像的路径
os.chdir(tifDir)
# 一个一个的执行拼接
for i in targetFile:
    # 拼接后的文件名对应的分块文件名列表
    sliceFileList = []
    for k in tifs:
        if k[:-26] == i[:-4]:
            sliceFileList.append(k)
 
    # 执行影像拼接
    outtif = os.path.join(outPath,i)  # 输出tif路径，使用绝对路径
    if os.path.exists(outtif):   # 如果输出路径存在则跳过
        print("%s already exists, will be ignored."%outtif)
        continue
    os.system("python %s -init %s  -o %s %s" % (
        os.path.join(currentPath,"gdal_merge.py"),0,  outtif, " ".join(sliceFileList)))
 
    print("%s -------- merge success"%outtif)