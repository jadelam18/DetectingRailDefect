# DetectingDefect
钢轨缺陷检测  
  
1、文件说明：  
文件夹“！Type-II RSDDs dataset”为钢轨缺陷图片；  
“mainEntry.py”为主程序文件  
“mainForm.py”为UI文件  
"【批】目标分割开运算丨删小块丨画轮廓丨画矩形"为批处理文件（路径需要改）
  
2、系统（mainEntry.py）使用说明：  
本系统会将上个步骤中所处理完的图片，传送到下一个步骤以供处理。  
首先，在“原图”区域点击“打开图片”，可显示数据集里的单张图片；  
第二步，在“钢轨图像预处理”依次点击“图像增强”和“图像去噪”，计算机将依次对钢轨图像进行CLAHE和中值滤波，从而获得一张对比度更高、总体质量更高的图像；  
第三步，在“钢轨缺陷分割”点击“目标分割”，将会用自适应阈值分割法分割出钢轨图像上的大致缺陷；  
第四步，在“形态学处理”区域依次点击“开运算”和“删小区域+膨胀”，将会去除大部分分割图像中的杂质的干扰，并且尽量恢复缺陷原貌；  
最后在“缺陷特征提取”区域点击“特征提取”，便会标记出这张钢轨图片的缺陷位置以及将它相应的特征值输出在最右边的“检测结果”区域。  
  
另外，也可以在“打开图片”后，直接点击“检测结果”，也会显示上述检测结果。
