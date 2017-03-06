
step 1：运用Dollar的原始Matlab代码训练分类器，训练集采用的是Caltech数据集中的INRIA部分，Caltech_code3.2.0.zip是Caltech数据集的读取工具包，只需要解压到Dollar代码的toolbox/detector目录中就可以。如需要添加自己的训练集训练，只需要修改训练集目录\train\pos\和\train\posGt\中的相应文件。（\train\pos\是正样本图片目录，\train\posGt\是正样本图片的注释文件目录）

step 2：训练好行人检测分类器后，将分类器加载到Matlab中，运行DetectorMat2Text.m，生成ClassifierOut.txt文件。

Step 3：将ClassifierOut.txt文件复制到行人检测程序（ChnFtrsAdaboost-OpencvPurePedestrianDetection）的根目录中，做好相应配置，运行程序即可。可根据实际需求修改程序中PedDetector实例的参数，以实现最好的检测效果。

PS.由于我自己写的训练代码训练效果并不稳定，所以附上的是Dollar的原始训练代码，效果基本一致。

参考文献：
Dollár P, Tu Z, Perona P, et al. Integral Channel Features[C]// British Machine Vision Conference, BMVC 2009, London, UK, September 7-10, 2009. Proceedings. DBLP, 2009.
Dollar P, Belongie S, Perona P. The fastest pedestrian detector in the west[J]. 2010.
Dollar P, Wojek C, Schiele B, et al. Pedestrian Detection: An Evaluation of the State of the Art[J]. IEEE Transactions on Pattern Analysis & Machine Intelligence, 2012, 34(4):743-761.