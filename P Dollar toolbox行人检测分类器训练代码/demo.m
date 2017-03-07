
t=load('models\AcfInriaDetector.mat');%%训练好的检测器
detector=t.detector;
filename = 'D:\test\03.BMP';%%需要检测图片的位置
%% modify detector (see acfModify)
detector = acfModify(detector,'cascThr',-1,'cascCal',0);

%% run detector on a sample image (see acfDetect)
I=imread(filename); tic, bbs=acfDetect(I,detector); toc
figure(1); im(I); bbApply('draw',bbs); pause(.1);