// ChnFtrsAdaboost.cpp : 定义控制台应用程序的入口点。FPDW版
//

#include "stdafx.h"
/*#include <vld.h>*/
#include "PedDetector.h"

void DetectionExample();
int main(int argc, _TCHAR* argv[])
{
	DetectionExample();

	return 0;
}

void DetectionExample()
{
	// 实例化行人检测器
	PedDetector PD;
	// 加载行人检测分类器
	PD.loadStrongClassifier("ClassifierOut.txt");
	// 读取待检测图像
	IplImage *img = cvLoadImage("D:\\test\\I00016.png");
	// 检测并输出结果显示
	CvMat *ReMat=NULL;
	PD.Detection_FPDW(img, &ReMat, 3);
	PD.show_detections(img, ReMat);

	cvNamedWindow("test");
	cvShowImage("test", img);
	cvWaitKey(0);
}