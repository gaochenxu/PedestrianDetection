#pragma once

#include <vector>
#include <cv.h>
#include <highgui.h>
using namespace std;
/*
*/
enum NMS_METHOD
{
	NMS_METHOD_NONE=0,
	NMS_METHOD_MeanShift,
	NMS_METHOD_MaxGreedy,
	NMS_METHOD_PairWiseMax,
	NMS_METHOD_PairWiseMax_1
};
// AdaBoost弱分类器结构
struct WeakClassifier	/*	两层深度的决策树depth-2 decision tree	*/
{
	float threshold[3];	/* 弱分类器每个节点的阈值threshold	*/
	float hs[7];		/* 每个节点的权值 1为第一层节点 2和3为第二层节点 3,4,5,6为第三层最终叶子节点 */

	//取叶子节点的最终权重
	float Classify(float *f)
	{
		return hs[GetLeafNode(f)];
	}
	// 根据决策树阈值判断，寻找对应叶子节点的位置
	int GetLeafNode(float *f)
	{
		if (f[0] < threshold[0])
		{
			return (f[1] < threshold[1]) ? 3:4;
		}
		else
		{
			return (f[2] < threshold[2]) ? 5:6;
		}
	}
};

// 存储分类器特征编号所对应的积分通道索引位置
struct FtrIndex 
{
	int Channel;
	int x;
	int y;
};

class PedDetector
{
public:
	int num;							/* 弱分类器个数 number of weak classifier	*/
	int *FeaID;							/* 决策树分类器每个节点所对应的特征编号，两层决策树，一共三个节点，4个叶子节点不对应特征 */
	int xStride, yStride;				/*滑窗的步长*/
	float scaleStride;					/*尺度的步长*/
	WeakClassifier *DCtree;				/*双层决策树分类器*/
	int m_Shrink;						/*模型稀释倍数，相当于特征矩形框的边长，一般默认为4 */
	int ChnFtrsChannelNum;				/*积分通道数，默认为10,3个LUV颜色通道+1个梯度幅值+6个梯度方向*/
	FtrIndex *FeaIn;				/*存储分类器特征编号所对应的积分通道图索引位置*/
	int nPerOct;						/*图像平均每缩放2倍，需要建立金字塔的层数*/

	NMS_METHOD nms;						/*非极大值抑制算法*/
	float OverLapRate;					/*当运用贪婪算法进行非极大值抑制时，该值代表覆盖率阈值，剔除掉覆盖率大于此值的检测框*/

	CvSize objectSize;					/* 模型大小: 64x128 */
	float softThreshold;				/* softThreshold阈值，默认取-1 */

public:
	PedDetector(void);
	~PedDetector(void);
	void Release();
	/* 
	将基于MatLab代码训练好的分类器导入
	输入：分类器路径、弱分类器数 */
	bool loadStrongClassifier(const char *pFilePath);
	/* 
	AdaBoost强分类器
	输入：特征序列
	输出：最终检测分数 */
	float StrongClassify(CvMat *pChnFtrsData);

	/*
	将行人检测结果打印到原始图像中并显示
	输入：
	pImg 原始图像
	pAllResults 检测结果
	color 检测框显示颜色 */
	bool show_detections(IplImage *pImg, CvMat *pAllResults, CvScalar color =  CV_RGB(0, 255, 0));
	/*
	提取积分通道特征
	输入：
	pImgInput float格式的图像数据
	h0 图像高
	w0 图像宽
	shrink_rate 图像稀释率，即特征矩形框边长
	nowScale 图像金字塔尺度值
	ChnFtrs 存储最终特征，10个通道 */
	bool GetChnFtrs(float *pImgInput, float h0, float w0,int shrink_rate, float nowScale,CvMat *ChnFtrs[]);

	/*
	行人检测接口函数，基于FPDW方法，详见BMVC2010 - the fastest pedestrian detector in the west）
	输入：
	pImgInput 待检测图像
	pAllResults 存储检测结果，存储格式为（每行数据按 检测框中心x坐标、检测框中心y坐标、检测框相对于行人模型的缩放倍数、最终检测得分 排列）
	UpScale 检测行人的最大尺度上限，以128*64大小为基准 */
	bool Detection_FPDW(IplImage *pImgInput, CvMat **pAllResults, float UpScale=99999);

	// NMS非极大值抑制，运动贪心的办法，将检测结果按分数排序，从前至后按覆盖率筛选结果;
	void MaxGreedy_NonMaxSuppression(CvMat *pDetections, float overlap, CvMat **pModes);
};

