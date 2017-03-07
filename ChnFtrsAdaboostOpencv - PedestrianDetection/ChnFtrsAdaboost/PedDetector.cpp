#include "stdafx.h"
#include "PedDetector.h"
#include "DollarMex.h"
#include <algorithm>
using namespace std;


PedDetector::PedDetector(void)
{
	ChnFtrsChannelNum = 10;
	nPerOct = 8;
	m_Shrink = 4;
	OverLapRate = 0.65;
	num = 0;
	softThreshold = NULL;
	FeaID = NULL; 
	DCtree = NULL;
}


PedDetector::~PedDetector(void)
{
	Release();
}

void PedDetector::Release()
{
	if (DCtree != NULL)
		delete []DCtree;

	if (FeaID != NULL){
		delete[] FeaID;
		FeaID = NULL;
	}
}

bool PedDetector::loadStrongClassifier(const char *pFilePath)
{
	FILE *fs=fopen(pFilePath, "r");
	Release();
/*	this->num = WeakNum;*/
	fscanf(fs, "%d", &this->num);
	this->objectSize.width = 64, this->objectSize.height = 128; 
	this->softThreshold = -1;
	this->FeaID = new int[this->num*3];
	this->xStride = 4;
	this->yStride = 4;
	this->scaleStride = 1.08; // 尺度在的步长没有采用此固定值
	this->nms = NMS_METHOD_MaxGreedy;
	// 读取2048个若分类器决策树上各个节点所对应的特征编号
	for (int i=0; i<this->num; i++)
	{
		for (int j=0; j<3; j++)
		{
			fscanf(fs, "%d ", &this->FeaID[i*3+j]);
		}
		int temp1, temp2, temp3, temp4;
		fscanf(fs, "%d %d %d %d ", &temp1, &temp2, &temp3, &temp4);
	}
	// 读取2048个弱分类器决策树上不同节点所对应的决策阈值
	this->DCtree=new WeakClassifier[this->num];
	for (int i=0; i<this->num; i++)
	{
		for (int j=0; j<3; j++)
		{
			fscanf(fs, "%f ", &this->DCtree[i].threshold[j]);
		}
		float temp1, temp2, temp3, temp4;
		fscanf(fs, "%f %f %f %f ", &temp1, &temp2, &temp3, &temp4);
	}
	// 读取2048个弱分类器决策树上不同节点所对应的权值
	for (int i=0; i<this->num; i++)
	{
		for (int j=0; j<7; j++)
		{
			fscanf(fs, "%f ", &this->DCtree[i].hs[j]);
		}
	}
	fclose(fs);
	// 初始化特征位置索引
	FeaIn = new FtrIndex[5120];
	int m=0;
	CvRect rect;
	rect.width = (this->objectSize.width)/m_Shrink;
	rect.height = (this->objectSize.height)/m_Shrink;
	for( int z=0; z<ChnFtrsChannelNum; z++ )
		for( int c=0; c<rect.width; c++ )
			for( int r=0; r<rect.height; r++ )
			{
				FeaIn[m].Channel=z;
				FeaIn[m].x=c;
				FeaIn[m++].y=r;
			}

			return true;
}

float PedDetector::StrongClassify(CvMat *pChnFtrsData)
{
	float* tempChnFtrs;
	tempChnFtrs = pChnFtrsData->data.fl;
	float ans=0.0f;
	for (int i=0; i<num; i++)
	{
		ans+=DCtree[i].Classify(tempChnFtrs+3*i);
		if (ans<-1) return ans;
	}
	return ans;
}

bool PedDetector::show_detections(IplImage *pImg, CvMat *pAllResults, CvScalar color)
{
	if (pAllResults == NULL)
	{
		return true;
	}
	CvScalar FondColor;
	CvFont font;
	char str[100];
	cvInitFont(&font, CV_FONT_HERSHEY_PLAIN, 1, 1, 0, 2);
	FondColor = CV_RGB(255,255,255);
	int i;
	for (i=0; i<pAllResults->rows; i++)
	{
		CvScalar tmp = cvGet1D(pAllResults, i);
		tmp.val[0] = tmp.val[0];
		tmp.val[1] = tmp.val[1];
		tmp.val[2] = tmp.val[2];
		CvRect rectDraw;
		rectDraw.width = 40;
		rectDraw.height = 100;
		if (tmp.val[3] > 0)
			color = CV_RGB(0, 255, 0);
		else
			color = CV_RGB(200, 100, 100);
		{
			// 显示分数
			sprintf(str, "%.4f", tmp.val[3]);
			cvPutText(pImg, str, 
				cvPoint(int(tmp.val[0]-rectDraw.width/2*tmp.val[2]), int(tmp.val[1]-rectDraw.height/2*tmp.val[2])+12), 
				&font, FondColor);
			// 显示检测结果框
			cvRectangle(pImg, 
				cvPoint(int(tmp.val[0]-rectDraw.width/2*tmp.val[2]), int(tmp.val[1]-rectDraw.height/2*tmp.val[2])), 
				cvPoint(int(tmp.val[0]+rectDraw.width/2*tmp.val[2]), int(tmp.val[1]+rectDraw.height/2*tmp.val[2])), 
				color, 2);
		}
	}
	return true;
}

bool PedDetector::Detection_FPDW(IplImage *pImgInput, CvMat **pAllResults, float UpScale)
{
	CvMat ***ChnFtrs; // 存储积分通道特征
	float ***ChnFtrs_float; // 平滑前的特征
	float scaleStridePerOctave = 2.0f; // FPDW中每个Octive的尺度宽度
	float nowScale; // 当前尺度
	CvRect rect, PicRect; // rect是检测框大小，PicRect是当前尺度下的整幅图像大小
	float ans; // 结果分数
	float *FeaData = new float[this->num*3]; // 特征
	int shrink_rate = m_Shrink; // 稀疏率
	CvScalar *tmpScalar; // 检测结果格式

	int step, t;
	int t_AllIntegral = 0;
	float *Scales; // 图像金字塔不同层所对应的缩放倍数
	int itemp1, itemp2;

	CvMemStorage *memory; // 检测结果队列内存
	CvSeq *seqDetections; // 检测结果队列
	memory = cvCreateMemStorage();
	seqDetections = cvCreateSeq(0, sizeof(CvSeq), sizeof(CvScalar), memory);
	rect.width = 64, rect.height = 128;

	// 图像的存储格式转换，并转换到LUV颜色空间
	int h = (pImgInput->height/shrink_rate)*shrink_rate, w = (pImgInput->width/shrink_rate)*shrink_rate;
	IplImage *pImg = cvCreateImage(cvSize(w, h), pImgInput->depth, pImgInput->nChannels);
	cvSetImageROI(pImgInput, cvRect(0, 0, w, h));
	cvCopyImage(pImgInput, pImg);
	cvResetImageROI(pImgInput);
	//int h = pImg->height, w = pImg->width;
	int d=pImg->nChannels;
	int ChnBox = h*w;
	unsigned char *data = new unsigned char[h*w*3];
	IplImage *imgB = cvCreateImage(cvGetSize(pImg),IPL_DEPTH_8U,1);
	IplImage *imgG = cvCreateImage(cvGetSize(pImg),IPL_DEPTH_8U,1);
	IplImage *imgR = cvCreateImage(cvGetSize(pImg),IPL_DEPTH_8U,1);
	cvSplit(pImg, imgB, imgG, imgR, NULL);
	for (int i=0; i<ChnBox; i++){
		data[i] = imgR->imageData[i];
		data[ChnBox+i] = imgG->imageData[i];
		data[2*ChnBox+i] = imgB->imageData[i];
	}
	cvReleaseImage(&imgR);
	cvReleaseImage(&imgG);
	cvReleaseImage(&imgB);
	// 	for (int col = 0; col < w; col++)
	// 	{
	// 		for(int row = 0; row < h; row++)
	// 		{
	// 			itemp1 = row * pImg->widthStep + col * pImg->nChannels;
	// 			data[2*ChnBox+w*row+col] = pImg->imageData[itemp1]; 
	// 			data[ChnBox+w*row+col] = pImg->imageData[itemp1+1];  
	// 			data[w*row+col] = pImg->imageData[itemp1+2]; 
	// 		}
	// 	}
	float *luvImg = rgbConvert(data, ChnBox, 3, 2, 1.0/255);
	delete[] data;

	//计算图像金字塔层数
	nowScale=1.0;
	int nScales = min((int)(nPerOct*(log(min((double)pImg->width/64, (double)pImg->height/128))/log(2.0))+1), 
		(int)(nPerOct*log(UpScale)/log(2.0)+1));
	ChnFtrs = new CvMat**[nScales];
	Scales = new float[nScales];
	// 计算每层金字塔所对应的缩放倍数
	for (int i=0; i<nScales; i++){
		Scales[i] = pow(2.0, (double)(i+1)/nPerOct);
	}
	int *Octives = new int[nScales];
	int nOctives = 0;
	bool *isChnFtrs = new bool[nScales]; // 标记每层金字塔是否计算
	memset(isChnFtrs, 0, nScales*sizeof(bool));
	// 标记Octives，先计算Octives，然后其他尺度由相邻的Octives估计出来（FPDW方法，详见BMVC2010 - the fastest pedestrian detector in the west）
	while (nOctives*nPerOct<nScales){
		Octives[nOctives] = nOctives*nPerOct;
		nOctives++;
	}

	int *NearOctive = new int[nScales]; // 标记每层图像金字塔的近似源
	int iTemp = 0;
	for (int i=1; i<nOctives; i++){
		for (int j=iTemp; j<=(Octives[i-1]+Octives[i])/2; j++){
			NearOctive[j] = Octives[i-1];
		}
		iTemp = (Octives[i-1]+Octives[i])/2+1;
	}
	for (int i=iTemp; i<nScales; i++) NearOctive[i] = Octives[nOctives-1];
	//先计算Octives的积分通道特征
	for (int i=0; i<nOctives; i++){
		isChnFtrs[Octives[i]] = true;
		ChnFtrs[Octives[i]] = new CvMat *[ChnFtrsChannelNum];
		GetChnFtrs(luvImg, h, w, shrink_rate, Scales[Octives[i]], ChnFtrs[Octives[i]]);
	}
	// 估计出图像金字塔参数 lambdas
	float lambdas[10] = {0};
	if (nOctives<2){
		for (int i=3; i<10; i++) lambdas[i] = 0.1158;
	}
	else{
		for (int i=3; i<10; i++){
			float f0, f1;
			CvScalar tempS;
			tempS = cvSum(ChnFtrs[Octives[0]][i]);
			f0 = tempS.val[0]/(ChnFtrs[Octives[0]][i]->width*ChnFtrs[Octives[0]][i]->height);
			tempS = cvSum(ChnFtrs[Octives[1]][i]);
			f1 = tempS.val[0]/(ChnFtrs[Octives[1]][i]->width*ChnFtrs[Octives[1]][i]->height);
			lambdas[i] = log(f1/f0)/log(2.0);
		}
	}

	// 根据Octives近似resample出其他层的图像金子塔
	for (int i=0; i<nScales; i++){
		if (isChnFtrs[i]) continue;
		int hNow = (int)(h/Scales[i]/shrink_rate+0.5);
		int wNow = (int)(w/Scales[i]/shrink_rate+0.5);
		int h0 = ChnFtrs[NearOctive[i]][0]->height;
		int w0 = ChnFtrs[NearOctive[i]][0]->width;
		ChnFtrs[i] = new CvMat*[ChnFtrsChannelNum];
		for (int j=0; j<ChnFtrsChannelNum; j++){
			float ratio = pow(Scales[NearOctive[i]]/Scales[i], -lambdas[j]);
			ChnFtrs[i][j] = cvCreateMat(hNow, wNow, CV_32FC1);
			float_resample(ChnFtrs[NearOctive[i]][j]->data.fl, ChnFtrs[i][j]->data.fl, w0, wNow, h0, hNow, 1, ratio);
		}
	}
	for (int i=0; i<nScales; i++){
		for (int j=0; j<ChnFtrsChannelNum; j++){
			convTri1(ChnFtrs[i][j]->data.fl, ChnFtrs[i][j]->data.fl, ChnFtrs[i][j]->width, ChnFtrs[i][j]->height, 1, (float)2.0, 1);
		}
	}

	// AdaBoost分类计算过程
	for (step=0; step<nScales; step++)
	{
		//尺度金字塔结束条件
		if ((int)(pImg->width/Scales[step]+0.5f)<rect.width || (int)(pImg->height/Scales[step]+0.5f)<rect.height)
		{
			break;
		}
		CvSize ScaleSize = cvSize(ChnFtrs[step][0]->width*shrink_rate, ChnFtrs[step][0]->height*shrink_rate);

		PicRect.width = ChnFtrs[step][0]->width;
		PicRect.height = ChnFtrs[step][0]->height;
		// 密集滑窗操作
		for (PicRect.y = 0; PicRect.y+rect.height <= ScaleSize.height; PicRect.y += yStride)
		{
			for (PicRect.x = 0; PicRect.x+rect.width <= ScaleSize.width; PicRect.x += xStride)
			{
				rect.x=(PicRect.x/m_Shrink);
				rect.y=(PicRect.y/m_Shrink);

				//对于每个窗口的级联分类过程
				float score = 0.0;
				for (t=0; t<this->num; t++)
				{
					for (int j=0; j<3; j++)
					{
						int temp;
						temp=this->FeaID[t*3+j];
						FeaData[t*3+j]=ChnFtrs[step][FeaIn[temp].Channel]->data.fl[(FeaIn[temp].y+rect.y)*PicRect.width+FeaIn[temp].x+rect.x];
					}
					score += this->DCtree[t].Classify(FeaData+t*3);
					if (score<softThreshold) break;
				}
				if (score > 0.0)
				{
					tmpScalar = (CvScalar *)cvSeqPush(seqDetections, NULL);
					tmpScalar->val[0] = (PicRect.x + rect.width/2) * Scales[step];//检测框中心x坐标;
					tmpScalar->val[1] = (PicRect.y + rect.height/2) * Scales[step];//检测框中心y坐标;
					tmpScalar->val[2] = Scales[step];//检测框相对于行人模型的缩放倍数;
					tmpScalar->val[3] = score ; // 检测分数
				}
			}
		}
	}
	cvReleaseImage(&pImg);
	// 释放积分通道特征内存
	for (step=0; step<nScales; step++){
		for (int i=0; i<ChnFtrsChannelNum; i++){
			cvReleaseMat(&ChnFtrs[step][i]);
		}
		delete[] ChnFtrs[step];
	}
	delete[] ChnFtrs;


	// non maximum suppression 非极大值抑制过程
	CvMat *pDetections = NULL;
	CvMat *pModes = NULL;

	if (seqDetections->total > 0)
	{
		pDetections = cvCreateMat(seqDetections->total, 1, CV_64FC4);
		for (int i=0; i<seqDetections->total; i++)
		{
			tmpScalar = (CvScalar *)cvGetSeqElem(seqDetections, i);
			cvSet1D(pDetections, i, *tmpScalar);
		}

		if (nms == NMS_METHOD_MaxGreedy)
			MaxGreedy_NonMaxSuppression(pDetections, OverLapRate, &pModes);
		else
		{
			//输出所有detection
			pModes = (CvMat *)cvClone(pDetections);
		}
	}
	cvReleaseMemStorage(&memory);
	cvReleaseMat(&pDetections);

	if (*pAllResults != NULL)
		cvReleaseMat(pAllResults);
	*pAllResults = pModes;
	delete[] luvImg;
	delete[] FeaData;
	delete[] isChnFtrs;
	delete[] Scales;
	delete[] Octives;
	delete[] NearOctive;
	return true;
}

// NMS非极大值抑制，运动贪心的办法，将检测结果按分数排序，从前至后按覆盖率筛选结果;
static int cmp_detection_by_score(const void *_a, const void *_b)
{
	double *a = (double *)_a;
	double *b = (double *)_b;
	if (a[3] > b[3]) // [0]:x, [1]:y, [2]:s, [3]:score
		return -1;
	if (a[3] < b[3])
		return 1;
	return 0;
}
// 获取两个检测框的相互覆盖率
double GetOverlapRate(double *D1, double *D2)
{
	double Pw, Ph;
	Pw = 41.0, Ph = 100.0;
	double xr1, yr1, xl1, yl1;
	double xr2, yr2, xl2, yl2;
	xl1 = D1[0]-D1[2]*Pw/2;	
	xr1 = D1[0]+D1[2]*Pw/2;
	xl2 = D2[0]-D2[2]*Pw/2;
	xr2 = D2[0]+D2[2]*Pw/2;
	double ix = min(xr1, xr2) - max(xl1, xl2);
	if (ix<0) return -1;
	yr2 = D2[1]+D2[2]*Ph/2;
	yl1 = D1[1]-D1[2]*Ph/2;
	yr1 = D1[1]+D1[2]*Ph/2;
	yl2 = D2[1]-D2[2]*Ph/2;
	double iy = min(yr1, yr2) - max(yl1, yl2);
	if (iy<0) return -1;
	return ix*iy/min((yr1-yl1)*(xr1-xl1), (yr2-yl2)*(xr2-xl2));

}
void PedDetector::MaxGreedy_NonMaxSuppression(CvMat *pDetections, float overlap, CvMat **pModes)
{
	// 对于所有检测框按分数高低排序
	qsort((void *)pDetections->data.db, pDetections->rows, 4*sizeof(double), cmp_detection_by_score);
	int n = pDetections->rows;
	int ReTotal=0;
	bool *isHold = new bool[n];
	memset(isHold, 1, n*sizeof(bool));
	// 按分数从高到底，保留高分检测框，并剔除掉与保留窗口覆盖率超过overlap的低分值窗口（贪心策略）
	for (int i=0; i<n; i++){
		if (!isHold[i]) continue;
		ReTotal++;
		CvScalar Di;
		Di = cvGet1D(pDetections, i);
		for (int j=i+1; j<n; j++){
			double overlapRate;
			CvScalar Dj;
			Dj = cvGet1D(pDetections, j);
			overlapRate = GetOverlapRate(Di.val, Dj.val);
			if (overlapRate<0) continue;
			if (overlapRate>overlap) isHold[j] = false;
		}
	}
	*pModes = cvCreateMat(ReTotal, 1, CV_64FC4);
	ReTotal=0;
	for (int i=0; i<n; i++){
		if (isHold[i]){
			CvScalar temp;
			temp = cvGet1D(pDetections, i);
			cvSet1D(*pModes, ReTotal, temp);
			ReTotal++;
		}
	}
	delete[] isHold;
}

float* convConst(float* I,int h, int w,int d, float r)
{
	int s=1;
	float *O = new float[h*w*d];
	if (r<=1){
		convTri1(I, O, h, w, d, (float)12/r/(r+2)-2, s);
	}
	else{
		convTri(I, O, h, w, d, (int)(r+0.1), s);
	}
	return O;
}

bool PedDetector::GetChnFtrs(float *pImgInput, float h0, float w0,int shrink_rate, float nowScale,CvMat *ChnFtrs[])
{
	float *I, *M, *O, *LUV, *S, *H;
	int h,w, wS, hS, d=3;
	int normRad=5;
	float normConst = 0.005;

	// 将原始图像数据缩放到当前尺度
	h = shrink_rate*(int)(h0/nowScale/shrink_rate+0.5), w = shrink_rate*(int)(w0/nowScale/shrink_rate+0.5);
	float *data = new float[h*w*3];
	wS = (w/m_Shrink), hS = (h/m_Shrink);
	float_resample(pImgInput, data, w0, w, h0, h, 3, 1.0);
	// 对图像数据进行卷积操作
	I = convConst(data, w, h, d, 1);

	//free(luvImg);
	M = new float[w*h];
	O = new float[w*h];
	// 计算梯度幅值
	gradMag(I, M, O, w, h, 3 );
	// 对梯度幅值图像数据进行卷积操作
	S = convConst(M, w, h, 1, normRad);
	// 归一化
	gradMagNorm(M, S, w, h, normConst);
	H = new float[wS*hS*6];
	memset(H, 0, wS*hS*6*sizeof(float));
	// 计算梯度方向
	gradHist(M, O, H, w, h, m_Shrink, 6, false);

	float *M_shrink = new float[wS*hS];
	float *I_shrink = new float[wS*hS*3];
	// 对图像进行稀疏重采样操作，相当于计算并列m_Shrink*m_Shrink矩形框特征
	float_resample(M, M_shrink, w, wS, h, hS, 1, (float)1.0);
	float_resample(I, I_shrink, w, wS, h, hS, 3, (float)1.0);

	// 保存最终结果
	for (int i=0; i<3; i++){
		ChnFtrs[i] = cvCreateMat(hS, wS, CV_32FC1);
		for (int j=0; j<hS*wS; j++){
			ChnFtrs[i]->data.fl[j] = I_shrink[i*hS*wS+j];
		}
	}
	ChnFtrs[3] = cvCreateMat(hS, wS, CV_32FC1);
	for (int i=0; i<hS*wS; i++){
		ChnFtrs[3]->data.fl[i] = M_shrink[i];
	}
	for (int i=4; i<10; i++){
		ChnFtrs[i] = cvCreateMat(hS, wS, CV_32FC1);
		for (int j=0; j<hS*wS; j++){
			int mod_i = (13-i)%6; // H的输出数据排列方向相反，需反向处理。
			ChnFtrs[i]->data.fl[j] = H[mod_i*hS*wS+j];
		}
	}

	free(I);
	free(M);
	free(O);
	free(S);
	free(M_shrink);
	free(I_shrink);
	free(H);
	free(data);

	return true;
}