#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <cstdio>
#include <ctime>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include "ParallelPro.h"
#include "parallel_for.h"
#include "TBBParallelPro.h"
#include "opencv2/gpu/gpu.hpp"
#include "Eigen/core"
#include "Eigen/sparse"
#include <glpk.h>
#include "imageLib.h"
#include "flowIO.h"
#include "colorcode.h"
#include "Image.h"
#include "Image.cpp"
#include "MotionToColor.h"
using namespace cv;
using namespace std;
using namespace tbb;
using namespace Eigen;

void neigborPixels(int i, int j, int rows, int cols, int neighWidth, int neigborHeight, vector<vector<int> > &matPixels)
{
	int iSt = i - neighWidth >= 0 ? i - neighWidth : 0;
	int iEnd = i + neighWidth < rows ? i + neighWidth : rows - 1;
	int jSt = j - neigborHeight >= 0 ? j - neigborHeight : 0;
	int jEnd = j + neigborHeight < cols ? j + neigborHeight : cols - 1;
	int elms = (iEnd - iSt + 1) * (jEnd - jSt + 1);
	int cnt = 0;
	vector<int> ngbPixels(elms, 0);
	for (int ir = iSt; ir <= iEnd; ++ir)
	{
		for (int jc = jSt; jc <= jEnd; ++jc)
		{
			ngbPixels[cnt++] = ir * cols + jc;
		}
	}
	matPixels.push_back(ngbPixels);
}

void imgMatToArray(Mat &img, VectorXd &imgAry)
{
	int rows = img.rows;
	int cols = img.cols;
	for (int i = 0; i != rows; ++i)
	{
		for (int j = 0; j != cols; ++j)
		{
			imgAry[i * cols + j] = img.at<double>(i, j);
		}
	}
}

void neighborMatrix(Mat &img, vector<vector<int> > &neighMat, int nWidth, int nHeight)
{
	int rows = img.rows;
	int cols = img.cols;
	int nPixels = rows * cols;
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			neigborPixels(i, j, rows, cols, nWidth, nHeight, neighMat);
		}
	}
}

double linSearch(VectorXd &img0, VectorXd &img1, double &gamma, SparseMatrix<double> &transMat, int iPixel, int minGradInd, int &nPixels, bool &flag)
{
	double s1 = 0;
	double s2 = 0;
	for (SparseMatrix<double>::InnerIterator it(transMat, iPixel); it; ++it)
	{
		int irow = it.row();
		if (irow == minGradInd)
		{
			s1 += (it.value() + gamma * (1 - it.value())) * img0(irow);
			flag = true;
		}
		else
		{
			s1 += (it.value() + gamma * (- it.value())) * img0(irow);
		}
		s2 += it.value() * img0(irow);
	}
	if (!flag)
	{
		s1 += gamma * img0(minGradInd);
	}
	s1 -= img1(iPixel);
	s2 -= img1(iPixel);
	s1 *= s1;
	s2 *= s2;
	return s1 - s2;
}

double multiplyDenseSparse(VectorXd &img, SparseMatrix<double> &trans, int cols)
{
	double mSum = 0;
	for (SparseMatrix<double>::InnerIterator it(trans, cols); it; ++it)
	{
		mSum += it.value() * img[it.row()];
	}
	return mSum;
}

void addDenseSparse(VectorXd &gradAry, SparseMatrix<double> &dmat, int cols, double lambda_1)
{
	for (SparseMatrix<double>::InnerIterator it(dmat, cols); it; ++it)
	{
		gradAry[it.row()] += lambda_1 * it.value();
	}
}

void homogMultiply(Vector2d &affHom, MatrixXd &affineMat, Vector3d &pHomg, int pix)
{
	affHom[0] = pHomg[0] * affineMat(pix, 0) + pHomg[1] * affineMat(pix, 1) + pHomg[2] * affineMat(pix, 2);
	affHom[1] = pHomg[0] * affineMat(pix, 3) + pHomg[1] * affineMat(pix, 4) + pHomg[2] * affineMat(pix, 5);
}

bool binarySearch(int irow, vector<vector<int> > &neighMat, int kPixel)
{
	int low = 0;
	int high = neighMat[kPixel].size() - 1;
	while (low <= high)
	{
		int mid = (low + high) / 2;
		if (neighMat[kPixel][mid] == irow)
		{
			return true;
		}
		else if (neighMat[kPixel][mid] < irow)
		{
			low = mid + 1;
		}
		else
		{
			high = mid - 1;
		}
	}
	return false;
}

bool IndexSearch(int irow, vector<vector<int>> &neighMat, int kPixel, int &ind)
{
	int low = 0;
	int high = neighMat[kPixel].size() - 1;
	ind = -1;
	while (low <= high)
	{
		int mid = (low + high) / 2;
		if (neighMat[kPixel][mid] == irow)
		{
			ind = mid;
			return true;
		}
		else if (neighMat[kPixel][mid] < irow)
		{
			low = mid + 1;
		}
		else
		{
			high = mid - 1;
		}
	}
	return false;
}
double filterPixelObj (VectorXd &img0, VectorXd &img1, SparseMatrix<double> &transMat, int iPixel)
{
	double s1 = 0;
	for (SparseMatrix<double>::InnerIterator it(transMat, iPixel); it; ++it)
	{
		int irow = it.row();
		s1 += it.value() * img0(irow);
	}
	s1 -= img1(iPixel);
	s1 *= s1;
	return s1;
}

int bSearch(vector<int> &ind, int target)
{
	int low = 0;
	int high = ind.size() - 1;
	while (low <= high)
	{
		int mid = low + (high - low) / 2;
		if (ind[mid] == target)
		{
			return mid;
		}
		else if (ind[mid] < target)
		{
			low = mid + 1;
		}
		else
		{
			high = mid - 1;
		}
	}
	return -1;
}
double OffilterPixelObj (VectorXd &img0, VectorXd &img1, SparseMatrix<double> &transMat, int iPixel, SparseMatrix<double> &DcomMat, double lambda_1, int reserveCap)
{
	double s1 = 0;
	vector<int> nonzeroInd;
	vector<int> nonZeroVal;
	nonzeroInd.reserve(reserveCap);
	nonZeroVal.reserve(reserveCap);
	for (SparseMatrix<double>::InnerIterator it(transMat, iPixel); it; ++it)
	{
		int irow = it.row();
		s1 += it.value() * img0(irow);
		nonzeroInd.push_back(irow);
		nonZeroVal.push_back(it.value());
	}
	s1 -= img1(iPixel);
	s1 *= s1;
	for (SparseMatrix<double>::InnerIterator it(DcomMat, iPixel); it; ++it)
	{
		int irow = it.row();
		int markPos = bSearch(nonzeroInd, irow);
		if (markPos != -1)
		{
			s1 += lambda_1 * it.value() * nonZeroVal[markPos];
		}
	}
	return s1;
}

double iterfilterPixelObj (VectorXd &img0, VectorXd &img1, SparseMatrix<double> &transMat, int iPixel, double &gamma, int minGradInd, bool &flag)
{
	double s1 = 0;
	flag = false;
	for (SparseMatrix<double>::InnerIterator it(transMat, iPixel); it; ++it)
	{
		int irow = it.row();
		if (irow == minGradInd)
		{
			flag = true;
			s1 += (it.value() + gamma * (1 - it.value())) * img0(irow);
		}
		else
		{
			s1 += (it.value() + gamma * (- it.value())) * img0(irow);
		}
	}
	if (!flag)
	{
		s1 += gamma * img0(minGradInd);
	}
	s1 -= img1(iPixel);
	s1 *= s1;
	return s1;
}

double totalFilterPixelObj(VectorXd &img0, VectorXd &img1, SparseMatrix<double> &transMat, int iPixel)
{
	double s1 = 0;
	for (SparseMatrix<double>::InnerIterator it(transMat, iPixel); it; ++it)
	{
		int irow = it.row();
		s1 += it.value() * img0(irow);
	}
	s1 -= img1(iPixel);
	s1 *= s1;
	return s1;
}

double OfTotalfilterPixelObj (VectorXd &img0, VectorXd &img1, SparseMatrix<double> &transMat, int iPixel, int reserveCap, double lambda_1, SparseMatrix<double> &DcompMat)
{
	double s1 = 0;
	double dtsum = 0;
	vector<int> nonzeroInd;
	vector<int> nonZeroVal;
	nonzeroInd.reserve(reserveCap);
	nonZeroVal.reserve(reserveCap);
	for (SparseMatrix<double>::InnerIterator it(DcompMat, iPixel); it; ++it)
	{
		int irow = it.row();
		nonzeroInd.push_back(irow);
		nonZeroVal.push_back(it.value());
	}

	for (SparseMatrix<double>::InnerIterator it(transMat, iPixel); it; ++it)
	{
		int irow = it.row();
		s1 += it.value() * img0(irow);
		int ind = -1;
		int posMark = bSearch(nonzeroInd, irow);
		if (posMark != -1)
		{
			dtsum += it.value() * nonZeroVal[posMark];
		}
	}

	s1 -= img1(iPixel);
	s1 *= s1;
	dtsum *= lambda_1;
	s1 += dtsum;
	return s1;
}

double ReIterfilterPixelObj (VectorXd &img0, VectorXd &img1, SparseMatrix<double> &transMat, int iPixel, double &gamma, vector<vector<int> > &neighMat, VectorXi &visited, VectorXd &transPixel)
{
	double s1 = 0;
	for (SparseMatrix<double>::InnerIterator it(transMat, iPixel); it; ++it)
	{
		int irow = it.row();
		int ind = -1;
		if (IndexSearch(irow, neighMat, iPixel, ind))
		{
			visited[ind] = 1;
			s1 += ((1 - gamma) * it.value() + gamma * transPixel(ind)) * img0(irow);
		}
		else
		{
			s1 += (1 - gamma) * it.value() * img0(irow);
		}
	}
	for (int i = 0; i != neighMat[iPixel].size(); ++i)
	{
		if (!visited[i])
		{
			s1 += gamma * transPixel(i) * img0(neighMat[iPixel][i]);
		}
	}
	s1 -= img1(iPixel);
	s1 *= s1;
	return s1;
}

double OfIterfilterPixelObj (VectorXd &img0, VectorXd &img1, SparseMatrix<double> &transMat, int iPixel, double &gamma, vector<vector<int> > &neighMat, VectorXi &visited, VectorXd &transPixel,
							 int reserveCap, double lambda_1, SparseMatrix<double> &DcompMat)
{
	double s1 = 0;
	double dtsum = 0;
	vector<int> nonzeroInd;
	vector<int> nonZeroVal;
	nonzeroInd.reserve(reserveCap);
	nonZeroVal.reserve(reserveCap);
	for (SparseMatrix<double>::InnerIterator it(DcompMat, iPixel); it; ++it)
	{
		int irow = it.row();
		nonzeroInd.push_back(irow);
		nonZeroVal.push_back(it.value());
	}

	for (SparseMatrix<double>::InnerIterator it(transMat, iPixel); it; ++it)
	{
		int irow = it.row();
		int ind = -1;
		int posMark = bSearch(nonzeroInd, irow);
		if (IndexSearch(irow, neighMat, iPixel, ind))
		{
			visited[ind] = 1;
			s1 += ((1 - gamma) * it.value() + gamma * transPixel(ind)) * img0(irow);
			if (posMark != -1)
			{
				dtsum += ((1 - gamma) * it.value() + gamma * transPixel(ind)) * nonZeroVal[posMark];
			}
		}
		else
		{
			s1 += (1 - gamma) * it.value() * img0(irow);
			if (posMark != -1)
			{
				dtsum += (1 - gamma) * it.value() * nonZeroVal[posMark];
			}
		}
	}
	for (int i = 0; i != neighMat[iPixel].size(); ++i)
	{
		if (!visited[i])
		{
			s1 += gamma * transPixel(i) * img0(neighMat[iPixel][i]);
			int posMark = bSearch(nonzeroInd, neighMat[iPixel][i]);
			if (posMark != -1)
			{
				dtsum += gamma * transPixel(i) * nonZeroVal[posMark];
			}
		}
	}
	s1 -= img1(iPixel);
	s1 *= s1;
	dtsum *= lambda_1;
	s1 += dtsum;
	return s1;
}

void ComputerFlow(Mat &img1, Mat &img2, vector<Mat> &flowAryX, vector<Mat> &flowAryY)
{

}

void NormalizeImg(Mat &img, float minP, float maxP)
{
	float minVal = INT_MAX;
	float maxVal = INT_MIN;
	int rows = img.rows;
	int cols = img.cols;
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			if (minVal > img.at<float>(i, j))
			{
				minVal = img.at<float>(i, j);
			}
			if (maxVal < img.at<float>(i, j))
			{
				maxVal = img.at<float>(i, j);
			}
		}
	}
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			img.at<float>(i, j) = (img.at<float>(i, j) - minVal) / (maxVal - minVal) * (maxP - minP) + minP;
		}
	}
}

void ScaleImg(Mat &img1, Mat &img2, double minP, double maxP)
{
	double minVal = INT_MAX;
	double maxVal = INT_MIN;
	int rows = img1.rows;
	int cols = img1.cols;
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			if (minVal > img1.at<double>(i, j))
			{
				minVal = img1.at<double>(i, j);
			}
			if (minVal > img2.at<double>(i, j))
			{
				minVal = img2.at<double>(i, j);
			}
			if (maxVal < img1.at<double>(i, j))
			{
				maxVal = img1.at<double>(i, j);
			}
			if (maxVal < img2.at<double>(i, j))
			{
				maxVal = img2.at<double>(i, j);
			}
		}
	}
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			img1.at<double>(i, j) = (img1.at<double>(i, j) - minVal) / (maxVal - minVal) * (maxP - minP) + minP;
			img2.at<double>(i, j) = (img2.at<double>(i, j) - minVal) / (maxVal - minVal) * (maxP - minP) + minP;
		}
	}
}

double getTransElement(SparseMatrix<double> &transMat, int iPixel, int ipx, vector<vector<int> > &neighMat)
{
	double elem = 0;
	for (SparseMatrix<double>::InnerIterator it(transMat, iPixel); it; ++it)
	{
		int irow = it.row();
		if (irow == neighMat[iPixel][ipx])
		{
			return elem;
		}
	}
	return elem;
}

double denomTransMat(SparseMatrix<double> &transMat, int iPixel, vector<vector<int> > &neighMat, VectorXd &transPixel, VectorXd &gradAry)
{
	double denom = 0.0;
	for (int ipx = 0; ipx != neighMat[iPixel].size(); ++ipx)
	{
		denom += gradAry[neighMat[iPixel][ipx]] * (-getTransElement(transMat, iPixel, ipx, neighMat) + transPixel(ipx));
	}
	return denom;
}

//int main(int argc, char *argv[])
//{
//	ifstream input("Centroid.txt");
//	int rows = 584 / 4;
//	int cols = 388 / 4 ;
//	int nPixels = rows * cols;
//	MatrixXd centroidMat = MatrixXd::Zero(nPixels, 2);
//	for (int i = 0; i != nPixels; ++i)
//	{
//		input >> centroidMat(i, 0);
//		input >> centroidMat(i, 1);
//	}
//
//	
//	cv::Mat flowX = Mat::zeros(rows, cols, CV_32FC1);
//	cv::Mat flowY = Mat::zeros(rows, cols, CV_32FC1);
//	//cv::Mat flowMag = Mat::zeros(rows, cols, CV_32FC1);
//	//cv::Mat flowAng = Mat::zeros(rows, cols, CV_32FC1);
//
//	cv::Mat fX = Mat::zeros(cols, rows, CV_32FC1);
//	cv::Mat fY = Mat::zeros(cols, rows, CV_32FC1);
//	cv::Mat flowMag = Mat::zeros(cols, rows, CV_32FC1);
//	cv::Mat flowAng = Mat::zeros(cols, rows, CV_32FC1);
//
//	for (int i = 0; i != nPixels; ++i)
//	{
//		int pRow = i / cols;
//		int pCol = i % cols;
//		flowX.at<float>(rows - pRow - 1, pCol) = centroidMat(i,0);
//		flowY.at<float>(rows - pRow - 1, pCol) = centroidMat(i,1);
//	}
//
//	for (int i = 0; i != rows; ++i)
//	{
//		for (int j = 0; j != cols; ++j)
//		{
//			fX.at<float>(j, rows - i - 1) = flowX.at<float>(i, j);
//			fY.at<float>(j, rows - i - 1) = flowY.at<float>(i, j);
//		}
//	}
//	//NormalizeImg(flowX, 0.0, 1.0);
//	//NormalizeImg(flowY, 0.0, 1.0);
//	cartToPolar(fX, fY, flowMag, flowAng, true);
//	double magMax = 0;
//	minMaxLoc(flowMag, 0, &magMax);
//	flowMag.convertTo(flowMag, -1, 1.0 / magMax);
//
//	cv::Mat hsvMat[3];
//	cv::Mat hsvM;
//	vector<Mat> channels;
//	channels.push_back(flowAng);
//	channels.push_back(Mat::ones(flowAng.size(), CV_32F));
//	channels.push_back(flowMag);
//	//hsvMat[0] = flowAng;
//	//hsvMat[1] = Mat::ones(flowAng.size(), CV_64F);
//	//hsvMat[1] = flowMag;
//	//cv::merge(hsvMat, 3, hsvM);
//	merge(channels, hsvM);
//	//convert to BGR and show
//	Mat bgr; 
//	cv::cvtColor(hsvM, bgr, cv::COLOR_HSV2BGR);
//	cv::imshow("optical flow", bgr);
//	imwrite("AFilterFlow.jpg", bgr);
//	CFloatImage cFlow(cols, rows, 2);
//	for (int i = 0; i < cols; ++i)
//	{
//		for (int j = 0; j < rows; ++j)
//		{
//			cFlow.Pixel(i, j, 0) = fX.at<float>(i, j);
//			cFlow.Pixel(i, j, 1) = fY.at<float>(i, j);
//		}
//	}
//	CByteImage cImage;
//	MotionToColor(cFlow, cImage, 1);
//	cv::Mat image(cols, rows, CV_8UC3, cv::Scalar(0, 0, 0));
//	// Compute back to cv::Mat with 3 channels in BGR:
//	for (int i = 0; i < cols; i++) {
//		for (int j = 0; j < rows; j++) {
//			image.at<cv::Vec3b>(i, j)[0] = cImage.Pixel(i, j, 0);
//			image.at<cv::Vec3b>(i, j)[1] = cImage.Pixel(i, j, 1);
//			image.at<cv::Vec3b>(i, j)[2] = cImage.Pixel(i, j, 2);
//		}
//	}
//	cv::imshow("Filter flow", image);
//	imwrite("FilterFlow.jpg", image);
//	waitKey();
//	return 0;
//}

int main(int argc, char *argv[])
{
	cv::Mat img[2];
	cv::Mat dbImg[2];
	cv::Mat scaImg[2];
	cv::Mat imgArray[2];
	cv::Mat tpImg[2];
	const double epison = 1e-6;
	const double appZero = 1e-10;
	double scaler = 1.0 / 32;
	img[0] = cv::imread("./eval-data-gray/Grove2/frame10.png", CV_LOAD_IMAGE_GRAYSCALE);
	img[1] = cv::imread("./eval-data-gray/Grove2/frame11.png", CV_LOAD_IMAGE_GRAYSCALE);
	img[0].convertTo(dbImg[0], CV_64F, 1.0 / 255.0);
	img[1].convertTo(dbImg[1], CV_64F, 1.0 / 255.0);
	resize(dbImg[0], scaImg[0], Size(), scaler, scaler, INTER_CUBIC);
	resize(dbImg[1], scaImg[1], Size(), scaler, scaler, INTER_CUBIC);
	int rows = scaImg[0].rows;
	int cols = scaImg[1].cols;

	ifstream input("CentroidAlbedo_0040.txt");
	int nPixels = rows * cols;
	MatrixXd centroidMat = MatrixXd::Zero(nPixels, 2);
	for (int i = 0; i != nPixels; ++i)
	{
		input >> centroidMat(i, 0);
		input >> centroidMat(i, 1);
	}

	imshow("Scaled image 1", scaImg[0]);
	imshow("Scaled image 2", scaImg[1]);
	int nImgs = 2;
	VectorXd scalImgAry[2];
	scalImgAry[0].resize(rows * cols);
	scalImgAry[1].resize(rows * cols);
	double sumPixel = 0;
	for (int i = 0; i != nImgs; ++i)
	{
		imgMatToArray(scaImg[i], scalImgAry[i]);
	}
	//int nPixels = rows * cols;
	cout << "rows "  << rows << endl;
	cout << "cols " << cols << endl;
	SparseMatrix <double> transMat(nPixels, nPixels);
	int neigbWidth = 25;
	int neigbHeight = 25;
	int reserveCap = 4 * (neigbHeight + 2) * (neigbWidth + 2);
	transMat.reserve(VectorXd::Constant(nPixels,  reserveCap));
	SparseMatrix<double> dcompMat(nPixels, nPixels);

	//MatrixXd mBarPixMat(nPixels, 2);
	vector<vector<int> > neighMat;

	int affineX = 2;
	int affineY = 3;
	int affineEntry = affineX * affineY;
	MatrixXd AffineMat = MatrixXd::Zero(nPixels, affineEntry);
	MatrixXd tmpAffineMat = MatrixXd::Zero(nPixels, affineEntry);
	dcompMat.reserve(VectorXd::Constant(nPixels,  reserveCap));
	neighborMatrix(scaImg[0], neighMat, neigbWidth, neigbHeight);

	double alpha = 0.5;
	int iterMax = 100;
	int iterMaxN = 100;
	int outIterMax = 1;
	const int GammaCon = 7000;
	vector<vector<double> > funVal(iterMax, vector<double> (nPixels, 0));
	vector<vector<double> > dualGap(iterMax, vector<double> (nPixels, 0));
	VectorXd gradAry(nPixels);
	Vector3d pHomog(0, 0, 1);
	Vector2d affHomog(0, 0);
	Vector2d tmpAffine(0, 0);
	Vector2d gradAffine(0, 0);
	MatrixXd gradApix[4];
	gradApix[0].resize(affineX, affineY);
	gradApix[1].resize(affineX, affineY);
	gradApix[2].resize(affineX, affineY);
	gradApix[3].resize(affineX, affineY);
	VectorXd tGranApix2(affineX * affineY);
	VectorXd tGranApix3(affineX * affineY);
	VectorXd affineNeighSum(affineX * affineY);
	//MatrixXd centroidMat = MatrixXd::Zero(nPixels, 2);
	int apixNorm = 10;
	double ptrFlowX = 0;
	double ptrFlowY = 0;
	int pfx = 0;
	int pfy = 0;
	for (int i = 0; i != nPixels; ++i)
	{
		transMat.insert(i, i) = 1;
	}

	double lambda_1 = 1;
	double lambda_3 = 0.005;
	double lambda_2 = 0;
	int ia[8000];
	int ja[8000];
	double ar[8000];
	char rName[3];
	char cName[4000][10];
	rName[0] = 'c';
	rName[2] = '\0';
	rName[1] = '\0';
	memset(cName, '\0', sizeof(cName));
	for (int i = 0; i != reserveCap; ++i)
	{
		itoa(i + 1, cName[i], 10);
	}

	vector<vector<double> > funcVal(outIterMax, vector<double>(iterMaxN, 0));
	vector<vector<double> > fixAFuncVal(outIterMax, vector<double>(iterMax, 0));
	vector<vector<double> > fixTFuncVal(outIterMax, vector<double>(iterMax, 0));
	vector<vector<double> > finalFuncVal(outIterMax, vector<double>(iterMax, 0));
	vector<double> tuningSeq(nPixels, 0);
	vector<double> pixelVal(nPixels, 0);

	string outputFn("PFilterFlow_");
	outputFn += std::to_string(lambda_3);
	outputFn += "_";
	outputFn += std::to_string(lambda_2);
	outputFn += "_";
	outputFn += std::to_string(lambda_1);
	outputFn += "_";
	outputFn += std::to_string(outIterMax);
	outputFn += "_";
	outputFn += std::to_string(iterMaxN);
	outputFn += "_";
	outputFn += std::to_string(GammaCon);
	outputFn += ".jpg";
	std::cout << "Output image is " << outputFn << endl;

	//Preallocate 
	VectorXd transPix(reserveCap);
	MatrixXi eqnLhsMat = MatrixXi::Zero(3, affineEntry + reserveCap);

	//Obtain initial filter flow
	for (int kPixel = 0; kPixel != nPixels; ++kPixel)
	{
		int pr = kPixel / cols;
		int pc = kPixel % cols;

		//Solve sub problems using a linear programming solver
		glp_prob *mip = glp_create_prob();
		glp_term_out(GLP_OFF);
		glp_set_prob_name(mip, "InitialT");
		glp_set_obj_dir(mip, GLP_MIN);
		glp_add_rows(mip, 3);

		for (int ipx = 0; ipx != neighMat[kPixel].size(); ++ipx)
		{
			int ir = neighMat[kPixel][ipx] / cols;
			int ic = neighMat[kPixel][ipx] % cols;
			eqnLhsMat(0, ipx) = 1;
			eqnLhsMat(1, ipx) = ir - pr;
			eqnLhsMat(2, ipx) = ic - pc;
		}
		rName[0] = 'c';
		rName[1] = '1';
		glp_set_row_name(mip, 1, rName);
		glp_set_row_bnds(mip, 1, GLP_FX, 1.0, 1.0);
		rName[1] = '2';
		glp_set_row_name(mip, 2, rName);
		glp_set_row_bnds(mip, 2, GLP_FX, centroidMat(kPixel, 0), centroidMat(kPixel, 0));
		rName[2] = '3';
		glp_set_row_name(mip, 3, rName);
		glp_set_row_bnds(mip, 3, GLP_FX, centroidMat(kPixel, 1), centroidMat(kPixel, 1));

		glp_add_cols(mip, neighMat[kPixel].size());
		for (int ipx = 0; ipx != neighMat[kPixel].size(); ++ipx)
		{
			glp_set_col_name(mip, ipx + 1, cName[ipx]);
			glp_set_col_bnds(mip, ipx + 1, GLP_DB, 0.0, 1.0);
			glp_set_obj_coef(mip, ipx + 1, 0);
		}
		int cntIter = 0;
		for (int ipx = 0; ipx != 3; ++ipx)
		{
			for (int jpx = 0; jpx != neighMat[kPixel].size(); ++jpx)
			{
				if (eqnLhsMat(ipx, jpx) != 0)
				{
					++cntIter;
					ia[cntIter] = ipx + 1;
					ja[cntIter] = jpx + 1;
					ar[cntIter] = eqnLhsMat(ipx, jpx);
				}
			}
		}
		glp_load_matrix(mip, cntIter, ia, ja, ar);
		glp_simplex(mip, NULL);

		for (int ipx = 0; ipx != neighMat[kPixel].size(); ++ipx)
		{
			transPix(ipx) = glp_get_col_prim(mip, ipx + 1);
			//cout << transPix(ipx) << endl;
		}
		glp_delete_prob(mip);

		for (int iPixel = 0; iPixel != neighMat[kPixel].size(); ++iPixel)
		{
			if (transPix(iPixel) != 0)
			{
				transMat.coeffRef(neighMat[kPixel][iPixel], kPixel) = transPix(iPixel);
			}
		} 
	}

	// Obtain Affine matrix
	MatrixXi eqnLhsMata = MatrixXi::Zero(2, affineEntry);
	for (int kPixel = 0; kPixel != nPixels; ++kPixel)
	{
		pixelVal[kPixel] = 0;
		int pr = kPixel / cols;
		int pc = kPixel % cols;
		//Solve sub problems using a linear programming solver
		glp_prob *mip = glp_create_prob();
		glp_term_out(GLP_OFF);
		glp_set_prob_name(mip, "InitialA");
		glp_set_obj_dir(mip, GLP_MIN);
		glp_add_rows(mip, 2);

		eqnLhsMata(0, 0) = -pr;
		eqnLhsMata(0, 1) = -pc;
		eqnLhsMata(0, 2) = -1;
		eqnLhsMata(0, 3) = 0;
		eqnLhsMata(0, 4) = 0;
		eqnLhsMata(0, 5) = 0;
		eqnLhsMata(1, 0) = 0;
		eqnLhsMata(1, 1) = 0;
		eqnLhsMata(1, 2) = 0;
		eqnLhsMata(1, 3) = -pr;
		eqnLhsMata(1, 4) = -pc;
		eqnLhsMata(1, 5) = -1;

		rName[0] = 'c';
		for (int ipx = 0; ipx != 2; ++ipx)
		{
			rName[1] = ipx + 1 + '0';
			glp_set_row_name(mip, ipx + 1, rName);
			glp_set_row_bnds(mip, ipx + 1, GLP_FX, 0.0, 0);
		}

		rName[0] = 'x';
		glp_add_cols(mip, affineEntry);
		for (int ipx = 0; ipx != affineEntry; ++ipx)
		{
			rName[1] = ipx + 1 + '0';
			glp_set_col_name(mip, ipx + 1, rName);
			glp_set_col_bnds(mip, ipx + 1, GLP_DB, -25.0, 25.0);
			glp_set_obj_coef(mip, ipx + 1, 0);
		}

		int cntIter = 0;
		for (int ipx = 0; ipx != 2; ++ipx)
		{
			for (int jpx = 0; jpx != affineEntry; ++jpx)
			{
				if (eqnLhsMata(ipx, jpx) != 0)
				{
					++cntIter;
					ia[cntIter] = ipx + 1;
					ja[cntIter] = jpx + 1;
					ar[cntIter] = eqnLhsMata(ipx, jpx);
				}
			}
		}
		glp_load_matrix(mip, cntIter, ia, ja, ar);
		glp_simplex(mip, NULL);
		for (int ipx = 0; ipx != affineEntry; ++ipx)
		{
			tGranApix2(ipx) = glp_get_col_prim(mip, ipx + 1);
		}
		glp_delete_prob(mip);
		AffineMat.row(kPixel) = tGranApix2.transpose();
	}

	VectorXi visited(reserveCap);
	for (int i = 0; i != outIterMax; ++i)
	{
		int st1 = clock();
		for (int j = 0; j != iterMaxN; ++j)
		{
#pragma  omp parallel for
			for (int kPixel = 0; kPixel != nPixels; ++kPixel)
			{
				pixelVal[kPixel] = 0;
				int pr = kPixel / cols;
				int pc = kPixel % cols;
				pHomog[0] = pr;
				pHomog[1] = pc;
				double iTrans = multiplyDenseSparse(scalImgAry[0], transMat, kPixel);
				gradAry = 2 * (iTrans - scalImgAry[1][kPixel]) * scalImgAry[0];
				addDenseSparse(gradAry, dcompMat, kPixel, lambda_1);
				for (int k = 0; k != tGranApix2.size(); ++k)
				{
					tGranApix2(k) = 0;
					affineNeighSum(k) = 0;
				}
				for (int k = 0; k != neighMat[kPixel].size(); ++k)
				{
					tGranApix2 += AffineMat.row(neighMat[kPixel][k]).transpose();
					affineNeighSum += AffineMat.row(neighMat[kPixel][k]).transpose();
				}
				double tGran = AffineMat.row(kPixel) * tGranApix2;
				tGranApix2 = 2 * (AffineMat.row(kPixel).transpose() * neighMat[kPixel].size() - tGranApix2) * lambda_3;
				for (int ki = 0; ki != affineX; ++ki)
				{
					for (int kj = 0; kj != affineY; ++kj)
					{
						gradApix[2](ki, kj) = tGranApix2(ki * affineY + kj);
					}
				}

				//Solve sub problems using a linear programming solver
				glp_prob *mip = glp_create_prob();
				glp_term_out(GLP_OFF);
				glp_set_prob_name(mip, "short");
				glp_set_obj_dir(mip, GLP_MIN);
				glp_add_rows(mip, 3);
				eqnLhsMat(0, 0) = -pr;
				eqnLhsMat(0, 1) = -pc;
				eqnLhsMat(0, 2) = -1;
				eqnLhsMat(0, 3) = 0;
				eqnLhsMat(0, 4) = 0;
				eqnLhsMat(0, 5) = 0;
				eqnLhsMat(1, 0) = 0;
				eqnLhsMat(1, 1) = 0;
				eqnLhsMat(1, 2) = 0;
				eqnLhsMat(1, 3) = -pr;
				eqnLhsMat(1, 4) = -pc;
				eqnLhsMat(1, 5) = -1;
				for (int ipx = 0; ipx != neighMat[kPixel].size(); ++ipx)
				{
					int ir = neighMat[kPixel][ipx] / cols;
					int ic = neighMat[kPixel][ipx] % cols;
					eqnLhsMat(0, ipx + affineEntry) = ir - pr;
					eqnLhsMat(1, ipx + affineEntry) = ic - pc;
					eqnLhsMat(2, ipx + affineEntry) = 1;
				}
				rName[0] = 'c';
				for (int ipx = 0; ipx != 2; ++ipx)
				{
					rName[1] = ipx + 1 + '0';
					glp_set_row_name(mip, ipx + 1, rName);
					glp_set_row_bnds(mip, ipx + 1, GLP_FX, 0.0, 0);
				}
				rName[1] = '3';
				glp_set_row_name(mip, 3, rName);
				glp_set_row_bnds(mip, 3, GLP_FX, 1.0, 1);

				rName[0] = 'x';
				glp_add_cols(mip, affineEntry + neighMat[kPixel].size());
				for (int ipx = 0; ipx != affineEntry; ++ipx)
				{
					rName[1] = ipx + 1 + '0';
					glp_set_col_name(mip, ipx + 1, rName);
					glp_set_col_bnds(mip, ipx + 1, GLP_DB, -25.0, 25.0);
					glp_set_obj_coef(mip, ipx + 1, tGranApix2(ipx));
				}
				for (int ipx = 0; ipx != neighMat[kPixel].size(); ++ipx)
				{
					glp_set_col_name(mip, ipx + 1 + affineEntry, cName[ipx + affineEntry + 1]);
					glp_set_col_bnds(mip, ipx + 1 + affineEntry, GLP_DB, 0.0, 1.0);
					glp_set_obj_coef(mip, ipx + 1 + affineEntry, gradAry[neighMat[kPixel][ipx]]);
				}
				int cntIter = 0;
				for (int ipx = 0; ipx != 3; ++ipx)
				{
					for (int jpx = 0; jpx != affineEntry + neighMat[kPixel].size(); ++jpx)
					{
						if (eqnLhsMat(ipx, jpx) != 0)
						{
							++cntIter;
							ia[cntIter] = ipx + 1;
							ja[cntIter] = jpx + 1;
							ar[cntIter] = eqnLhsMat(ipx, jpx);
						}
					}
				}
				glp_load_matrix(mip, cntIter, ia, ja, ar);
				glp_simplex(mip, NULL);

				for (int ipx = 0; ipx != affineEntry; ++ipx)
				{
					tGranApix2(ipx) = glp_get_col_prim(mip, ipx + 1);
					//cout << tGranApix2(ipx) << endl;
				}

				for (int ipx = 0; ipx != neighMat[kPixel].size(); ++ipx)
				{
					transPix(ipx) = glp_get_col_prim(mip, ipx + affineEntry + 1);
					//cout << transPix(ipx) << endl;
				}
				glp_delete_prob(mip);

				//Search optimial gamma
				visited.setZero(neighMat[kPixel].size());
				double gamma = 1;
				gamma = 2.0 /(GammaCon + (2 + (j + 1))); 
				double term1 = 0;
				double term2 = 0;
				double term3 = 0;
				for (int k = 0; k != neighMat[kPixel].size(); ++k)
				{
					term3 += AffineMat.row(neighMat[kPixel][k]) * AffineMat.row(neighMat[kPixel][k]).transpose();
				}

				term1 = neighMat[kPixel].size() * AffineMat.row(kPixel).squaredNorm();
				term2 = -2 * AffineMat.row(kPixel) * affineNeighSum;
				//double curVal = totalFilterPixelObj(scalImgAry[0], scalImgAry[1], transMat, kPixel);
				double curVal = OfTotalfilterPixelObj(scalImgAry[0], scalImgAry[1], transMat, kPixel, reserveCap, lambda_1, dcompMat);
				curVal += lambda_3 * (term1 + term2 + term3);
				pixelVal[kPixel] = curVal;

				visited.setZero(neighMat[kPixel].size());
				for (SparseMatrix<double>::InnerIterator it(transMat, kPixel); it; ++it)
				{
					int irow = it.row();
					int ind = -1;
					if (IndexSearch(irow, neighMat, kPixel, ind))
					{
						visited[ind] = 1;
						transMat.coeffRef(irow, kPixel) = ((1 - gamma) * it.value() + gamma * transPix(ind));
					}
					else
					{
						transMat.coeffRef(irow, kPixel) = (1 - gamma) * it.value();
					}
				}
				for (int iPixel = 0; iPixel != neighMat[kPixel].size(); ++iPixel)
				{
					if (!visited[iPixel])
					{
						transMat.coeffRef(neighMat[kPixel][iPixel], kPixel) = gamma * transPix(iPixel);
					}
				}
				AffineMat.row(kPixel) = (1 - gamma) * AffineMat.row(kPixel) + gamma * tGranApix2.transpose();
			}
			double fval = 0;
			for (int kpixel = 0; kpixel != nPixels; ++kpixel)
			{
				fval += pixelVal[kpixel];
			}
			//cout << fval << endl;
			funcVal[i][j] = fval;
		}

		int stp1 = clock();
		cout << "time: " << (stp1 - st1) * 1.0 /(CLOCKS_PER_SEC) * 1000 << endl;
		int st = clock();
		int r = 0;
		int c = 0;
		//#pragma  omp parallel for
		for (int kPixel = 0; kPixel != nPixels; ++kPixel)
		{
			double centroidx = 0;
			double centroidy = 0;
			int pr = kPixel / cols;
			int pc = kPixel % cols;
			for (SparseMatrix<double>::InnerIterator it(transMat, kPixel); it; ++it)
			{
				int irow = it.row();
				int ir = irow / cols;
				int ic = irow % cols;
				centroidx += (ir - pr) * it.value();
				centroidy += (ic - pc) * it.value();
			}
			for (int id = 0; id != neighMat[kPixel].size(); ++id)
			{
				r = neighMat[kPixel][id] / cols;
				c = neighMat[kPixel][id] % cols;
				dcompMat.coeffRef(neighMat[kPixel][id], kPixel) = ((r - pr - centroidx) * (r - pr -centroidx) + (c - pc - centroidy) * (c - pc - centroidy)); 
			}
		}
		int stp = clock();
		cout << "time: " << (stp - st) * 1.0 /(CLOCKS_PER_SEC) * 1000 << endl;
	}

	ofstream objFile;
	objFile.open("CentroidMat.txt");
	//Centroid
	for (int kPixel = 0; kPixel != nPixels; ++kPixel)
	{
		int pr = kPixel / cols;
		int pc = kPixel % cols;
		double centroidx = 0;
		double centroidy = 0;
		for (SparseMatrix<double>::InnerIterator it(transMat, kPixel); it; ++it)
		{
			int irow = it.row();
			if (binarySearch(irow, neighMat, kPixel))
			{
				int ir = irow / cols;
				int ic = irow % cols;
				centroidx += (ir - pr) * it.value();
				centroidy += (ic - pc) * it.value();
			}
		}
		centroidMat(kPixel, 0) = centroidx;
		centroidMat(kPixel, 1) = centroidy;
		objFile << centroidx << " " << centroidy << endl;
	}
	objFile.close();

	objFile.open("step1ObjVal.txt");
	for (int i = 0; i != outIterMax; ++i)
	{
		for (int j = 0; j != iterMaxN; ++j)
		{
			objFile << funcVal[i][j] << " ";
		}
		objFile << endl;
	}
	objFile.close();

	objFile.open("step2ObjVal.txt");
	for (int i = 0; i != outIterMax; ++i)
	{
		for (int j = 0; j != iterMax; ++j)
		{
			objFile << fixAFuncVal[i][j] << " ";
		}
		objFile << endl;
	}
	objFile.close();

	objFile.open("step3ObjVal.txt");
	for (int i = 0; i != outIterMax; ++i)
	{
		for (int j = 0; j != iterMax; ++j)
		{
			objFile << fixTFuncVal[i][j] << " ";
		}
		objFile << endl;
	}
	objFile.close();

	objFile.open("step4ObjVal.txt");
	for (int i = 0; i != outIterMax; ++i)
	{
		for (int j = 0; j != iterMax; ++j)
		{
			objFile << finalFuncVal[i][j] << " ";
		}
		objFile << endl;
	}
	objFile.close();

	VectorXd tranImg(nPixels);
	cv::Mat errImg(rows, cols, CV_64F);
	cv::Mat tImg(rows, cols, CV_64F);
	for (int i = 0; i != nPixels; ++i)
	{
		tranImg(i) = 0;
		for (SparseMatrix<double>::InnerIterator it(transMat, i); it; ++it)
		{
			tranImg(i) += it.value() * scalImgAry[0](it.row());
		}
		int pRow = i / cols;
		int pCol = i % cols;
		errImg.at<double>(pRow, pCol) = tranImg(i) - scalImgAry[1](i);
		tImg.at<double>(pRow, pCol) = tranImg(i);
	}
	//NormalizeImg(tImg, 0.0, 1.0);
	imshow("Transform Image", tImg);
	imwrite("TransformImg.jpg", tImg);
	//NormalizeImg(errImg, 0.0, 1.0);
	imshow("Error Image", errImg);
	imwrite("ErrorImg.jpg", errImg);
	cv::Mat flowX = Mat::zeros(rows, cols, CV_32FC1);
	cv::Mat flowY = Mat::zeros(rows, cols, CV_32FC1);
	cv::Mat flowMag = Mat::zeros(rows, cols, CV_32FC1);
	cv::Mat flowAng = Mat::zeros(rows, cols, CV_32FC1);

	for (int i = 0; i != nPixels; ++i)
	{
		int pRow = i / cols;
		int pCol = i % cols;
		flowX.at<float>(pRow, pCol) = centroidMat(i,0);
		flowY.at<float>(pRow, pCol) = centroidMat(i,1);
	}
	ofstream ofFile;
	ofFile.open("OpticFlowsX1b4.txt");
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			ofFile << flowX.at<float>(i, j) << " ";
		}
		ofFile << endl;
	}
	ofFile.close();
	ofFile.open("OpticFlowsY1b4.txt");
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			ofFile << flowY.at<float>(i, j) << " ";
		}
		ofFile << endl;
	}
	ofFile.close();
	ofFile.open("ErrorImg.txt");
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			ofFile << errImg.at<double>(i, j) << " ";
		}
		ofFile << endl;
	}
	ofFile.close();
	ofFile.open("TranImg.txt");
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			ofFile << tImg.at<double>(i, j) << " ";
		}
		ofFile << endl;
	}
	ofFile.close();
	//NormalizeImg(flowX, 0.0, 1.0);
	//NormalizeImg(flowY, 0.0, 1.0);
	cartToPolar(flowX, flowY, flowMag, flowAng, true);
	double magMax = 0;
	minMaxLoc(flowMag, 0, &magMax);
	flowMag.convertTo(flowMag, -1, 1.0 / magMax);

	cv::Mat hsvMat[3];
	cv::Mat hsvM;
	vector<Mat> channels;
	channels.push_back(flowAng);
	channels.push_back(Mat::ones(flowAng.size(), CV_32F));
	channels.push_back(flowMag);
	//hsvMat[0] = flowAng;
	//hsvMat[1] = Mat::ones(flowAng.size(), CV_64F);
	//hsvMat[1] = flowMag;
	//cv::merge(hsvMat, 3, hsvM);
	merge(channels, hsvM);
	//convert to BGR and show
	Mat bgr; 
	cv::cvtColor(hsvM, bgr, cv::COLOR_HSV2BGR);
	cv::imshow("optical flow", bgr);
	imwrite("AFilterFlow.jpg", bgr);
	CFloatImage cFlow(cols, rows, 2);
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			cFlow.Pixel(j, i, 0) = flowX.at<float>(i, j);
			cFlow.Pixel(j, i, 1) = flowY.at<float>(i, j);
		}
	}
	CByteImage cImage;
	MotionToColor(cFlow, cImage, 1);
	cv::Mat image(rows, cols, CV_8UC3, cv::Scalar(0, 0, 0));
	// Compute back to cv::Mat with 3 channels in BGR:
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			image.at<cv::Vec3b>(i, j)[0] = cImage.Pixel(j, i, 0);
			image.at<cv::Vec3b>(i, j)[1] = cImage.Pixel(j, i, 1);
			image.at<cv::Vec3b>(i, j)[2] = cImage.Pixel(j, i, 2);
		}
	}
	cv::imshow("Filter flow", image);
	imwrite(outputFn.c_str(), image);
	waitKey();
	return 0;
}