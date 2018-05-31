#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <fstream>
#include "cv.h"
#include "highgui.h"
#include "CannyLine.h"

using namespace cv;
using namespace std;

void pre(Mat &img)
{
	//resize(img, img, Size(img.cols / 2, img.rows / 2), 0, 0, INTER_LANCZOS4);
	Mat grayImg;
	cvtColor(img, grayImg, CV_BGR2GRAY);
	//namedWindow("cannyline", 2);
	//imshow("cannyline", grayImg);
	Mat bw;
	adaptiveThreshold(~grayImg, bw, 255, CV_ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, -2);
	//namedWindow("orign", 2);
	//imshow("orign", bw);
	//CvMemStorage *contourStorage = cvCreateMemStorage();
	vector<vector<Point>> contours1;
	findContours(bw, contours1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	drawContours(bw, contours1, -1, Scalar(255, 255, 255));
	printf("%d\n", contours1.size());
	//namedWindow("draw", 2);
	//imshow("draw", bw);
	//使用二值化后的图像来获取表格横纵的线
	Mat horizontal = bw.clone();
	Mat vertical = bw.clone();
	int scale1 = 22; //这个值越大，检测到水平线越多
	int horizontalsize = horizontal.cols / scale1;

	// 为了获取横向的表格线，设置腐蚀和膨胀的操作区域为一个比较大的横向直条
	Mat horizontalStructure = getStructuringElement(MORPH_RECT, Size(horizontalsize, 1));

	// 先腐蚀再膨胀
	erode(horizontal, horizontal, horizontalStructure, Point(-1, -1));
	dilate(horizontal, horizontal, horizontalStructure, Point(-1, -1));
	//namedWindow("horizontal", 2);
	//imshow("horizontal", horizontal);
	int scale2 = 65;      //越大，检测到的垂直直线越多
	int verticalsize = vertical.rows / scale2;
	Mat verticalStructure = getStructuringElement(MORPH_RECT, Size(1, verticalsize));
	erode(vertical, vertical, verticalStructure, Point(-1, -1));
	dilate(vertical, vertical, verticalStructure, Point(-1, -1));
	//namedWindow("vertical", 2);
	//imshow("vertical", vertical);
	//Mat joints;
	//bitwise_and(horizontal, vertical, joints);
	//namedWindow("joints", 2);
	//imshow("joints", joints);
	Mat mask = horizontal + vertical;
	img = mask.clone();
	//namedWindow("mask", 2);
	//imshow("mask", mask);
	vector<Vec4i> hierarchy;
	std::vector<std::vector<cv::Point> > contours2;
	cv::findContours(mask, contours2, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	vector<vector<Point> > contours_poly(contours2.size());
	vector<Rect> boundRect(contours2.size());
	vector<Mat> rois;

	cv::waitKey(0);
}
float haiLun(float x1, float y1, float x2, float y2, float x3, float y3)
{
	float S;
	return S = float(1 / 2)*(x1*y2 + x2 * y3 + x3 * y1 - x1 * y3 - x2 * y1 - x3 * y2);
}
bool equ(float x1, float y1, float x2, float y2, float x3, float y3)
{
	if ((x1 - x2) == 0 && (x1 - x3) == 0)
		return 1;
	else if (((y1 - y2) / (x1 - x2)) == ((y1 - y3) / (x1 - x3)))
		return 1;
	else
		return 0;
}
float dist(float x1, float y1, float x2, float y2, float x3, float y3)
{
	float A, B, C;
	A = y1 - y2;
	B = x2 - x1;
	C = x1 * y2 - x2 * y1;
	return (abs(A*x3 + B * y3 + C) / (sqrt(A*A + B * B)));
}

Mat VerticalProjection(Mat srcImage, int *&colsBlack)//垂直积分投影  
{
	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION); //PNG格式图片的压缩级别  
	compression_params.push_back(9);  //这里设置保存的图像质量级别
	if (srcImage.channels() > 1)
		cvtColor(srcImage, srcImage, CV_RGB2GRAY);
	Mat srcImageBin;
	int *colswidth = new int[srcImage.cols];  //申请src.image.cols个int型的内存空间  
	memset(colswidth, 0, srcImage.cols * 4);  //数组必须赋初值为零，否则出错。无法遍历数组。  
	threshold(srcImage, srcImageBin, 120, 255, CV_THRESH_BINARY_INV);
	//imshow("二值图", srcImageBin);
	int value;
	for (int i = 0; i < srcImage.cols; i++)
		for (int j = 0; j < srcImage.rows; j++)
		{
			//value=cvGet2D(src,j,i);  
			value = srcImageBin.at<uchar>(j, i);
			if (value == 255)
			{
				colswidth[i]++; //统计每列的白色像素点    
			}
			else
			{
				colsBlack[i]++;//统计每列的黑色像素点
			}
		}
	Mat histogramImage(srcImage.rows, srcImage.cols, CV_8UC1);
	for (int i = 0; i < srcImage.rows; i++)
		for (int j = 0; j < srcImage.cols; j++)
		{
			value = 255;  //背景设置为白色。   
			histogramImage.at<uchar>(i, j) = value;
		}
	for (int i = 0; i < srcImage.cols; i++)
		for (int j = 0; j < colswidth[i]; j++)
		{
			value = 0;  //直方图设置为黑色  
			histogramImage.at<uchar>(srcImage.rows - 1 - j, i) = value;
		}
	namedWindow("垂直积分投影图", 2);
	imshow("垂直积分投影图", histogramImage);
	imwrite("C:\\Users\\Mz\\Desktop\\垂直投影.png", histogramImage, compression_params);
	return histogramImage;
}
Mat HorizonProjection(Mat srcImage, int *rowswidth)//水平积分投影  
{
	printf("rows is %d\n", srcImage.rows);
	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION); //PNG格式图片的压缩级别  
	compression_params.push_back(9);  //这里设置保存的图像质量级别
	if (srcImage.channels() > 1)
		cvtColor(srcImage, srcImage, CV_RGB2GRAY);
	Mat srcImageBin;
	threshold(srcImage, srcImageBin, 120, 255, CV_THRESH_BINARY_INV);
	int value;
	for (int i = 0; i < srcImage.rows; i++)
	{
		for (int j = 0; j < srcImage.cols; j++)
		{
			//value=cvGet2D(src,j,i);  
			value = srcImageBin.at<uchar>(i, j);
			if (value == 255)
			{
				rowswidth[i]++; //统计每行的白色像素点    
			}
		}
		//printf("%d is %d\n", i,rowswidth[i]);//打印每行白色像素个数
	}
	Mat histogramImage(srcImage.rows, srcImage.cols, CV_8UC1);
	for (int i = 0; i<srcImage.rows; i++)
		for (int j = 0; j<srcImage.cols; j++)
		{
			value = 255;  //背景设置为白色。   
			histogramImage.at<uchar>(i, j) = value;
		}
	//imshow("d", histogramImage);  
	for (int i = 0; i<srcImage.rows; i++)
		for (int j = 0; j<rowswidth[i]; j++)
		{
			value = 0;  //直方图设置为黑色  
			histogramImage.at<uchar>(i, j) = value;
		}
	namedWindow("水平积分投影图", 2);
	imshow("水平积分投影图", histogramImage);
	imwrite("C:\\Users\\Mz\\Desktop\\水平投影.png", histogramImage, compression_params);
	return histogramImage;
}
bool isVerWhite(float y1, float y2, float y3, float y4, int *p)
{
	bool flag = 0;
	int p1, p2, p3, p4;
	p1 = y1;
	p2 = y2;
	if (y3 < y4)
	{
		p3 = y3;
		p4 = y4;
	}
	else
	{
		p3 = y4;
		p4 = y3;
	}
	if (p1 > p4)
	{
		//printf("AAAAAAAAAAAAAAAAAAAAAAA\n p1:%d p4:%d\n",p1,p4);
		for (int m = p4; m <= p1; m++)
		{
			//printf("%d is %d\n",m, p[m]);
			if (p[m] == 0)
				return 1;
		}
	}
	if (p2 < p3)
	{
		//printf("BBBBBBBBBBBBBBBBBBBBBB p2:%d p3:%d\n",p2,p3);
		for (int m = p2; m <= p3; m++)
		{
			//printf("%d is %d\n",m, p[m]);
			if (p[m] == 0)
				return 1;
		}
	}
	return 0;
}
bool isHorWhite(float x1, float x2, float x3, float x4)
{
	if (x1 > x4)
		return 1;
	if (x2 < x3)
		return 1;
	return 0;
}
void main()
{
	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION); //PNG格式图片的压缩级别  
	compression_params.push_back(9);  //这里设置保存的图像质量级别
	string fileCur;
	string path = "C:\\Users\\Mz\\Desktop\\";
	//cv::Mat img = imread("C:\\Users\\Mz\\Desktop\\框线检测\\Indoor\\1.jpg", 2 );
	cv::Mat img = imread("C:\\Users\\Mz\\Desktop\\框线数据集\\001-人寿保险投保单-V3\\000001.tif", 1);
	resize(img, img, Size(img.cols / 2.2, img.rows / 2.5), 0, 0, INTER_LANCZOS4);
	//Laplacian(img, img, CV_8U, 3);
	//namedWindow("forign", 2);
	//imshow("forign", img);
	//pre(img);
	CannyLine detector1;
	std::vector<std::vector<float> > lines1;
	detector1.cannyLine(img, lines1);

	// cannyline show
	cv::Mat imgShow(img.rows, img.cols, CV_8UC3, cv::Scalar(255, 255, 255));
	printf("%d\n", lines1.size());
	for (int m = 0; m<lines1.size(); ++m)
	{
		cv::line(imgShow, cv::Point(lines1[m][0], lines1[m][1]), cv::Point(lines1[m][2], lines1[m][3]), cv::Scalar(0, 0, 0), 1, CV_AA);
	}
	//namedWindow("1", 2);
	//imshow("1", imgShow);
	pre(imgShow);
	namedWindow("line segment", 4);
	imshow("line segment", imgShow);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(imgShow, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
	/*cv::Mat fImg(imgShow.rows, imgShow.cols, CV_8UC3, cv::Scalar(255, 255, 255));
	//drawContours(fImg, contours, -1, Scalar(0, 0, 0));
	int index = 0;
	for (; index >= 0; index = hierarchy[index][0])
	{
	cv::Scalar color(rand() & 255, rand() & 255, rand() & 255);
	//cv::drawContours(fImg, contours, index, Scalar(0), 1, 8, hierarchy);//描绘字符的外轮廓

	Rect rect = boundingRect(contours[index]);//检测外轮廓
	rectangle(fImg, rect, Scalar(0, 0, 255), 3);//对外轮廓加矩形框
	}
	namedWindow("ff", 2);
	imshow("ff", fImg);
	imwrite(path + "test.png", fImg, compression_params);*/
	printf("%d\n", contours.size());


	//框线建模部分***************************************************************************框线建模部分
	CannyLine detector2;
	std::vector<std::vector<float> > lines2;
	std::vector<std::vector<float> > ver;
	std::vector<std::vector<float> > hor;
	std::vector<std::vector<float> > allhor;
	std::vector<std::vector<float> > longver;
	std::vector<std::vector<float> > shortver;
	std::vector<std::vector<float> > longverline;
	std::vector<std::vector<float> > shortverline;
	std::vector<std::vector<float> > longshortverline;
	std::vector<std::vector<float> > ::iterator it1;
	std::vector<std::vector<float> > ::iterator it2;
	detector2.cannyLine(imgShow, lines2);
	printf("%d\n", lines2.size());
	cv::Mat finalImg(imgShow.rows, imgShow.cols, CV_8UC3, cv::Scalar(255, 255, 255));
	cv::Mat horImg(imgShow.rows, imgShow.cols, CV_8UC3, cv::Scalar(255, 255, 255));
	cv::Mat allhorImg(imgShow.rows, imgShow.cols, CV_8UC3, cv::Scalar(255, 255, 255));
	cv::Mat verImg(imgShow.rows, imgShow.cols, CV_8UC3, cv::Scalar(255, 255, 255));
	cv::Mat verhor(imgShow.rows, imgShow.cols, CV_8UC3, cv::Scalar(255, 255, 255));
	cv::Mat longverlineimg(imgShow.rows, imgShow.cols, CV_8UC3, cv::Scalar(255, 255, 255));
	cv::Mat shortverlineimg(imgShow.rows, imgShow.cols, CV_8UC3, cv::Scalar(255, 255, 255));
	cv::Mat longshortverlineimg(imgShow.rows, imgShow.cols, CV_8UC3, cv::Scalar(255, 255, 255));
	cv::Mat tImg(imgShow.rows, imgShow.cols, CV_8UC3, cv::Scalar(255, 255, 255));
	int *colsBlack = new int[imgShow.cols];  //申请src.image.cols个int型的内存空间  
	memset(colsBlack, 0, imgShow.cols * 4);  //数组必须赋初值为零，否则出错。无法遍历数组。  
											 //  memset(colheight,0,src->width*4);    
											 // CvScalar value;  
	int *rowsBlack = new int[imgShow.rows];  //申请src.image.rows个int型的内存空间  
	memset(rowsBlack, 0, imgShow.rows * 4);  //数组必须赋初值为零，否则出错。无法遍历数组。 
	float length = 0;
	float mid = 0;
	double   x1, x2, x, y1, y2, y;
	float slope;
	std::vector<float> lineTemp(4);
	float line[4];
	for (int m = 0; m<lines2.size(); ++m)
	{
		length = sqrt((lines2[m][1] - lines2[m][3])*(lines2[m][1] - lines2[m][3]) + (lines2[m][0] - lines2[m][2])*(lines2[m][0] - lines2[m][2]));
		slope = abs((lines2[m][1] - lines2[m][3]) / (lines2[m][0] - lines2[m][2]));
		if (lines2[m][0] <= lines2[m][2])
		{
			line[0] = lines2[m][0];
			line[1] = lines2[m][1];
			line[2] = lines2[m][2];
			line[3] = lines2[m][3];
		}
		else
		{
			line[0] = lines2[m][2];
			line[1] = lines2[m][3];
			line[2] = lines2[m][0];
			line[3] = lines2[m][1];
		}
		lineTemp[0] = line[0];
		lineTemp[1] = line[1];
		lineTemp[2] = line[2];
		lineTemp[3] = line[3];
		if (line[0] == line[2])
		{
			if ((lineTemp[0] < 700 && lineTemp[1] < 500) || (lineTemp[0]>500 && lineTemp[1]>1380))
				continue;
			ver.push_back(lineTemp);
		}
		else if (line[1] == line[3])
		{
			hor.push_back(lineTemp);
		}
		else if (slope > 5)
		{
			if ((lineTemp[0] < 700 && lineTemp[1] < 250) || (lineTemp[0]>500 && lineTemp[1]>1380))
				continue;
			ver.push_back(lineTemp);
		}
		else
		{
			hor.push_back(lineTemp);
		}
		//if (length < 23)
		//	continue;
		//Mat* temp = new Mat(imgShow.rows, imgShow.cols, CV_8UC3, cv::Scalar(255, 255, 255));
		//cv::line(*temp, cv::Point(lines2[m][0], lines2[m][1]), cv::Point(lines2[m][2], lines2[m][3]), cv::Scalar(0, 0, 0), 1, CV_AA);
		//string path = "C:\\Users\\Mz\\Desktop\\线集合\\" + to_string(m) + "test.png";
		//imwrite(path, *temp, compression_params);
		//printf("!!!%f\n", length);
		cv::line(finalImg, cv::Point(lines2[m][0], lines2[m][1]), cv::Point(lines2[m][2], lines2[m][3]), cv::Scalar(0, 0, 0), 1, CV_AA);
	}
	//投影
	HorizonProjection(finalImg, rowsBlack);
	VerticalProjection(finalImg, colsBlack);

	//提取水平线*************
	int numb1, numb2, numb11, numb22;
	numb1 = numb2 = numb11 = numb22 = 0;
	int m = 0;
	int n = 0;
	for (m = 0; m < ver.size(); m++)
	{
		lineTemp[0] = ver[m][0];
		lineTemp[1] = ver[m][1];
		lineTemp[2] = ver[m][2];
		lineTemp[3] = ver[m][3];
		if (sqrt((ver[m][0] - ver[m][2])*(ver[m][0] - ver[m][2]) + (ver[m][1] - ver[m][3])*(ver[m][1] - ver[m][3])) < 50)
		{
			shortver.push_back(lineTemp);
			continue;
		}
		longver.push_back(lineTemp);
		cv::line(verhor, cv::Point(ver[m][0], ver[m][1]), cv::Point(ver[m][2], ver[m][3]), cv::Scalar(0, 0, 0), 1, CV_AA);
		numb2++;
		if (ver[m][1] <= ver[m][3])
			numb22++;
	}
	/*for (m = 0; m < longver.size(); m++)
	{
		cv::line(tImg, cv::Point(longver[m][0], longver[m][1]), cv::Point(longver[m][2], longver[m][3]), cv::Scalar(0, 0, 0), 1, CV_AA);
	}
	namedWindow("t", 2);
	imshow("t", tImg);*/
	//提取水平线*************
	for (int k = 0; k < hor.size(); k++)
	{
		cv::line(horImg, cv::Point(hor[k][0], hor[k][1]), cv::Point(hor[k][2], hor[k][3]), cv::Scalar(0, 0, 0), 1, CV_AA);
	}
	namedWindow("hor", 2);
	imshow("hor", horImg);
	imwrite(path + "hor.png", horImg, compression_params);

	printf("hor:%d %d,ver:%d %d\n", numb1, numb11, numb2, numb22);
	//第一次水平直线拟合**********************************************
	for (m = 0; m<hor.size();)
	{
			lineTemp[0] = hor[m][0];     //记录线边界
			lineTemp[1] = hor[m][1];
			lineTemp[2] = hor[m][2];
			lineTemp[3] = hor[m][3];
			for (n = m + 1; m < hor.size();)
			{
				if ((dist(hor[m][0], hor[m][1], hor[m][2], hor[m][3], (hor[n][0] + hor[n][2]) / 2, (hor[n][1] + hor[n][3]) / 2) < 10))
				{
					
					if (lineTemp[0] > hor[n][2])
					{
						if (abs(lineTemp[0]-hor[n][2])>10)
						{
							n++;
							continue;
						}
					}
					if (lineTemp[2] < hor[n][0])
					{
						if (abs(lineTemp[2] - hor[n][0])>10)
						{
							n++;
							continue;
						}
					}
					if (hor[n][0] < lineTemp[0])
						lineTemp[0] = hor[n][0];
					if (hor[n][1] > lineTemp[1])
						lineTemp[1] = hor[n][1];
					if (hor[n][2] > lineTemp[2])
						lineTemp[2] = hor[n][2];
					if (hor[n][3] > lineTemp[3])
						lineTemp[3] = hor[n][3];
					hor.erase(hor.begin() + n);
				}
				else
				{
					n++;    //不需要拟合
				}
			}
			allhor.push_back(lineTemp);
			hor.erase(hor.begin());
	}
	//第一次水平直线拟合**********************************************
	for (int k = 0; k < allhor.size(); k++)
	{
		cv::line(allhorImg, cv::Point(allhor[k][0], allhor[k][1]), cv::Point(allhor[k][2], allhor[k][3]), cv::Scalar(0, 0, 0), 1, CV_AA);
	}
	namedWindow("allhor", 2);
	imshow("allhor", allhorImg);
	imwrite(path + "allhor.png", allhorImg, compression_params);

	printf("allhor is %d\n", allhor.size());
	//第一次长垂直线拟合**********************************************
	for (m = 0; m<longver.size();)
	{
		if (longver[m][1] <= longver[m][3])     //左边点在上
		{
			lineTemp[0] = longver[m][0];     //记录线边界
			lineTemp[1] = longver[m][1];
			lineTemp[2] = longver[m][2];
			lineTemp[3] = longver[m][3];
			for (n = m + 1; n<longver.size();)
			{
				if ((dist(longver[m][0], longver[m][1], longver[m][2], longver[m][3], (longver[n][0] + longver[n][2]) / 2, (longver[n][1] + longver[n][3]) / 2)<20) && !isVerWhite(lineTemp[1], lineTemp[3], longver[n][1], longver[n][3], rowsBlack))
				{
					if (longver[n][0] < (img.cols / 2))
					{
						if (longver[n][1] < longver[n][3])   //左边点在上
						{
							if (longver[n][0] < lineTemp[0])
								lineTemp[0] = longver[n][0];
							if (longver[n][1] < lineTemp[1])
								lineTemp[1] = longver[n][1];
							if (longver[n][2] < lineTemp[2])
								lineTemp[2] = longver[n][2];
							if (longver[n][3] > lineTemp[3])
								lineTemp[3] = longver[n][3];
						}
						else   //左边点在下
						{
							if (longver[n][0] < lineTemp[0])
								lineTemp[0] = longver[n][0];
							if (longver[n][3] < lineTemp[1])
								lineTemp[1] = longver[n][3];
							if (longver[n][2] < lineTemp[2])
								lineTemp[2] = longver[n][2];
							if (longver[n][1] > lineTemp[3])
								lineTemp[3] = longver[n][1];
						}
						longver.erase(longver.begin() + n); //删除longver[n]
					}
					else
					{
						if (longver[n][1] < longver[n][3])   //左边点在上
						{
							if (longver[n][0] > lineTemp[0])
								lineTemp[0] = longver[n][0];
							if (longver[n][1] < lineTemp[1])
								lineTemp[1] = longver[n][1];
							if (longver[n][2] > lineTemp[2])
								lineTemp[2] = longver[n][2];
							if (longver[n][3] > lineTemp[3])
								lineTemp[3] = longver[n][3];
						}
						else   //左边点在下
						{
							if (longver[n][0] > lineTemp[0])
								lineTemp[0] = longver[n][0];
							if (longver[n][3] < lineTemp[1])
								lineTemp[1] = longver[n][3];
							if (longver[n][2] > lineTemp[2])
								lineTemp[2] = longver[n][2];
							if (longver[n][1] > lineTemp[3])
								lineTemp[3] = longver[n][1];
						}
						longver.erase(longver.begin() + n); //删除longver[n]
					}
				}
				else
				{
					n++;    //不需要拟合，直接跳过
				}
			}
			longverline.push_back(lineTemp);
			longver.erase(longver.begin());     //删除longver[m]
		}
		else  //左边点在下
		{
			lineTemp[0] = longver[m][0];
			lineTemp[1] = longver[m][3];
			lineTemp[2] = longver[m][2];
			lineTemp[3] = longver[m][1];
			for (n = m + 1; n<longver.size();)
			{
				if (dist(longver[m][0], longver[m][1], longver[m][2], longver[m][3], (longver[n][0] + longver[n][2]) / 2, (longver[n][1] + longver[n][3]) / 2)<20 && !isVerWhite(lineTemp[1], lineTemp[3], longver[n][1], longver[n][3], rowsBlack))
				{
					if (longver[n][1] < longver[n][3])   //左边点在上
					{
						if (longver[n][0] < lineTemp[0])
							lineTemp[0] = longver[n][0];
						if (longver[n][1] < lineTemp[1])
							lineTemp[1] = longver[n][1];
						if (longver[n][2] < lineTemp[2])
							lineTemp[2] = longver[n][2];
						if (longver[n][3] > lineTemp[3])
							lineTemp[3] = longver[n][3];
					}
					else   //左边点在下
					{
						if (longver[n][0] < lineTemp[0])
							lineTemp[0] = longver[n][0];
						if (longver[n][3] < lineTemp[1])
							lineTemp[1] = longver[n][3];
						if (longver[n][2] < lineTemp[2])
							lineTemp[2] = longver[n][2];
						if (longver[n][1] > lineTemp[3])
							lineTemp[3] = longver[n][1];
					}
					longver.erase(longver.begin() + n); //删除longver[n]
				}
				else
				{
					n++;
				}
			}
			longverline.push_back(lineTemp);
			longver.erase(longver.begin());     //删除longver[m]
		}
	}
	//第一次长垂直线拟合**********************************************

	//第一次短垂直线拟合**********************************************
	for (m = 0; m<shortver.size();)
	{
		if (shortver[m][1] <= shortver[m][3])     //左边点在上
		{
			lineTemp[0] = shortver[m][0];     //记录线边界
			lineTemp[1] = shortver[m][1];
			lineTemp[2] = shortver[m][2];
			lineTemp[3] = shortver[m][3];
			for (n = m + 1; n<shortver.size();)
			{
				if ((dist(shortver[m][0], shortver[m][1], shortver[m][2], shortver[m][3], (shortver[n][0] + shortver[n][2]) / 2, (shortver[n][1] + shortver[n][3]) / 2)<20) && !isVerWhite(lineTemp[1], lineTemp[3], shortver[n][1], shortver[n][3], rowsBlack))
				{
					if (shortver[n][1] < shortver[n][3])   //左边点在上
					{

						if (lineTemp[1] > shortver[n][3])
						{
							if (abs(lineTemp[1] - shortver[n][3]) > 10)
							{
								n++;
								continue;
							}
						}
						if (lineTemp[3] < shortver[n][1])
						{
							if (abs(lineTemp[3] - shortver[n][1]) > 10)
							{
								n++;
								continue;
							}
						}
						if (shortver[n][0] < lineTemp[0])
							lineTemp[0] = shortver[n][0];
						if (shortver[n][1] < lineTemp[1])
							lineTemp[1] = shortver[n][1];
						if (shortver[n][2] < lineTemp[2])
							lineTemp[2] = shortver[n][2];
						if (shortver[n][3] > lineTemp[3])
							lineTemp[3] = shortver[n][3];
					}
					else   //左边点在下
					{
						if (lineTemp[1] > shortver[n][1])
						{
							if (abs(lineTemp[1] - shortver[n][1]) > 20)
							{
								n++;
								continue;
							}
						}
						if (lineTemp[3] < shortver[n][3])
						{
							if (abs(lineTemp[3] - shortver[n][3]) > 20)
							{
								n++;
								continue;
							}
						}
						if (shortver[n][0] < lineTemp[0])
							lineTemp[0] = shortver[n][0];
						if (shortver[n][3] < lineTemp[1])
							lineTemp[1] = shortver[n][3];
						if (shortver[n][2] < lineTemp[2])
							lineTemp[2] = shortver[n][2];
						if (shortver[n][1] > lineTemp[3])
							lineTemp[3] = shortver[n][1];
					}
					shortver.erase(shortver.begin() + n); //删除longver[n]
				}
				else
				{
					n++;    //不需要拟合，直接跳过
				}
			}
			shortverline.push_back(lineTemp);
			shortver.erase(shortver.begin());     //删除longver[m]
		}
		else  //左边点在下
		{
			lineTemp[0] = shortver[m][0];
			lineTemp[1] = shortver[m][3];
			lineTemp[2] = shortver[m][2];
			lineTemp[3] = shortver[m][1];
			for (n = m + 1; n<shortver.size();)
			{
				if (dist(shortver[m][0], shortver[m][1], shortver[m][2], shortver[m][3], (shortver[n][0] + shortver[n][2]) / 2, (shortver[n][1] + shortver[n][3]) / 2)<20 && !isVerWhite(lineTemp[1], lineTemp[3], shortver[n][1], shortver[n][3], rowsBlack))
				{
					if (shortver[n][1] < shortver[n][3])   //左边点在上
					{
						if (lineTemp[1] > shortver[n][3])
						{
							if (abs(lineTemp[1] - shortver[n][3]) > 20)
							{
								n++;
								continue;
							}
						}
						if (lineTemp[3] < shortver[n][1])
						{
							if (abs(lineTemp[3] - shortver[n][1]) > 20)
							{
								n++;
								continue;
							}
						}
						if (shortver[n][0] < lineTemp[0])
							lineTemp[0] = shortver[n][0];
						if (shortver[n][1] < lineTemp[1])
							lineTemp[1] = shortver[n][1];
						if (shortver[n][2] < lineTemp[2])
							lineTemp[2] = shortver[n][2];
						if (shortver[n][3] > lineTemp[3])
							lineTemp[3] = shortver[n][3];
					}
					else   //左边点在下
					{
						if (lineTemp[1] > shortver[n][1])
						{
							if (abs(lineTemp[1] - shortver[n][1]) > 20)
							{
								n++;
								continue;
							}
						}
						if (lineTemp[3] < shortver[n][3])
						{
							if (abs(lineTemp[3] - shortver[n][3]) > 20)
							{
								n++;
								continue;
							}
						}
						if (shortver[n][0] < lineTemp[0])
							lineTemp[0] = shortver[n][0];
						if (shortver[n][3] < lineTemp[1])
							lineTemp[1] = shortver[n][3];
						if (shortver[n][2] < lineTemp[2])
							lineTemp[2] = shortver[n][2];
						if (shortver[n][1] > lineTemp[3])
							lineTemp[3] = shortver[n][1];
					}
					shortver.erase(shortver.begin() + n); //删除longver[n]
				}
				else
				{
					n++;
				}
			}
			shortverline.push_back(lineTemp);
			shortver.erase(shortver.begin());     //删除longver[m]
		}
	}
	//第一次短垂直线拟合**********************************************

	//打印长垂线
	for (m = 0; m < longverline.size(); m++)
	{
		//Mat* temp = new Mat(imgShow.rows, imgShow.cols, CV_8UC3, cv::Scalar(255, 255, 255));
		//cv::line(*temp, cv::Point(longverline[m][0], longverline[m][1]), cv::Point(longverline[m][2], longverline[m][3]), cv::Scalar(0, 0, 0), 1, CV_AA);
		cv::line(longverlineimg, cv::Point(longverline[m][0], longverline[m][1]), cv::Point(longverline[m][2], longverline[m][3]), cv::Scalar(0, 0, 0), 1, CV_AA);
		//imwrite("C:\\Users\\Mz\\Desktop\\垂直线集合\\"+to_string(m) + "ver.png", *temp, compression_params);
	}
	namedWindow("longverline", 2);
	imshow("longverline", longverlineimg);
	imwrite(path + "longverline.png", longverlineimg, compression_params);
	//打印短垂线
	for (m = 0; m < shortverline.size(); m++)
	{
		//Mat* temp = new Mat(imgShow.rows, imgShow.cols, CV_8UC3, cv::Scalar(255, 255, 255));
		//cv::line(*temp, cv::Point(longverline[m][0], longverline[m][1]), cv::Point(longverline[m][2], longverline[m][3]), cv::Scalar(0, 0, 0), 1, CV_AA);
		cv::line(shortverlineimg, cv::Point(shortverline[m][0], shortverline[m][1]), cv::Point(shortverline[m][2], shortverline[m][3]), cv::Scalar(0, 0, 0), 1, CV_AA);
		//imwrite("C:\\Users\\Mz\\Desktop\\垂直线集合\\"+to_string(m) + "ver.png", *temp, compression_params);
	}
	namedWindow("shortverline", 2);
	imshow("shortverline", shortverlineimg);
	imwrite(path + "shortverline.png", shortverlineimg, compression_params);


	//第一次长和短垂直线拟合**********************************************
	for (m = 0; m<longverline.size();)
	{
		if (longverline[m][1] <= longverline[m][3])     //左边点在上
		{
			lineTemp[0] = longverline[m][0];     //记录线边界
			lineTemp[1] = longverline[m][1];
			lineTemp[2] = longverline[m][2];
			lineTemp[3] = longverline[m][3];
			for (n = 0; n<shortverline.size();)
			{
				if ((dist(longverline[m][0], longverline[m][1], longverline[m][2], longverline[m][3], (shortverline[n][0] + shortverline[n][2]) / 2, (shortverline[n][1] + shortverline[n][3]) / 2)<20) && !isVerWhite(lineTemp[1], lineTemp[3], shortverline[n][1], shortverline[n][3], rowsBlack))
				{
					if (shortverline[n][1] < shortverline[n][3])   //左边点在上
					{

						if (lineTemp[1] > shortverline[n][3])
						{
							if (abs(lineTemp[1] - shortverline[n][3]) > 10)
							{
								n++;
								continue;
							}
						}
						if (lineTemp[3] < shortverline[n][1])
						{
							if (abs(lineTemp[3] - shortverline[n][1]) > 10)
							{
								n++;
								continue;
							}
						}
						if (shortverline[n][0] < lineTemp[0])
							lineTemp[0] = shortverline[n][0];
						if (shortverline[n][1] < lineTemp[1])
							lineTemp[1] = shortverline[n][1];
						if (shortverline[n][2] < lineTemp[2])
							lineTemp[2] = shortverline[n][2];
						if (shortverline[n][3] > lineTemp[3])
							lineTemp[3] = shortverline[n][3];
					}
					else   //左边点在下
					{
						if (lineTemp[1] > shortverline[n][1])
						{
							if (abs(lineTemp[1] - shortverline[n][1]) > 20)
							{
								n++;
								continue;
							}
						}
						if (lineTemp[3] < shortverline[n][3])
						{
							if (abs(lineTemp[3] - shortverline[n][3]) > 20)
							{
								n++;
								continue;
							}
						}
						if (shortverline[n][0] < lineTemp[0])
							lineTemp[0] = shortverline[n][0];
						if (shortverline[n][3] < lineTemp[1])
							lineTemp[1] = shortverline[n][3];
						if (shortverline[n][2] < lineTemp[2])
							lineTemp[2] = shortverline[n][2];
						if (shortverline[n][1] > lineTemp[3])
							lineTemp[3] = shortverline[n][1];
					}
					shortverline.erase(shortverline.begin() + n); //删除longver[n]
				}
				else
				{
					n++;    //不需要拟合，直接跳过
				}
			}
			longshortverline.push_back(lineTemp);
			longverline.erase(longverline.begin());     //删除longver[m]
		}
		else  //左边点在下
		{
			lineTemp[0] = longverline[m][0];
			lineTemp[1] = longverline[m][3];
			lineTemp[2] = longverline[m][2];
			lineTemp[3] = longverline[m][1];
			for (n = 0; n<longverline.size();)
			{
				if (dist(longverline[m][0], longverline[m][1], longverline[m][2], longverline[m][3], (shortverline[n][0] + shortverline[n][2]) / 2, (shortverline[n][1] + shortverline[n][3]) / 2)<20 && !isVerWhite(lineTemp[1], lineTemp[3], shortverline[n][1], shortverline[n][3], rowsBlack))
				{
					if (shortverline[n][1] < shortverline[n][3])   //左边点在上
					{
						if (lineTemp[1] > shortverline[n][3])
						{
							if (abs(lineTemp[1] - shortverline[n][3]) > 20)
							{
								n++;
								continue;
							}
						}
						if (lineTemp[3] < shortverline[n][1])
						{
							if (abs(lineTemp[3] - shortverline[n][1]) > 20)
							{
								n++;
								continue;
							}
						}
						if (shortverline[n][0] < lineTemp[0])
							lineTemp[0] = shortverline[n][0];
						if (shortverline[n][1] < lineTemp[1])
							lineTemp[1] = shortverline[n][1];
						if (shortverline[n][2] < lineTemp[2])
							lineTemp[2] = shortverline[n][2];
						if (shortverline[n][3] > lineTemp[3])
							lineTemp[3] = shortverline[n][3];
					}
					else   //左边点在下
					{
						if (lineTemp[1] > shortverline[n][1])
						{
							if (abs(lineTemp[1] - shortverline[n][1]) > 20)
							{
								n++;
								continue;
							}
						}
						if (lineTemp[3] < shortverline[n][3])
						{
							if (abs(lineTemp[3] - shortverline[n][3]) > 20)
							{
								n++;
								continue;
							}
						}
						if (shortverline[n][0] < lineTemp[0])
							lineTemp[0] = shortverline[n][0];
						if (shortverline[n][3] < lineTemp[1])
							lineTemp[1] = shortverline[n][3];
						if (shortverline[n][2] < lineTemp[2])
							lineTemp[2] = shortverline[n][2];
						if (shortverline[n][1] > lineTemp[3])
							lineTemp[3] = shortverline[n][1];
					}
					shortverline.erase(shortverline.begin() + n); //删除longver[n]
				}
				else
				{
					n++;
				}
			}
			longshortverline.push_back(lineTemp);
			longverline.erase(longverline.begin());     //删除longver[m]
		}
	}
	//第一次长和短垂直线拟合**********************************************
	//打印长线和短线
	for (m = 0; m < longshortverline.size(); m++)
	{
		//Mat* temp = new Mat(imgShow.rows, imgShow.cols, CV_8UC3, cv::Scalar(255, 255, 255));
		//cv::line(*temp, cv::Point(longverline[m][0], longverline[m][1]), cv::Point(longverline[m][2], longverline[m][3]), cv::Scalar(0, 0, 0), 1, CV_AA);
		cv::line(longshortverlineimg, cv::Point(longshortverline[m][0], longshortverline[m][1]), cv::Point(longshortverline[m][2], longshortverline[m][3]), cv::Scalar(0, 0, 0), 1, CV_AA);
		//imwrite("C:\\Users\\Mz\\Desktop\\垂直线集合\\"+to_string(m) + "ver.png", *temp, compression_params);

	}
	for (m = 0; m < shortverline.size(); m++)
	{
		//Mat* temp = new Mat(imgShow.rows, imgShow.cols, CV_8UC3, cv::Scalar(255, 255, 255));
		//cv::line(*temp, cv::Point(longverline[m][0], longverline[m][1]), cv::Point(longverline[m][2], longverline[m][3]), cv::Scalar(0, 0, 0), 1, CV_AA);
		cv::line(longshortverlineimg, cv::Point(shortverline[m][0], shortverline[m][1]), cv::Point(shortverline[m][2], shortverline[m][3]), cv::Scalar(0, 0, 0), 1, CV_AA);
		//imwrite("C:\\Users\\Mz\\Desktop\\垂直线集合\\"+to_string(m) + "ver.png", *temp, compression_params);
	}
	namedWindow("long and short", 2);
	imshow("long and short", longshortverlineimg);
	imwrite(path + "long and short.png", longshortverlineimg, compression_params);
	printf("longverline：%d\n", longverline.size());
	printf("allverlinenumber：%d\n", shortverline.size() + longshortverline.size());
	for (m = 0; m < longver.size(); m++)
	{

	}
	/*for (it1=ver.begin(),m=0; it1!= ver.end();it1++,m++)
	{
	lineTemp[0] = 9999;
	lineTemp[1] = 9999;
	lineTemp[2] =9999;
	lineTemp[3] =9999;
	for (it2 = ver.begin(),n=0; it2 != ver.end(); it2++,n++)
	{
	if (it1 == it2)
	continue;
	if (equ(it1[m][0], it1[m][1], it1[m][2], it1[m][3], it2[n][0], it2[n][1]))       //共线情况(隐藏共点)
	{
	if (equ(it1[m][0], it1[m][1], it1[m][2], it1[m][3], it2[n][2], it2[n][3]))   //四点共线
	{
	//ver.erase(it1);   这个操作放到最后来
	ver.erase(it2);
	if(it1[m][0]<=it2[n][0])
	{
	lineTemp[0] = it1[m][0];
	lineTemp[1] = it1[m][1];
	}
	else
	{
	lineTemp[0] = it2[m][0];
	lineTemp[1] = it2[m][1];
	}
	if (it1[m][2] <= it2[n][2])
	{
	lineTemp[2] = it1[n][2];
	lineTemp[3] = it1[n][3];
	}
	else
	{
	lineTemp[2] = it2[n][2];
	lineTemp[3] = it2[n][3];
	}
	}
	else  //三点共线
	{

	}
	}
	}
	}
	*/
	namedWindow("final", 2);
	imshow("final", finalImg);
	imwrite(path + "final.png", finalImg, compression_params);
	cv::waitKey(0);
}