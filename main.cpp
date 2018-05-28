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
	namedWindow("cannyline", 2);
	imshow("cannyline", grayImg);
	Mat bw;
	adaptiveThreshold(~grayImg, bw, 255, CV_ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, -2);
	namedWindow("orign", 2);
	imshow("orign", bw);
	//CvMemStorage *contourStorage = cvCreateMemStorage();
	vector<vector<Point>> contours1;
	findContours(bw, contours1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	drawContours(bw, contours1, -1, Scalar(255, 255, 255));
	printf("%d\n", contours1.size());
	namedWindow("draw", 2);
	imshow("draw", bw);
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
	namedWindow("horizontal", 2);
	imshow("horizontal", horizontal);
	int scale2 = 65;      //越大，检测到的垂直直线越多
	int verticalsize = vertical.rows / scale2;
	Mat verticalStructure = getStructuringElement(MORPH_RECT, Size(1, verticalsize));
	erode(vertical, vertical, verticalStructure, Point(-1, -1));
	dilate(vertical, vertical, verticalStructure, Point(-1, -1));
	namedWindow("vertical", 2);
	imshow("vertical", vertical);
	//Mat joints;
	//bitwise_and(horizontal, vertical, joints);
	//namedWindow("joints", 2);
	//imshow("joints", joints);
	Mat mask = horizontal + vertical;
	img = mask.clone();
	namedWindow("mask", 2);
	imshow("mask", mask);
	vector<Vec4i> hierarchy;
	std::vector<std::vector<cv::Point> > contours2;
	cv::findContours(mask, contours2, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	vector<vector<Point> > contours_poly(contours2.size());
	vector<Rect> boundRect(contours2.size());
	vector<Mat> rois;

	cv::waitKey(0);
}

//bool equ()
void main()
{
	string fileCur;
	//cv::Mat img = imread("C:\\Users\\Mz\\Desktop\\框线检测\\Indoor\\1.jpg", 2 );
	cv::Mat img = imread("C:\\Users\\Mz\\Desktop\\框线数据集\\001-人寿保险投保单-V3\\000001.tif", 1);
	resize(img, img, Size(img.cols / 2.2, img.rows / 2.5), 0, 0, INTER_LANCZOS4);
	//Laplacian(img, img, CV_8U, 3);
	namedWindow("forign", 2);
	imshow("forign", img);
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
	namedWindow("1", 2);
	imshow("1", imgShow);
	pre(imgShow);
	namedWindow("line segment", 4);
	imshow("line segment", imgShow);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(imgShow, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
	cv::Mat fImg(imgShow.rows, imgShow.cols, CV_8UC3, cv::Scalar(255, 255, 255));
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
	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION); //PNG格式图片的压缩级别  
	compression_params.push_back(9);  //这里设置保存的图像质量级别
	string path = "C:\\Users\\Mz\\Desktop\\";
	imwrite(path + "test.png", fImg, compression_params);
	printf("%d\n", contours.size());
	CannyLine detector2;
	std::vector<std::vector<float> > lines2;
	std::vector<std::vector<float> > ver;
	std::vector<std::vector<float> > hor;
	detector2.cannyLine(imgShow, lines2);
	printf("%d\n", lines2.size());
	cv::Mat finalImg(imgShow.rows, imgShow.cols, CV_8UC3, cv::Scalar(255, 255, 255));
	cv::Mat horImg(imgShow.rows, imgShow.cols, CV_8UC3, cv::Scalar(255, 255, 255));
	cv::Mat verImg(imgShow.rows, imgShow.cols, CV_8UC3, cv::Scalar(255, 255, 255));
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
	for (int m = 0; m < hor.size(); m++)
	{
		cv::line(horImg, cv::Point(hor[m][0], hor[m][1]), cv::Point(hor[m][2], hor[m][3]), cv::Scalar(0, 0, 0), 1, CV_AA);
	}
	for (int m = 0; m < ver.size(); m++)
	{
		cv::line(verImg, cv::Point(ver[m][0], ver[m][1]), cv::Point(ver[m][2], ver[m][3]), cv::Scalar(0, 0, 0), 1, CV_AA);
	}
	namedWindow("hor", 2);
	imshow("hor", horImg);
	imwrite(path + "hor.png", horImg, compression_params);
	namedWindow("ver", 2);
	imshow("ver", verImg);
	imwrite(path + "ver.png", verImg, compression_params);
	for (int m = 0; m < lines2.size(); ++m)
	{
		for (int n = 0; n < lines2.size(); ++n)
		{
			if (m == n)
				continue;
			//x1 = (lines2[m][0] + lines2[m][2])/2; y1 = (lines2[m][1] + lines2[m][3])/2;
			//x2 = (lines2[n][0] + lines2[n][2]) / 2; y2 = (lines2[n][1] + lines2[n][3]) / 2;
			//mid = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
			if (mid < 10)
			{

			}
		}
	}
	namedWindow("final", 2);
	imshow("final", finalImg);
	imwrite(path + "final.png", finalImg, compression_params);
	cv::waitKey(0);
}