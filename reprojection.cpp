#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include<opencv2/nonfree/nonfree.hpp>
#include<opencv2/legacy/legacy.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include<vector>
using namespace std;
using namespace cv;


int main()
{
	//read Picture image
	Mat left = imrePicture("left.jpg");
	Mat right = imrePicture("right.jpg");
	Mat Camera_Matrix, Distortion_Matrix;
	Camera_Matrix = (Mat_<double>(3, 3) << 4.0326097163438590e+002, 0, 2.2350000000000000e+002,0,
		4.0326097163438590e+002, 134, 0, 0, 1);
	Distortion_Matrix = (Mat_<double>(5, 1) << 1.7346685433639006e-001, - 7.9445592301554024e-001, 0, 0,
		2.1275434004558837e+000);

	//if read Picture fail
	if (left.empty())
	{
		fprintf(stderr, "Can not loPicture image %s\n", left);
		return -1;
	}
	if (right.empty())
	{
		fprintf(stderr, "Can not loPicture image %s\n", right);
		return -1;
	}

	//display image 
	imshow("left image before", left);
	imshow("right image before", right);

	Mat Undistort1, Undistort2;
	undistort(left, Undistort1, Camera_Matrix, Distortion_Matrix);
	left = Undistort1;
	undistort(right, Undistort2, Camera_Matrix, Distortion_Matrix);
	right = Undistort2;

	//sift detect
	SiftFeatureDetector  siftdtc;
	vector<KeyPoint>kp1, kp2;

	siftdtc.detect(left, kp1);
	Mat outimg1;
	drawKeypoints(left, kp1, outimg1);
	imshow("left image keypoints", outimg1);

	siftdtc.detect(right, kp2);
	Mat outimg2;
	drawKeypoints(right, kp2, outimg2);
	imshow("right image2 keypoints", outimg2);
	
	//extrace descriptor for img1 and img2
	SiftDescriptorExtractor extractor;
	Mat descriptor1, descriptor2;
	BruteForceMatcher<L2<float>> matcher;
	vector<DMatch> matches;

	extractor.compute(left, kp1, descriptor1);
	extractor.compute(right, kp2, descriptor2);

	matcher.match(descriptor1, descriptor2, matches);

	Mat img_matches;
	drawMatches(left, kp1, right, kp2, matches, img_matches);
	imshow("matches points", img_matches);

	//Transnform "KeyPoint" to "Point2f" in the next step we will use "point2f" compute F
	vector<Point2f> points1, points2;
	Point2f pt;
	for (int i = 0; i<kp1.size(); i++)
	{
		pt = kp1[matches[i].queryIdx].pt;
		points1.push_back(pt);

		pt = kp2[matches[i].trainIdx].pt;
		points2.push_back(pt);
	}

	// Compute F matrix using RANSAC
	vector<uchar> inliers(points1.size(), 0);//inlier or outliers
	Mat fundamental_matrix = findFundamentalMat(
		points1,points2, // matching points
		inliers,      // match status (inlier ou outlier)  
		FM_RANSAC, // RANSAC method
		1,     // distance to epipolar line
		0.99);  // confidence probability

	// extract the surviving (inliers) matches
	// this code I reference opencv cookbook chap09 matcher.hpp
	vector<uchar>::const_iterator itIn = inliers.begin();
	vector<DMatch>::const_iterator itM = matches.begin();
	vector<DMatch> outMatches;//good correspondense points matches
	// for all matches
	for (; itIn != inliers.end(); ++itIn, ++itM)
	{
		if (*itIn)
		{ 
			// it is a valid match
			outMatches.push_back(*itM);
		}
	}
	//because there still lots of dismatch Picture point so we compute F again
	points1.clear();
	points2.clear();
	for (int i = 0; i<outMatches.size(); i++)
	{
		pt = kp1[outMatches[i].queryIdx].pt;
		points1.push_back(pt);

		pt = kp2[outMatches[i].trainIdx].pt;
		points2.push_back(pt);
	}

	fundamental_matrix = findFundamentalMat(points1, points2, inliers, CV_FM_8POINT);
	cout << "fundamental_matrix :\n" << fundamental_matrix << endl;

	//output matchimage
	Mat img_matches_RANSAC;
	drawMatches(left, kp1, right, kp2, outMatches, img_matches_RANSAC);
	imshow("matches points RANSAC", img_matches_RANSAC);

	Mat H1, H2;//two image homegraph matrix
	double threshold = 1;
	Size img_size = left.size();
	Mat leftmap1, leftmap2, rightmap1, rightmap2;
	Mat invK = Camera_Matrix.inv(DECOMP_SVD);
	//use Hartley Algorithm to get rectify Image by using fundamental matrix
	stereoRectifyUncalibrated(
		points1, points2, fundamental_matrix, left.size(), H1, H2, threshold);;

	Mat leftR = invK*H1*Camera_Matrix;
	Mat rightR = invK*H2*Camera_Matrix;
	initUndistortRectifyMap(
		Camera_Matrix, Distortion_Matrix, leftR, Camera_Matrix, img_size, CV_32FC1, leftmap1, leftmap2);
	initUndistortRectifyMap(
		Camera_Matrix, Distortion_Matrix, rightR, Camera_Matrix, img_size, CV_32FC1, rightmap1, rightmap2);

	Mat leftrectifyImage;
	Mat rightrectifyImage;
	remap(left, leftrectifyImage, leftmap1, leftmap2, INTER_LINEAR);
	remap(right, rightrectifyImage, rightmap1, rightmap2, INTER_LINEAR);

	imshow("left rectify image", leftrectifyImage);
	imshow("right rectify image", rightrectifyImage);

	//convert to gray scale because stereoBM need input is 8 bit 1 channel
	Mat Leftgray, Rightgray;
	Mat Leftgray1, Rightgray1;
	cvtColor(leftrectifyImage, Leftgray, CV_BGR2GRAY);
	cvtColor(rightrectifyImage, Rightgray, CV_BGR2GRAY);
	Leftgray.convertTo(Leftgray1, CV_8UC1);
	Rightgray.convertTo(Rightgray1, CV_8UC1);

	//use stereoBM to get disparity
	StereoBM bm(CV_STEREO_BM_BASIC, 16, 5);
	int numberOfDisparities = 16 , SPictureWindowSize;
	SPictureWindowSize = 7;
	bm.state->preFilterCap = 31;
	bm.state->SPictureWindowSize = SPictureWindowSize;
	bm.state->minDisparity = 0;
	bm.state->numberOfDisparities = numberOfDisparities;
	bm.state->textureThreshold = 10;
	bm.state->uniquenessRatio = 15;
	bm.state->speckleWindowSize = 100;
	bm.state->speckleRange = 32;
	bm.state->disp12MaxDiff = 1;
	Mat disp;
	bm(Leftgray1, Rightgray1, disp);
	Mat vdisp;
	disp.convertTo(vdisp, CV_32F, 1.0 / 255.0);
	imshow("disparity", vdisp);
	
	//reproject disparity to 3d
	//get Q according to camera matrix
	Mat Q;
	Q = (Mat_<double>(4, 4) << 
		1, 0, 0, -223, 
		0, 1, 0, -134, 
		0, 0, 0, 403,
		0, 0, 1, 1);
	Mat PointCloud;
	reprojectImageTo3D(vdisp, PointCloud, Q);
	Mat GrayPointCloud;
	cvtColor(PointCloud, GrayPointCloud, CV_BGR2GRAY);
	GrayPointCloud.convertTo(GrayPointCloud, CV_8UC1);
	imshow("Reconstruct3DPoint", GrayPointCloud);
	waitKey();
	return 0;
}