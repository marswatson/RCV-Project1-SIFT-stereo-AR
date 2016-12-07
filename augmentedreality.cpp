#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
using namespace std;
using namespace cv;

int main()
{
	VideoCapture Camera;
	Camera.open(0);
	Mat capture;
	Mat image;
	Mat neg_img, cpy_img;
	int key = 0;

	Mat Picture = imread("dota.png");

	//chessboard is 6*8
	int b_width = 6;
	int b_height = 8;
	Size b_size(b_width, b_height);

	Mat warp_matrix(3, 3, CV_32FC1);
	vector<Point2f> corners;

	//First we use pictures to get augmented reality picture
	image = imread("chessboard.jpg");

	int found = findChessboardCorners(image, b_size, corners,
		CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

	if (corners.size() == 48)
	{
		//extract marching points from image and chessboard
		Point2f p[4];
		Point2f q[4];

		q[0].x = Picture.cols * 0;
		q[0].y = Picture.rows * 0;
		q[1].x = Picture.cols;
		q[1].y = Picture.rows * 0;
		q[2].x = Picture.cols;
		q[2].y = Picture.rows;
		q[3].x = Picture.cols * 0;
		q[3].y = Picture.rows;

		p[0].x = corners[0].x;
		p[0].y = corners[0].y;
		p[1].x = corners[5].x;
		p[1].y = corners[5].y;
		p[2].x = corners[47].x;
		p[2].y = corners[47].y;
		p[3].x = corners[42].x;
		p[3].y = corners[42].y;

		//Calculate warp_matrix
		warp_matrix = getPerspectiveTransform(q, p);
		Mat blank(Picture.size(), CV_8UC3, Scalar(255, 255, 255));

		//warp the image
		warpPerspective(Picture, neg_img, warp_matrix, image.size(), INTER_LINEAR, BORDER_CONSTANT);
		imshow("neg", neg_img);
		warpPerspective(blank, cpy_img, warp_matrix, image.size(), INTER_LINEAR, BORDER_CONSTANT);
		imshow("cpy", cpy_img);
		bitwise_not(cpy_img, cpy_img);
		bitwise_and(cpy_img, image, cpy_img);
		bitwise_or(cpy_img, neg_img, image);
		imshow("Augmented Reality From Picture", image);
	}

	//Then we use camera from laptop	
	Camera.open(0);
	while (key != 'p')
	{
		Camera.read(capture);
		image = capture;
		if (image.empty()) 
			break;

		int found = findChessboardCorners(image, b_size, corners,
			CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

		if (corners.size() == 48)
		{
			//extract matching points from image and chessboard
			Point2f p[4];
			Point2f q[4];

			q[0].x = Picture.cols * 0;
			q[0].y = Picture.rows * 0;
			q[1].x = Picture.cols;
			q[1].y = Picture.rows * 0;
			q[2].x = Picture.cols;
			q[2].y = Picture.rows;
			q[3].x = Picture.cols * 0;
			q[3].y = Picture.rows;

			p[0].x = corners[0].x;
			p[0].y = corners[0].y;
			p[1].x = corners[5].x;
			p[1].y = corners[5].y;
			p[2].x = corners[47].x;
			p[2].y = corners[47].y;
			p[3].x = corners[42].x;
			p[3].y = corners[42].y;

			//Calculate warp matrix
			warp_matrix = getPerspectiveTransform(q, p);
			Mat blank(Picture.size(), CV_8UC3, Scalar(255, 255, 255));

			//warp the image
			warpPerspective(Picture, neg_img, warp_matrix, image.size(), INTER_LINEAR, BORDER_CONSTANT);
			warpPerspective(blank, cpy_img, warp_matrix, image.size(), INTER_LINEAR, BORDER_CONSTANT);
			bitwise_not(cpy_img, cpy_img);
			bitwise_and(cpy_img, image, cpy_img);
			bitwise_or(cpy_img, neg_img, image);
			imshow("Augmented Reality From Camera", image);
		}
		else
		{
			imshow("Augmented Reality From Camera", image);
		}
		key = waitKey(1);
	}
	return 0;
}