/*

##############################          sources          ##############################

	1. https://stackoverflow.com/questions/19443908/detecting-how-blurred-an-image-is
	2. https://learnopencv.com/automatic-red-eye-remover-using-opencv-cpp-python/

*/

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/objdetect.hpp>

cv::Mat maxSecDer(const cv::Mat& img) {
	cv::Mat kernelX = (cv::Mat_<float>(1, 3) << 1, -2, 1);
	cv::Mat kernelY = (cv::Mat_<float>(3, 1) << 1, -2, 1);

	cv::Mat dx;
	cv::Mat dy;
	filter2D(img, dx, -1, kernelX);
	filter2D(img, dy, -1, kernelY);
	return abs(dx) + abs(dy);
}

void fillHoles(cv::Mat& mask) {
	cv::Mat maskFloodfill = mask.clone();
	floodFill(maskFloodfill, cv::Point(0, 0), cv::Scalar(255));
	cv::Mat mask2;
	bitwise_not(maskFloodfill, mask2);
	mask = (mask2 | mask);
}

std::vector <cv::Mat> input(std::string filename) {
	std::ifstream fin(filename);

	std::vector <cv::Mat> images;
	int str_count = 1;

	for (std::string image_name; fin >> image_name; ) {
		if (fin.bad()) { std::cerr << "File read/write error\n"; break; }
		if (fin.fail()) { std::cerr << "Invalid data format. Line in file: " << str_count << std::endl; }

		cv::Mat image = cv::imread(image_name);
		if (image.empty()) { std::cerr << "Can't open your image. Line in file: " << str_count << std::endl; }
		else { images.push_back(image.clone()); }

		if (fin.eof()) { break; }
		str_count++;
	}

	return images;
}

double blur_detector(cv::Mat im_rgb, bool& isBlur, double lvl = 3.5) {
	/*
		В зависимости от задачи lvl меняется. Порог для размытых фотографий 3.5 с таким методом.
	*/
	cv::Mat im_gray;
	cv::cvtColor(im_rgb, im_gray, CV_RGB2GRAY);

	cv::Mat d = maxSecDer(im_gray);							// считаем вторую производную

	// алгоритм №1
	cv::Mat hist;											// создаем гистограмму 
	int histSize = 256;										// Количество бинов гистограммы
	float range[] = { 0, 256 };								// Диапазон значений пикселей
	const float* histRange = { range };
	calcHist(&d, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);

	cv::Mat cumulativeHist;
	hist.copyTo(cumulativeHist);
	for (int i = 1; i < histSize; i++) { cumulativeHist.at<float>(i) += cumulativeHist.at<float>(i - 1); }

	double quantileVal = -1;								// Находим верхний квантиль (0.999)
	for (int i = 0; i < histSize; ++i) {
		if (cumulativeHist.at<float>(i) >= 0.999 * cumulativeHist.at<float>(histSize - 1)) {
			quantileVal = i;
			break;
		}
	}

	double psf_size = cv::sum(cv::abs(d))[0] / 10000;		// доп параметр для точности

	isBlur = false;
	if (psf_size / quantileVal < lvl) isBlur = true;
	return psf_size / quantileVal;
}

std::vector <cv::Mat> remove_red_eyes(std::vector <cv::Mat> images) {
	std::vector <cv::Mat> fixed_images;
	cv::CascadeClassifier eyesCascade("haarcascade_eye.xml");												// Load HAAR cascade

	for (size_t j = 0; j < images.size(); j++) {
		cv::Mat img = images[j].clone();
		cv::Mat imgOut = img.clone();

		std::vector<cv::Rect> eyes;																			// Detect eyes
		eyesCascade.detectMultiScale(img, eyes, 1.3, 4, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(100, 100));

		for (size_t i = 0; i < eyes.size(); i++) {
			cv::Mat eye = img(eyes[i]);																		// Extract eye from the image.
			std::vector <cv::Mat> bgr(3);																	// Split eye image into 3 channels
			split(eye, bgr);

			cv::Mat mask = (bgr[2] > 150) & (bgr[2] > (bgr[1] + bgr[0]));									// Simple red eye detector

			fillHoles(mask);																				// Clean mask -- 1) File holes 2) Dilate (expand) mask
			dilate(mask, mask, cv::Mat(), cv::Point(-1, -1), 3, 1, 1);

			cv::Mat mean = (bgr[0] + bgr[1]) / 2;															// Calculate the mean channel by averaging and the green and blue channels
			mean.copyTo(bgr[2], mask);
			mean.copyTo(bgr[0], mask);
			mean.copyTo(bgr[1], mask);

			cv::Mat eyeOut;																					// Merge channels
			cv::merge(bgr, eyeOut);

			eyeOut.copyTo(imgOut(eyes[i]));
		}

		fixed_images.push_back(imgOut);
	}

	return fixed_images;
}

int main() {
	std::vector <cv::Mat> images = input("images/.names.txt");

	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}