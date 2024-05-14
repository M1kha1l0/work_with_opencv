#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    cv::Mat image = cv::imread("images/screenshot.png");

    cv::namedWindow("image1", cv::WINDOW_NORMAL);
    cv::imshow("image1", image);
    cv::waitKey(0);
    cv::destroyAllWindows();
}