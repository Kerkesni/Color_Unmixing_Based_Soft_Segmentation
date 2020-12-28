#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

int main(void) {

	std::string path = "C:\\Users\\MrZanziba\\Desktop\\Cours\\Projet CSTI\\Color_Unmixing_Based_Soft_Segmentation\\assets\\a.jpg";
	cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);

	if (img.empty()) {
		std::cout << "Could not read image in " << path << std::endl;
		return -1;
	}

	cv::imshow("Image", img);
	cv::waitKey(0);

	return 0;
}