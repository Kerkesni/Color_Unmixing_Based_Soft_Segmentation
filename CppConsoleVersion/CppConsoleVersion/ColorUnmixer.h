#pragma once
#ifndef _COLOR_UNMIXER_H
#define _COLOR_UNMIXER_H

#include "Util.h"

// Singelton Class
class ColorUnmixer {

private:
	static ColorUnmixer* instance;

	ColorUnmixer();

	// returns initial color and alpha values of a pixel
	std::vector<double> GetInitialValuesOfLayersForPixel(cv::Vec3b& pixelColor, std::vector<Distribution>& distributions);

	// Alpha deviation constraint
	double Ga(cv::Mat& alphas, int num_layers);

	// Color deviation constraint
	cv::Mat Gu(cv::Mat& alphas, cv::Mat& colors, cv::Vec3b& PixelColor, int num_layers);

	// Generl deviation constraint
	cv::Mat G(cv::Mat& values, cv::Vec3b& PixelColor, int num_layers);

public:
	~ColorUnmixer();
	static ColorUnmixer* getInstance();
	
	// Minimizing Energy function of a pixel
	cv::Mat MinimizePixelValues(cv::Vec3b& imgPoint , std::vector<Distribution>& distributions);

};

#endif