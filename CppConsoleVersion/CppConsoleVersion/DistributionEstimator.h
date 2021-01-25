#pragma once
#ifndef _DISTRIBUTION_ESTIMATOR_H
#define _DISTRIBUTION_ESTIMATOR_H

#include "Util.h"

class DistributionEstimator {

private:

	// Insance of class
	static DistributionEstimator *instance;

	// Image reference and it's gradient
	cv::Mat* img;
	cv::Mat* gradient;


	// Default = 25
	int representation_threshold;

	DistributionEstimator(cv::Mat* img, cv::Mat* gradient, cv::Mat* representation_score, int representation_threshold);

	// Calculates mahalanobis distribution between the image and a distribution
	cv::Mat getMahalanobisImageDistribution(Distribution& distribution);

public:
	// 2D Matrix 
	cv::Mat* representation_score;

	~DistributionEstimator();

	// Returns Singelton Instance
	static DistributionEstimator* getInstance(cv::Mat* img, cv::Mat* gradient, cv::Mat* representation_score, int representation_threshold);

	// Bin Elections
	VotesData VoteOnBin();

	// Getting Winner Bin Pixel Coords
	std::vector<std::pair<int,int>> GetPopularBinPixelCoords(VotesData& votes_data);

	// Picking The Best Seed Pixel
	std::pair<int, int> GetSeedPixel(VotesData& votes_data, std::vector<std::pair<int, int>>& points_coords, int size = 10);

	// Estimates Distribution
	Distribution EstimateDistribution(std::pair<int, int>& seed_coords, int size = 3);

	// Estimates The representation score
	cv::Mat EstimateRepresentationScore(std::vector<Distribution>& distributions);

};

#endif