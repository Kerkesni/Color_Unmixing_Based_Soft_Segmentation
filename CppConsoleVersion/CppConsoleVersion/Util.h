#pragma once
#ifndef _UTIL_H
#define _UTIL_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <vector>
#include <iostream>
#include <map>
#include <fstream>
#include "opencv2/core/optim.hpp"

struct VotesData {
	double winner_bin;
	double **votes;
};

struct Distribution {
	cv::Mat mean;
	cv::Mat covariance;
};

// Helper Function to find the Entry with largest Value in a Map 
std::pair<double, int> findEntryWithLargestValue(std::map<double, int>& sampleMap);

// Returns number of not well represented pixels
int GetNotWellRepresenedPixel(cv::Mat& representation_score, int representation_threshold);

// Saves Distributions Vector in a file
void SaveDistributionsToFile(std::vector<Distribution>& distributions, std::string path);

// Loads Distribution Vector from file
std::vector<Distribution> LoadDistributionsFromFile(std::string path);

// Method for translating floats into range
float TranslateValueToRange(float value, bool reversed);

// Method for translating pixel color values to range
std::vector<double> TranslateColorVectorToRange(cv::Vec3d oldVec, bool reversed);
std::vector<double> TranslateColorMatToRange(cv::Mat oldVec, bool reversed);

// Clips matrix to range
void clip(cv::Mat* mat, int min = 0, int max = 1);

#endif