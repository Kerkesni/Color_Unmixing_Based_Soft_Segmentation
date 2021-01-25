#include "Util.h"

// Helper Function to find the Entry with largest Value in a Map 
std::pair<double, int> findEntryWithLargestValue(std::map<double, int>& sampleMap)
{
	// Reference variable to help find the entry with the highest value 
	std::pair<double, int> entryWithMaxValue = std::make_pair(0, 0);

	// Iterate in the map to find the required entry 
	std::map<double, int>::iterator currentEntry;
	for (currentEntry = sampleMap.begin(); currentEntry != sampleMap.end(); ++currentEntry)
	{
		// If this entry's value is more than the max value 
		// Set this entry as the max 
		if (currentEntry->second > entryWithMaxValue.second)
		{
			entryWithMaxValue = std::make_pair(currentEntry->first, currentEntry->second);
		}
	}

	return entryWithMaxValue;
}

// Returns number of not well represented pixels
int GetNotWellRepresenedPixel(cv::Mat& representation_score, int representation_threshold) {
	
	int number_unrepresented_pixels = 0;

	// Looping Over Pixels
	for (int j = 0; j < representation_score.rows; j++) {
		for (int i = 0; i < representation_score.cols; i++) {
			if (representation_score.at<uchar>(j, i) > representation_threshold)
				number_unrepresented_pixels++;
		}
	}
	return number_unrepresented_pixels;
}

// Saves Distributions Vector in a file
void SaveDistributionsToFile(std::vector<Distribution>& distributions, std::string path)
{
	// Create File
	cv::FileStorage output_file(path, cv::FileStorage::WRITE);

	// Check if opened
	if (!output_file.isOpened()) {
		std::cerr << "Error, couldn't open the file" << std::endl;
		return;
	}

	// Writing the number of Distributions
	output_file << "Count" << (int)distributions.size();

	// Writing the vector content
	for (int i = 0; i < distributions.size(); i++) {
		output_file << "Distributions_" + std::to_string(i);
		output_file << "{" << "Mean" << distributions[i].mean;
		output_file << "Cov" << distributions[i].covariance << "}";
	}

	// Closing file
	output_file.release();
}

// Loads Distribution Vector from file
std::vector<Distribution> LoadDistributionsFromFile(std::string path) {
	
	// Read File
	cv::FileStorage input_file(path, cv::FileStorage::READ);

	// Check if opened
	if (!input_file.isOpened()) {
		std::cerr << "Error, couldn't open the file" << std::endl;
		return std::vector<Distribution>();
	}

	std::vector<Distribution> distributions;

	// Retreiving the number of distributions
	int dist_count;
	dist_count = (int)input_file["Count"];

	// Writing the vector content
	for (int i = 0; i < dist_count; i++) {

		// Getting the distribution node
		cv::FileNode dist_file = input_file["Distributions_" + std::to_string(i)];

		// Retreiving Distribution values
		Distribution dist;
		dist.mean = dist_file["Mean"].mat();
		dist.covariance = dist_file["Cov"].mat();

		// Adding Distribution to vector
		distributions.push_back(dist);
	}

	// Closing file
	input_file.release();

	return distributions;
}

// Method for translating floats into range
float TranslateValueToRange(float value, bool reversed) {
	
	// From 0-1 to 0-255
	if (reversed)
		return value * 255;
	else
		return value / 255;
}

// Method for translating pixel color values to range
std::vector<double> TranslateColorVectorToRange(cv::Vec3d oldVec, bool reversed) {
	
	std::vector<double> newVec;

	// From 0-1 to 0-255
	for (int i = 0; i < 3; i++) {
		newVec.push_back(TranslateValueToRange(oldVec[i], reversed));
	}
	
	return newVec;
}

std::vector<double> TranslateColorMatToRange(cv::Mat oldVec, bool reversed) {

	std::vector<double> newVec;

	// From 0-1 to 0-255
	for (int i = 0; i < 3; i++) {
		newVec.push_back(TranslateValueToRange(oldVec.at<float>(0, i), reversed));
	}

	return newVec;
}

void clip(cv::Mat *mat, int min, int max) {

	for (int i = 0; i < mat->rows; i++) {
		double value = mat->at<double>(i, 0);

		if (value > max)
			mat->at<double>(i, 0) = (double)max;
		else if (value < min)
			mat->at<double>(i, 0) = (double)min;
	}
}