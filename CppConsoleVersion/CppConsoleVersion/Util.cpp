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
	// Open / Create File
	cv::FileStorage output_file(path, cv::FileStorage::WRITE);

	// Check if opened
	if (!output_file.isOpened()) {
		std::cerr << "Error, couldn't open the file" << std::endl;
		return;
	}

	for (int i = 0; i < distributions.size(); i++) {
		output_file << "Mean_" + std::to_string(i) << distributions[i].mean;
		output_file << "Cov_" + std::to_string(i) << distributions[i].covariance;
	}

	// Close file and release memory buffer
	output_file.release();
}

// Loads Distribution Vector from file
std::vector<Distribution> LoadDistributionsFromFile(std::string path) {

}