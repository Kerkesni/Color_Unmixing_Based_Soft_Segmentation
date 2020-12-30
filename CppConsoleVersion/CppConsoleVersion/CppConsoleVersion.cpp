#include "Util.h"
#include "DistributionEstimator.h"


int main(void)
{
	std::string path = "C:\\Users\\MrZanziba\\Desktop\\Cours\\Projet CSTI\\Color_Unmixing_Based_Soft_Segmentation\\assets\\b.jpg";
	cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
	cv::Mat gray, laplacian;

	if (!img.data) {
		std::cout << "Error ! can't read image in " << path << std::endl;
		return -1;
	}

	// Image Size
	int rows = img.rows;
	int cols = img.cols;

	// Reduce noise by blurring with a Gaussian filter ( kernel size = 3 )
	cv::GaussianBlur(img, gray, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);

	// Image RGB to GrayScale
	cv::cvtColor(gray, gray, cv::COLOR_RGB2GRAY);

	// Calculate Laplacian Gradient
	cv::Laplacian(gray, laplacian, CV_64F, 3, 1.0, 0.0, cv::BORDER_DEFAULT);
	// converting back to CV_8U
	convertScaleAbs(laplacian, laplacian);

	// Initializing Representation Threshold
	int representation_threshold = 25;

	// Initializing Representation Score with 255
	cv::Mat representation_score(cv::Size(cols, rows), CV_8UC1);
	representation_score = 255;

	// Badly Represented Pixels Count Initialization
	int unrepresented_pixels = GetNotWellRepresenedPixel(representation_score, representation_threshold);

	// Initializing Distribution vector
	std::vector<Distribution> distributions;

	DistributionEstimator* dist_estimator = DistributionEstimator::getInstance(&img, &laplacian, &representation_score, representation_threshold);

	int iteration = 1;

	// Do While All Pixels Not Well Represented
	while (unrepresented_pixels > 0) {

		// Estimate Distribution
		std::cout << "************** Iteration  " << iteration << " **************"<< std::endl;

		// Initiate Vote
		std::cout << "Initiating Vorting..." << std::endl;
		VotesData voting_data = dist_estimator->VoteOnBin();

		// Get Winner Bin Pixels
		std::vector<std::pair<int, int>> pixels_coords = dist_estimator->GetPopularBinPixelCoords(voting_data);

		// Get Seed Pixel
		std::cout << "Getting Seed Pixel..." << std::endl;
		std::pair<int, int> seed_coords = dist_estimator->GetSeedPixel(voting_data, pixels_coords);
		std::cout << "seed is : " << seed_coords.first << "," << seed_coords.second << std::endl;

		std::cout << "Estimating Distribution..." << std::endl;
		Distribution estimated_distribution = dist_estimator->EstimateDistribution(seed_coords);
		distributions.push_back(estimated_distribution);
		std::cout << "Estimated Distribution Mean : " << estimated_distribution.mean << std::endl;
		std::cout << "Estimated Distribution Covariance : " << estimated_distribution.covariance << std::endl;
		
		// Estimate Representation Score
		std::cout << "Estimating Representation Score..." << std::endl;
		representation_score = dist_estimator->EstimateRepresentationScore(distributions);

		// Verify if All Pixels Are Well Represented
		std::cout << "Counting Unrepresented Pixels..." << std::endl;
		unrepresented_pixels = GetNotWellRepresenedPixel(representation_score, representation_threshold);
		std::cout << "Number of unrepresented pixels : " << unrepresented_pixels << std::endl;

		//Store Representation Score Image
		cv::imwrite("./res_imgs/rep_" + std::to_string(iteration) + ".jpg", representation_score);

		iteration++;
	}

	//Storing distributions in a file
	SaveDistributionsToFile(distributions, "./res_data/dist.yml");

	return 0;

}
