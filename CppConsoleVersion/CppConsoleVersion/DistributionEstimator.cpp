#include "DistributionEstimator.h"

DistributionEstimator* DistributionEstimator::instance = nullptr;

DistributionEstimator::DistributionEstimator(cv::Mat * img, cv::Mat * gradient, cv::Mat* representation_score, int representation_threshold) :
	img{ img }, gradient{ gradient }, representation_score{ representation_score }, representation_threshold{ representation_threshold }{
}

DistributionEstimator::~DistributionEstimator()
{
}

DistributionEstimator* DistributionEstimator::getInstance(cv::Mat * img, cv::Mat * gradient, cv::Mat* representation_score, int representation_threshold)
{
	if (!instance) {
		instance = new DistributionEstimator(img, gradient, representation_score, representation_threshold);
	}
	return instance;
}

VotesData DistributionEstimator::VoteOnBin()
{
	// Image Size
	int rows = img->rows;
	int cols = img->cols;

	// Votes Per Bin HashMap
	std::map<double, int> votes_per_bin = {};

	// Votes Matrix
	double** votes = new double* [rows];
	for (int j = 0; j < rows; j++)
		votes[j] = new double[cols];

	// Looping Over All Pixels
	for (int j = 0; j < rows; j++) {
		for (int i = 0; i < cols; i++) {

			// Ignoring pixel if well represented
			if (representation_score->at<uchar>(j,i) < representation_threshold)
				continue;
			
			// Vote on bin
			votes[j][i] = exp(-gradient->at<uchar>(j, i)) * exp(-representation_score->at<uchar>(j, i));
			

			// Adding vote to HashMap
			auto search = votes_per_bin.find(votes[j][i]);
			if (search != votes_per_bin.end()) {
				votes_per_bin[votes[j][i]] += 1;
			}
			else {
				votes_per_bin.emplace(votes[j][i], 1);
			}

		}
	}

	// If No Votes return empty results
	if (votes_per_bin.empty()) {

		VotesData data;
		data.winner_bin = 0.f;
		data.votes = nullptr;

		// Deleting Unused Allocated Space
		delete votes;

		return data;

	}
	else {
		// Else Get Most Popular Bin
		std::pair<double, int> winner_bin = findEntryWithLargestValue(votes_per_bin);
		VotesData data;
		data.winner_bin = winner_bin.first;
		data.votes = votes;
		return data;
	}
}

std::vector<std::pair<int, int>> DistributionEstimator::GetPopularBinPixelCoords(VotesData & votes_data)
{
	// Image Size
	int rows = img->rows;
	int cols = img->cols;

	// Initializing vector
	std::vector<std::pair<int, int>> coords;

	// Looping Over All Pixels
	for (int j = 0; j < rows; j++) {
		for (int i = 0; i < cols; i++) {

			// Adding Pixel Coords if it voted on winner bin
			if (votes_data.votes[j][i] == votes_data.winner_bin)
				coords.push_back(std::make_pair(j, i));
		}
	}

	return coords;
}

std::pair<int, int> DistributionEstimator::GetSeedPixel(VotesData& votes_data, std::vector<std::pair<int, int>>& points_coords, int size)
{
	// Image Size
	int rows = img->rows;
	int cols = img->cols;

	std::vector<float> scores;

	for (std::pair<int, int> pixel : points_coords) {
		// Defining the 20x20 Kernel edges

		// Height
		int lower_h = (pixel.first - size) >= 0 ? (pixel.first - size) : 0;
		int higher_h = (pixel.first + size) < rows ? (pixel.first + size) : rows-1;

		// Width
		int lower_l = (pixel.second - size) >= 0 ? (pixel.second - size) : 0;
		int higher_l = (pixel.second + size) < cols ? (pixel.second + size) : cols - 1;

		// Looping Over Kernel pixels
		int neighbors_in_bin = 0;
		for (int j = lower_h; j < higher_h; j++) {
			for (int i = lower_l; i < higher_l; i++) {
				if (votes_data.votes[j][i] == votes_data.winner_bin)
					neighbors_in_bin += 1;
			}
		}
		
		// Calculating score of pixel
		float score = neighbors_in_bin * exp(-(float)gradient->at<uchar>(pixel.first,pixel.second));
		scores.push_back(score);

	}
	// Getting Pixel Coords with Heighest Score
	int argmax = (int) std::distance(scores.begin(), std::max_element(scores.begin(), scores.end()));
	return points_coords.at(argmax);
}

Distribution DistributionEstimator::EstimateDistribution(std::pair<int, int>& seed_coords, int size)
{
	// Image size
	int rows = img->rows;
	int cols = img->cols;

	// Defining patch edges
	// Height
	int lower_h = (seed_coords.first - size) >= 0 ? (seed_coords.first - size) : 0;
	int higher_h = (seed_coords.first + size) < rows ? (seed_coords.first + size) : rows - 1;

	// Width
	int lower_l = (seed_coords.second - size) >= 0 ? (seed_coords.second - size) : 0;
	int higher_l = (seed_coords.second + size) < cols ? (seed_coords.second + size) : cols - 1;

	// Setting ranges of Patch
	cv::Range h_range(lower_h, higher_h);
	cv::Range l_range(lower_l, higher_l);

	// Building patch
	cv::Mat patch(*img, std::vector<cv::Range>{h_range, l_range});
	// Making Patch Continuous
	patch = patch.clone();
	// Flattening patch
	patch = patch.reshape(1, patch.cols*patch.rows);
	
	// Calculating mean and covariance matrix
	cv::Mat mean, covar;
	cv::calcCovarMatrix(patch, covar, mean, cv::COVAR_ROWS | cv::COVAR_NORMAL | cv::COVAR_SCALE, CV_8U);
	// Inverting cavariance matrix
	cv::invert(covar, covar, cv::DECOMP_SVD);

	Distribution dist;
	dist.mean = mean;
	dist.covariance = covar;

	return dist;
}

cv::Mat DistributionEstimator::getMahalanobisImageDistribution(Distribution& distribution)
{	
	// Image Size
	int rows = img->rows;
	int cols = img->cols;

	cv::Mat distances(rows, cols, CV_8UC1);

	for (int j = 0; j < rows; j++) {
		for (int i = 0; i < cols; i++) {

			// Ignoring well represented pixels
			if (representation_score->at<uchar>(j, i) < representation_threshold)
				continue;

			// Pixel Color to Mat
			cv::Mat pixelColor(1, 3, CV_32F);
			pixelColor.at<float>(0, 0) = (float)img->at<cv::Vec3b>(j, i)[0];
			pixelColor.at<float>(0, 1) = (float)img->at<cv::Vec3b>(j, i)[1];
			pixelColor.at<float>(0, 2) = (float)img->at<cv::Vec3b>(j, i)[2];


			// Calculating Maalanobis Distance between distribution mean and pixel color
			uchar dist = (uchar)cv::Mahalanobis(pixelColor, distribution.mean, distribution.covariance);

			// Getting minimum representation value 
			uchar min_dist = std::min(dist, representation_score->at<uchar>(j, i));

			// Adding Value to Matrix
			distances.at<uchar>(j, i) = min_dist;

		}
	}
	return distances;
}


cv::Mat DistributionEstimator::EstimateRepresentationScore(std::vector<Distribution>& distributions)
{
	// Calculating Mahalanobis distance between image and last distribution
	cv::Mat maha_dist = getMahalanobisImageDistribution(distributions.back());
	 
	// If only one distribution exist
	// Return mahalanobis distance
	if (distributions.size() == 1)
		return maha_dist;

	cv::Mat rep_score = representation_score->clone();

	// Looping over each pair of different distributions
	for (int dist1 = 0; dist1 < distributions.size(); dist1++) {
		for (int dist2 = 0; dist2 < distributions.size(); dist2++) {
		
			// Skip if same distribution
			if (dist1 == dist2)
				continue;

			cv::Mat n;
			cv::subtract(distributions[dist1].mean, distributions[dist2].mean, n);

			// Looping Over Pixels
			for (int j = 0; j < img->rows; j++) {
				for (int i = 0; i < img->cols; i++) {

					// Ignoring well represented pixels
					if (representation_score->at<uchar>(j, i) < representation_threshold)
						continue;

					cv::Mat u1, u2, tmp1, tmp2;

					// Pixel Color to Mat
					cv::Mat pixelColor(1, 3, CV_32F);
					pixelColor.at<float>(0, 0) = (float)img->at<cv::Vec3b>(j, i)[0];
					pixelColor.at<float>(0, 1) = (float)img->at<cv::Vec3b>(j, i)[1];
					pixelColor.at<float>(0, 2) = (float)img->at<cv::Vec3b>(j, i)[2];

					// Calculating U1
					// ImgPoint - DistMean
					tmp1 = pixelColor - distributions[dist1].mean;
					// (ImgPoint - DistMean) * n
					cv::multiply(tmp1, n, tmp1);
					// n * n
					cv::multiply(n, n, tmp2);
					// ((ImgPoint - DistMean) * n) / n * n
					cv::divide(tmp1, tmp2, tmp1);
					// ImgPoint - (((ImgPoint - DistMean) * n) / n * n)*n
					u1 = pixelColor - tmp1;

					// Calculating U2
					// ImgPoint - DistMean
					tmp1 = pixelColor - distributions[dist2].mean;
					// (ImgPoint - DistMean) * n
					cv::multiply(tmp1, n, tmp1);
					// ((ImgPoint - DistMean) * n) / n * n
					cv::divide(tmp1, tmp2, tmp1);
					// ImgPoint - (((ImgPoint - DistMean) * n) / n * n)*n
					u2 = pixelColor - tmp1;

					// Calculating Alpha
					double alpha1, alpha2;
					alpha1 = (cv::norm(pixelColor - u1, cv::NORM_L2) / cv::norm(u1 - u2, cv::NORM_L2));
					alpha2 = 1 - alpha1;

					// Calculating Mahalanobis Distances
					double maha_dist_1 = cv::Mahalanobis(pixelColor, distributions[dist1].mean, distributions[dist1].covariance);
					double maha_dist_2 = cv::Mahalanobis(pixelColor, distributions[dist2].mean, distributions[dist2].covariance);

					uchar energy = (uchar) (alpha1 * maha_dist_1 + alpha2 * maha_dist_2);

					rep_score.at<uchar>(j, i) = std::min({energy, maha_dist.at<uchar>(j,i), rep_score.at<uchar>(j, i)});

				}
			}

		}
	}
	return rep_score;
}
