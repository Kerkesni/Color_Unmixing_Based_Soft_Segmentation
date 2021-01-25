#include "ColorUnmixer.h"

ColorUnmixer* ColorUnmixer::instance = nullptr;

ColorUnmixer::ColorUnmixer() {
}

ColorUnmixer::~ColorUnmixer() {
}

// Returns singleton instance of class
ColorUnmixer* ColorUnmixer::getInstance() {

	if (!instance) {
		instance = new ColorUnmixer();
		return instance;
	}
	return instance;
}

std::vector<double> ColorUnmixer::GetInitialValuesOfLayersForPixel(cv::Vec3b& pixelColor, std::vector<Distribution>& distributions)
{
	// Initializing values vecor
	std::vector<double> values;

	// Initializing distances vector
	std::vector<float> distances;

	// Initializing Alpha and Color Vectors
	std::vector<double> alphas;
	std::vector<double> colors;

	// Pixel Color to Mat
	cv::Mat pixelColorMat(1, 3, CV_32F);
	pixelColorMat.at<float>(0, 0) = (float)pixelColor[0];
	pixelColorMat.at<float>(0, 1) = (float)pixelColor[1];
	pixelColorMat.at<float>(0, 2) = (float)pixelColor[2];

	// Looping over distributions
	for (Distribution dist : distributions) {

		// Calculating distance between point and distribution
		float distance = (float)cv::Mahalanobis(pixelColorMat, dist.mean, dist.covariance);

		// Pushing to vector
		distances.push_back(distance);
	}

	// Getting the best fitting distribution's index (small distance)
	int argmin = (int)std::distance(distances.begin(), std::min_element(distances.begin(), distances.end()));

	// Setting alpha values
	for (int i = 0; i < distributions.size(); i++) {

		// If best Distribution
		if (i == argmin) {

			alphas.push_back(1.f);

			// Range Translation
			std::vector<double> col = TranslateColorVectorToRange(pixelColor, false);
			colors.insert(colors.end(), col.begin(), col.end());
		}
		else {

			alphas.push_back(0.f);

			// Range Translation
			std::vector<double> col = TranslateColorMatToRange(distributions[i].mean, false);
			colors.insert(colors.end(), col.begin(), col.end());
		}
	}

	// Appending value to final vector
	values.insert(values.end(), alphas.begin(), alphas.end());
	values.insert(values.end(), colors.begin(), colors.end());

	return values;
}

double ColorUnmixer::Ga(cv::Mat& alphas, int num_layers) {

	// Initializing deviation
	float deviation = 0;

	// Calculating deviation
	for (int i = 0; i < num_layers; i++) {
		deviation += alphas.at<double>(i);
	}
	
	deviation -= 1;

	return (float) (deviation * deviation);
}

cv::Mat ColorUnmixer::Gu(cv::Mat& alphas, cv::Mat& colors, cv::Vec3b& PixelColor, int num_layers) {

	// Transforming pixel layer color to a matrix
	cv::Mat PixelColorMat(1, 3, CV_64F);
	PixelColorMat.at<double>(0, 0) = (double) PixelColor[0];
	PixelColorMat.at<double>(0, 1) = (double)PixelColor[1];
	PixelColorMat.at<double>(0, 2) = (double)PixelColor[2];

	// Initializing deviation
	cv::Mat deviation(1, 3, CV_64F);
	deviation = 0;

	// Calculating deviation
	for (int i = 0; i < num_layers; i++) {

		// Transforming pixel layer color to a matrix
		cv::Mat layerPixelColorMat(1, 3, CV_64F);
		layerPixelColorMat.at<double>(0, 0) = colors.at<double>(0, i * 3 + 0);
		layerPixelColorMat.at<double>(0, 1) = colors.at<double>(0, i * 3 + 1);
		layerPixelColorMat.at<double>(0, 2) = colors.at<double>(0, i * 3 + 2);

		deviation += alphas.at<double>(i)* layerPixelColorMat;
	}

	deviation -= PixelColorMat;

	cv::pow(deviation, 2, deviation);

	return deviation;
}

cv::Mat ColorUnmixer::G(cv::Mat& values, cv::Vec3b& PixelColor, int num_layers) {

	// Transforming array into mat
	cv::Mat alphas(1, num_layers, CV_64F);
	cv::Mat colors(1, num_layers * 3, CV_64F);

	for (int i = 0; i < num_layers; i++) {
		alphas.at<double>(0, i) = values.at<double>(i, 0);
	}
	for (int i = 0; i < num_layers * 3; i++) {
		colors.at<double>(0, i) = values.at<double>(num_layers+i, 0);
	}

	// Initializing deviation
	cv::Mat general_deviation(1, 4, CV_64F);

	// Calculating and adding color deviation to vector
	cv::Mat color_deviation = Gu(alphas, colors, PixelColor, num_layers);
	for (int col = 0; col < 3; col++) {
		general_deviation.at<double>(0, col) = color_deviation.at<double>(0, col);
	}

	// Calculating and adding alpha deviation to vector
	double alpha_deviation = Ga(alphas, num_layers);
	general_deviation.at<double>(0, 3) = alpha_deviation;

	return general_deviation;
}

// Energy Function Class
class UnmixEnergy : public cv::MinProblemSolver::Function {
public:

	std::vector<Distribution> distributions;
	cv::Vec3b imgPoint;
	float rho;
	cv::Mat lambda;

	UnmixEnergy(std::vector<Distribution>& distributions, cv::Vec3b& imgPoint, float& rho, cv::Mat& lambda) {
		this->distributions = distributions;
		this->imgPoint = imgPoint;
		this->rho = rho;
		this->lambda = lambda;
	}
	int getDims() const {
		return distributions.size()+ distributions.size()*3;
	};

	double CalculateEnergy(cv::Mat& alphas, cv::Mat& colors) const
	{
		// Number of distributions
		int nb_distributions = this->distributions.size();

		// Initialize energy
		double energy = 0;

		// Calculating first term of energy function
		for (int i = 0; i < nb_distributions; i++) {

			// Transforming pixel layer color to a matrix
			cv::Mat layerPixelColorMat(1, 3, CV_32F);
			layerPixelColorMat.at<float>(0, 0) = (float)colors.at<double>(0, i * 3 + 0);
			layerPixelColorMat.at<float>(0, 1) = (float)colors.at<double>(0, i * 3 + 1);
			layerPixelColorMat.at<float>(0, 2) = (float)colors.at<double>(0, i * 3 + 2);

			// Calculating distance to distribution
			double maha_dist = (double)cv::Mahalanobis(layerPixelColorMat, this->distributions[i].mean, this->distributions[i].covariance);

			// Adding to energy
			energy += alphas.at<double>(i) * maha_dist;
		}

		// Calculating the second part of energy function
		double sparcity_term = 0.f;
		double top_part = 0.f;
		double bottom_part = 0.f;

		for (int i = 0; i < nb_distributions; i++) {
			top_part += alphas.at<double>(i);
			bottom_part += alphas.at<double>(i)*alphas.at<double>(i);
		}

		if (top_part == 0 || bottom_part == 0)
			sparcity_term = 0;
		else
			sparcity_term = 10 * ((top_part / bottom_part) - 1);

		// Calculating energy value
		energy += sparcity_term;

		return energy;
	}

	double Ga(cv::Mat& alphas) const{

		// Initializing deviation
		float deviation = 0;

		// Calculating deviation
		for (int i = 0; i < this->distributions.size(); i++) {
			deviation += alphas.at<double>(i);
		}

		deviation -= 1;

		return (float)(deviation * deviation);
	}

	cv::Mat Gu(cv::Mat& alphas, cv::Mat& colors) const{

		// Transforming pixel layer color to a matrix
		cv::Mat PixelColorMat(1, 3, CV_64F);
		PixelColorMat.at<double>(0, 0) = (double)this->imgPoint[0];
		PixelColorMat.at<double>(0, 1) = (double)this->imgPoint[1];
		PixelColorMat.at<double>(0, 2) = (double)this->imgPoint[2];

		// Initializing deviation
		cv::Mat deviation(1, 3, CV_64F);
		deviation = 0;

		// Calculating deviation
		for (int i = 0; i < this->distributions.size(); i++) {

			// Transforming pixel layer color to a matrix
			cv::Mat layerPixelColorMat(1, 3, CV_64F);
			layerPixelColorMat.at<double>(0, 0) = colors.at<double>(0, i * 3 + 0);
			layerPixelColorMat.at<double>(0, 1) = colors.at<double>(0, i * 3 + 1);
			layerPixelColorMat.at<double>(0, 2) = colors.at<double>(0, i * 3 + 2);

			deviation += alphas.at<double>(i)* layerPixelColorMat;
		}

		deviation -= PixelColorMat;

		cv::pow(deviation, 2, deviation);

		return deviation;
	}

	cv::Mat G(cv::Mat& alphas, cv::Mat& colors) const{

		// Initializing deviation
		cv::Mat general_deviation(1, 4, CV_64F);

		// Calculating and adding color deviation to vector
		cv::Mat color_deviation = Gu(alphas, colors);
		for (int col = 0; col < 3; col++) {
			general_deviation.at<double>(0, col) = color_deviation.at<double>(0, col);
		}

		// Calculating and adding alpha deviation to vector
		double alpha_deviation = Ga(alphas);
		general_deviation.at<double>(0, 3) = alpha_deviation;

		return general_deviation;
	}

	cv::Mat MinimizedFunction(cv::Mat& alphas, cv::Mat& colors) const
	{
		// Calculating Energy
		double energy = CalculateEnergy(alphas, colors);

		// Calculating diviation
		cv::Mat deviation = G(alphas, colors);

		cv::Mat result, g_lambda, rho_g;

		rho_g = cv::norm(deviation, cv::NORM_L2);
		cv::pow(rho_g, 2, rho_g);
		rho_g *= rho * 0.5f;

		cv::multiply(deviation, lambda, g_lambda);

		result = energy + g_lambda + rho_g;

		return result;
	}

	double calc(const double* x) const {
		// Transforming array into mat
		cv::Mat alphas(1, this->distributions.size(), CV_64F);
		cv::Mat colors(1, this->distributions.size()*3, CV_64F);

		for (int i = 0; i < this->distributions.size(); i++) {
			alphas.at<double>(0, i) = x[i];
		}		
		for (int i = 0; i < this->distributions.size()*3; i++) {
			colors.at<double>(0, i) = x[this->distributions.size()+i];
		}

		cv::Mat res = this->MinimizedFunction(alphas, colors);

		return (double)cv::norm(res, cv::NORM_L2);
	}

	void getGradient(const double *x, double *grad) {
		std::vector<double> res;
		for (int j = 0; j < sizeof(x); j++) {
			int j_left = j - 1;
			int j_right = j + 1;
			if (j_left < 0) {
				j_left = 0;
				j_right = 1;
			}
			if (j_right >= sizeof(x)) {
				j_right = sizeof(x)) - 1;
				j_left = j_right - 1;
			}
			// gradient value at position j
			double dist_grad = (x[j_right] - x[j_left]) / 2.0;
			res.push_back(dist_grad);
		}
	}
};

cv::Mat ColorUnmixer::MinimizePixelValues(cv::Vec3b& imgPoint, std::vector<Distribution>& distributions)
{
	
	// Original Method of Multipliers parameter initialization
	float rho = 0.1f;
	cv::Mat lambda(1, 4, CV_64F);
	lambda = 0.1f;
	float beta = 10.f;
	float epsilon = 0.25f;
	float sigma = 0.00001f;

	// Initializing Solver
	cv::Ptr<cv::ConjGradSolver> solver = cv::ConjGradSolver::create();
	cv::Ptr<cv::ConjGradSolver::Function> energyfunction;

	// Getting initial layer values
	std::vector<double> x_vec = GetInitialValuesOfLayersForPixel(imgPoint, distributions);
	// To matrix
	cv::Mat x(x_vec.size(), 1, CV_64F, &x_vec[0]);

	// Minimization
	while (true) {
		// Setting the minimized function class with new params
		energyfunction = new UnmixEnergy(distributions, imgPoint, rho, lambda);
		//solver->setTermCriteria(cv::TermCriteria::TermCriteria(cv::TermCriteria::Type::EPS, 30, epsilon));
		solver->setFunction(energyfunction);

		// Calculating deviation
		cv::Mat Gk = G(x, imgPoint, distributions.size());

		// Storing previous state
		cv::Mat Xkm1 = x.clone();

		// Minimizing Point
		solver->minimize(x);

		// Clip values to [0, 1]
		clip(&x);

		// Verifiying completion
		if (cv::norm(x - Xkm1, cv::NORM_L2) < sigma) {
			std::cout << x << std::endl;
			return x;
		}
		else {
			cv::Mat Gkm1 = Gk.clone();
			Gk = G(x, imgPoint, distributions.size());

			lambda += rho * Gk;

			rho = cv::norm(Gk, cv::NORM_L2) > epsilon * cv::norm(Gkm1, cv::NORM_L2) ? beta * rho : rho;
		}

	}
}
