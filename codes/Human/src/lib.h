#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <cmath>
#include <random>
#include <vector>
#include <string>
#include <fftw3.h>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <iomanip>
#include <valarray>
#include <iostream>
#include <algorithm>
#include <sys/stat.h>
#include <sys/time.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_gamma.h>

#define REAL 0
#define IMAG 1

#define RANDOM gsl_rng_uniform(gsl_rng_r)
#define RANDOM_INT(A) gsl_rng_uniform_int(gsl_rng_r, A)
#define RANDOM_GAUSS(S) gsl_ran_gaussian(gsl_rng_r, S)
#define RANDOM_POISSON(M) gsl_ran_poisson(gsl_rng_r, M)
#define INITIALIZE_RANDOM_CLOCK(seed)              \
	{                                              \
		gsl_rng_env_setup();                       \
		if (!getenv("GSL_RNG_SEED"))               \
			gsl_rng_default_seed = time(0) + seed; \
		gsl_rng_T = gsl_rng_default;               \
		gsl_rng_r = gsl_rng_alloc(gsl_rng_T);      \
	}
#define INITIALIZE_RANDOM_F(seed)             \
	{                                         \
		gsl_rng_env_setup();                  \
		if (!getenv("GSL_RNG_SEED"))          \
			gsl_rng_default_seed = seed;      \
		gsl_rng_T = gsl_rng_default;          \
		gsl_rng_r = gsl_rng_alloc(gsl_rng_T); \
	}
#define FREE_RANDOM gsl_rng_free(gsl_rng_r)

static const gsl_rng_type *gsl_rng_T;
static gsl_rng *gsl_rng_r;

using std::cout;
using std::endl;
using std::string;
using std::to_string;
using std::valarray;
using std::vector;

extern unsigned seed;

// using dim1 = std::vector<double>;
// using dim2 = std::vector<std::vector<double>>;
typedef std::vector<double> dim1;
typedef std::vector<std::vector<double>> dim2;
typedef std::vector<std::vector<long unsigned int>> dim2I;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SRD;
typedef Eigen::Matrix<unsigned long int, Eigen::Dynamic, Eigen::Dynamic> Mdim2I;
typedef std::vector<int> vecInt;
typedef std::vector<vecInt> vecInt2;
typedef std::valarray<double> valD;
// typedef std::valarray<double> va;

bool fileExists(const std::string &filename);
Eigen::MatrixXd read_matrix(int, std::string);
std::vector<std::vector<double>> kuramoto_correlation(const std::vector<double> &x);
std::vector<double> arange(const double start, const double end, const double step);

double get_wall_time();
double get_cpu_time();
void display_timing(double wtime, double cptime);
void print_matrix(const dim2 &A, std::string filename);

vector<vector<int>> nodes_of_each_cluster(vector<int> &clusters);
std::vector<std::vector<int>> read_nodes_of_each_cluster(const std::string, const int);
std::vector<std::vector<int>> adjmat_to_adjlist(const Eigen::MatrixXd &);

void write_matrix_to_file(
	const string fname,
	Eigen::MatrixXd &m);

std::vector<int> read_from_file(std::string filename);
void find_dominant_frequency(const dim1 &,
							 const dim1 &,
							 double &,
							 double &);

inline double mean(const std::vector<double> &vec, const int id)
{
	/*average the vector from element "id" to end of the vector */
	return accumulate(vec.begin() + id, vec.end(), 0.0) / (vec.size() - id);
}
void print2DVector(const dim2 &vec2d,
				   const std::string fileName);
				   
void fftRealSignal(const dim1 &signal,
				   valarray<double> &freq,
				   valarray<double> &freqAmplitudes,
				   const double dt,
				   int FRACTION);
#endif
