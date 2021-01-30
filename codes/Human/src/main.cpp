#include "sdde_solver.h"
#include "lib.h"

unsigned seed;

int main(int argc, char const *argv[])
{

	if (argc < 3)
	{
		std::cerr << "input error \n";
		exit(2);
	}

	/*-----simulation parameters---------------------------------------------*/
	const int N = atoi(argv[1]);
	const double t_trans = atof(argv[2]);
	const double t_sim = atof(argv[3]);
	const double g = atof(argv[4]);
	const double muMean = atof(argv[5]);
	const double muStd = atof(argv[6]);
	const double noiseAmplitude = atof(argv[7]);
	const string net_label = argv[8];
	const int ncluster = atoi(argv[9]);
	const int numEnsembles = atoi(argv[10]);
	const bool PRINT_CORRELATION = atoi(argv[11]);
	const bool PRINT_COORDINATES = atoi(argv[12]);
	const double dt = atof(argv[13]);

	const double tinitial = 0.0;
	const double dtmax = 0.05;
	/*-----------------------------------------------------------------------*/

	seed = 1234;
	double wtime = get_wall_time();

	if (numEnsembles > 1)
	{
		INITIALIZE_RANDOM_CLOCK(seed);
	}
	else
	{
		INITIALIZE_RANDOM_F(seed);
	}

	int index_transition = (int)(t_trans / dt);
	dim1 hi(N);
	dim1 omega(N);
	dim1 r_global;
	dim1 R_local(ncluster);
	dim1 R_local_2(2);
	// const double sigma = 0.0;

	/*-----------------------------------------------------------------------*/
	const string subname = net_label + "-" + to_string(g) + "-" + to_string(muMean);
	const string adj_name = "../data/text/networks/" + net_label + "_connectmat.txt";
	const string delay_name = "../data/text/networks/" + net_label + "_delaymat.txt";
	const string R_FILE_NAME = "../data/text/R-" + subname + ".txt";
	const string COMM_L1_NAME = "../data/text/networks/" + net_label + "_comm_l1.txt";
	const string COMM_L2_NAME = "../data/text/networks/" + net_label + "_comm_l2.txt";

	FILE *R_FILE;

	R_FILE = fopen(R_FILE_NAME.c_str(), "a");
	if (!fileExists(R_FILE_NAME))
	{
		cout << "output file for r did not open correctly \n!";
		exit(EXIT_FAILURE);
	}

	vecInt clst = {N};
	vecInt2 local_nodes_l1 = read_nodes_of_each_cluster(COMM_L1_NAME, ncluster);
	vecInt2 local_nodes_l2 = read_nodes_of_each_cluster(COMM_L2_NAME, 2);
	vecInt2 global_nodes = nodes_of_each_cluster(clst);

	Eigen::MatrixXd Cij = read_matrix(N, adj_name);
	Eigen::MatrixXd Dij = read_matrix(N, delay_name);
	// exit(EXIT_SUCCESS);

	const double maxdelay = Dij.maxCoeff();

	for (int ens = 0; ens < numEnsembles; ens++)
	{

		printf("%s, g = %10.3f, omega = %10.3f, sim = %5d \n", net_label.c_str(), g, muMean, ens);
		for (int ii = 0; ii < N; ii++)
		{
			hi[ii] = RANDOM * 2 * M_PI - M_PI;
			omega[ii] = muStd * RANDOM_GAUSS(1) + muMean;
		}

		SDDE sdde(N);
		sdde.set_params(tinitial, t_sim, g, maxdelay, dtmax, noiseAmplitude);
		sdde.set_matrices(Cij, Dij);
		sdde.set_history(hi);
		sdde.set_initial_frequencies(omega);
		sdde.integrate(numEnsembles);

		if (PRINT_CORRELATION)
		{
			Eigen::MatrixXd cor = sdde.get_correlation();
			const string COR_FILE_NAME = "../data/text/c-" + subname +
										 "-" + to_string(ens);
			write_matrix_to_file(COR_FILE_NAME, cor);
		}

		r_global = sdde.interpolate_order_parameter(
			global_nodes[0], 1.0, t_sim - 5, dt, "linear");
		double R_global = mean(r_global, index_transition);
		// exit(EXIT_SUCCESS);

		for (int nl = 0; nl < ncluster; nl++)
		{
			dim1 r = sdde.interpolate_order_parameter(
				local_nodes_l1[nl], 1.0, t_sim - 5, dt, "linear");
			R_local[nl] = mean(r, index_transition);
		}
		for (int nl = 0; nl < 2; nl++)
		{
			dim1 r = sdde.interpolate_order_parameter(
				local_nodes_l2[nl], 1.0, t_sim - 5, dt, "linear");
			R_local_2[nl] = mean(r, index_transition);
		}

		sdde.interpolate_coordinates(t_trans, t_sim - 5.0, dt, "linear");
		if (PRINT_COORDINATES)
		{
			print2DVector(sdde.pt_x_intp,
						  "../data/text/Coor-" + subname + "-" + to_string(ens));
		}

		fprintf(R_FILE, "%15.9f %15.9f %15.9f", g, muMean, R_global);
		for (int nl = 0; nl < ncluster; nl++)
			fprintf(R_FILE, "%15.9f", R_local[nl]);
		for (int nl = 0; nl < 2; nl++)
			fprintf(R_FILE, "%15.9f", R_local_2[nl]);
		fprintf(R_FILE, "\n");
	}
	fclose(R_FILE);
	/*-----------------------------------------------------------------------*/
	FREE_RANDOM;
	wtime = get_wall_time() - wtime;
	display_timing(wtime, 0);

	return 0;
}
