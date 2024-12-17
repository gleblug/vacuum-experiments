#define constrain(amt,low,high) ((amt)<(low)?(low):((amt)>(high)?(high):(amt)))
#define T_0 300.0


double eps(double T) {
	T = constrain(T, 273.0, 3500.0);
	if (T < 1500.0) {
		double v = T / 1000.0;
		return 0.01804211
			- 0.005214754 * v
			+ 0.123332100 * v*v
			- 0.081413040 * v*v*v
			+ 0.057359380 * v*v*v*v
			- 0.016521790 * v*v*v*v*v;
	} else {
		return -0.02158799
			- 1.236257e-4 * T
			+ 4.143182e-7 * T*T
			- 2.342655e-10 * T*T*T
			+ 5.537822e-14 * T*T*T*T
			- 4.869300e-18 * T*T*T*T*T;
	}
}

double rho_lin(double T) {
	double rho_0 = 5.5e-8; // ohm*m
	double alpha = 5e-3; // 1/K
	return rho_0 * (1 + alpha * (T - T_0));
}

double rho(double T) {
	double v = T / 1000.0;
	return 1.03853549e-8 * (
		- 1 + 19.1 * v + 6.8 * v*v - 1.509 * v*v*v + 0.154 * v*v*v*v
	);
}