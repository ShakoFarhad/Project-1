#include <iostream>
#include <cmath>
#include <armadillo>
#include <string>
#include <time.h>
#include <fstream>
#include <unistd.h>

using namespace arma;
using namespace std;

double sourceTerm(double x) {
    return 100.0*exp(-10.0*x);
}

double uExact(double x) {
    return 1.0-(1.0-exp(-10.0))*x-exp(-10.0*x);
}

void firstDerivative(double* a, double* b, double* c, double* f, int n) {
    //Setting up this finite difference: u_{i+1}-u_{i-1} = 2h*f

    double h = 1.0/(n+1);

    for(int i = 0; i < n+2; i++) {
        b[i] = 0.0;
        c[i] = 1.0;
        a[i] = -1.0;
        f[i] = 2.0*h*sourceTerm(i*h);
    }
}

void secondDerivative(double* a, double* b, double* c, double* f, int n) {
    //Setting up this finite difference: -u_{i+1}+2u_{i}-u_{i-1} = h^2*f

    double h = 1.0/(n+1);

    for(int i = 0; i < n+2; i++) {
        b[i] = 2.0;
        c[i] = -1.0;
        a[i] = -1.0;
        f[i] = h*h*sourceTerm(i*h);
    }
}

double* generalAlgorithm(int n, int derivative) {
    //This general algorithm solves the system -u' = f, and/or -u'' = f, with initial conditions u(0) = u(1) = 0.

    double* a = new double[n+2];
    double* b = new double[n+2];
    double* c = new double[n+2];
    double* f = new double[n+2];
    double* v = new double[n+2];

    //Initial conditions force us to set these to 0.
    a[0] = 0; b[0] = 0; c[0] = 0; v[0] = 0; f[0] = 0;
    a[n+1] = 0; b[n+1] = 0; c[n+1] = 0; v[n+1] = 0; f[n+1] = 0;

    if(derivative == 1) {
        firstDerivative(a,b,c,f,n);
    }
    if(derivative == 2) {
        secondDerivative(a, b, c, f, n);
    }

    for(int i=2; i < n+1; i++) {
        b[i] = b[i] - (a[i-1]*c[i-1])/b[i-1];

        f[i] = f[i] - (a[i-1]*f[i-1])/b[i-1];
    }

    v[n] = f[n]/b[n];
    for(int i=n-1; i > 0; i--) {
        v[i] = (f[i] - c[i]*v[i+1])/b[i];
    }
    delete[] a, delete[] b, delete[] c, delete[] f;
    return v;
}

double* specializedAlgorithm(int n) {
    //This specialized algorithm solves the system -u'' = f, u(0) = u(1) = 0.

    //Initializing the needed arrays for this specialized algorithm.
    double* b = new double[n+2];
    double* f = new double[n+2];
    double* v = new double[n+2];
    double h = 1.0/(n+1);

    for(int i = 0; i < n+2; i++) {
        f[i] = h*h*sourceTerm(i*h);
    }

    //Initial conditions force us to set these to 0.
    b[0] = 0; v[0] = 0; f[0] = 0;
    b[n+1] = 0; v[n+1] = 0; f[n+1] = 0;

    b[1] = 2;
    for(int i=2; i < n+1; i++) {
        b[i] = (i+1.0)/i;

        f[i] = f[i] + f[i-1]/b[i-1];
    }

    v[n] = f[n]/b[n];
    for(int i=n-1; i > 0; i--) {
        v[i] = (f[i] + v[i+1])/b[i];
    }
    delete[] b, delete[] f;
    return v;
}

double* centeredDifferenceSolver(int n) {
    double h = 1.0/(n+1);

    double* v = new double[n+2];

    //Initialising the boundary conditions, where we assume v[-1] = 0.
    v[0] = 0; v[n+1] = 0; v[1] = -h*h*sourceTerm(0.0)+2*v[0];

    //solving the system:
    for(int i=1; i< n+2; i++) {
        v[i+1] = -h*h*sourceTerm(i*h)+2*v[i]-v[i-1];
    }
    return v;
}

void convergenceRate(int numberOfRuns, int n, string scheme, int returnMaximum) {
    //Finding how well our scheme approximates the solution by looking at the convergence rate, r.

    double* r = new double[numberOfRuns - 1];
    double* h = new double[numberOfRuns];
    double* E = new double[numberOfRuns];

    double* v;

    if(returnMaximum == 0) {
        if(scheme == "specializedAlgorithm"){
            for(int j=0; j < numberOfRuns; j++) {
                double sum = 0.0;
                v = specializedAlgorithm(n);
                h[j] = 1.0/(n+1);

                for(int i=0; i < n+2; i++) {
                    sum = sum + pow((v[i] - uExact(i*h[j])), 2.0);
                }

                E[j] = sqrt(sum);

                n = n*4;
            }

        }

        if(scheme == "generalAlgorithm"){
            for(int j=0; j < numberOfRuns; j++) {
                double sum = 0.0;
                v = generalAlgorithm(n, 2);
                h[j] = 1.0/(n+1);

                for(int i=0; i < n+2; i++) {
                    sum = sum + pow((v[i] - uExact(i*h[j])), 2.0);
                }

                E[j] = sqrt(sum);

                n = n*4;
            }
        }

        if(scheme == "centeredDifferenceSolver"){
            for(int j=0; j < numberOfRuns; j++) {
                double sum = 0.0;
                v = centeredDifferenceSolver(n);
                h[j] = 1.0/(n+1);

                for(int i=0; i < n+2; i++) {
                    sum = sum + pow((v[i] - uExact(i*h[j])), 2.0);
                }

                E[j] = sqrt(sum);

                n = n*4;
            }

        }

        for(int i=0; i< numberOfRuns - 1; i++){
            r[i] = log(E[i+1]/E[i])/log(h[i+1]/h[i]);
        }

        cout << "Step length:                   Convergence rate:" << endl;
        cout << "------------------------------------------------" << endl;
        for(int i = 0; i < numberOfRuns -1; i++) {
            cout << "h = " << h[i] << "----> h = " << h[i + 1] << " gives r = " << r[i] << endl;

        }

    } else {
        double* maxError = new double[numberOfRuns];
        double temporaryMax;
        double absoluteValue;


        if(scheme == "centeredDifferenceSolver") {
            returnMaximum = 0;
            convergenceRate(numberOfRuns, n, scheme, returnMaximum);
        }

        if(scheme == "specializedAlgorithm"){
            for(int j=0; j < numberOfRuns; j++) {
                maxError[j] = 0.0;
                v = specializedAlgorithm(n);
                h[j] = 1.0/(n+1);
                double tol = 0.0001;

                for(int i = 0; i< n+2; i++) {
                    //|v_i - u_i|
                    absoluteValue = abs(v[i] - uExact(i*h[j]));

                    if(absoluteValue < (-1)*tol && absoluteValue > tol) { //We do not want log10(0)

                        if(uExact(i*h[j]) < (-1)*tol && uExact(i*h[j]) > tol) { //We do not want log(x/0)
                            //Updating temporaryMax
                            temporaryMax = log10(absoluteValue/abs(uExact(i*h[j])));

                            if(maxError[j] < temporaryMax) { //Checking if our current max value is less than temporaryMax
                                //Updating maxError
                                maxError[j] = temporaryMax;
                                cout <<"u = "<< uExact(i*h[j]) << endl;
                                cout <<"v = "<< v[i] << endl;

                            }
                        }
                    }
                }
                //Updating h
                n = n*10;
            }
        }

        if(scheme == "generalAlgorithm"){
            for(int j=0; j < numberOfRuns; j++) {
                maxError[j] = 0.0;
                v = generalAlgorithm(n, 2);
                h[j] = 1.0/(n+1);
                double tol = 0.0001;

                for(int i = 0; i< n+2; i++) {
                    //|v_i - u_i|
                    absoluteValue = abs(v[i] - uExact(i*h[j]));

                    if(absoluteValue < (-1)*tol && absoluteValue > tol) { //We do not want log10(0)

                        if(uExact(i*h[j]) < (-1)*tol && uExact(i*h[j]) > tol) { //We do not want log(x/0)
                            //Updating temporaryMax
                            temporaryMax = log10(absoluteValue/abs(uExact(i*h[j])));

                            if(maxError[j] < temporaryMax) { //Checking if our current max value is less than temporaryMax
                                //Updating maxError
                                maxError[j] = temporaryMax;
                                cout <<"u = "<< uExact(i*h[j]) << endl;
                                cout <<"v = "<< v[i] << endl;

                            }
                        }
                    }
                }
                //Updating h
                n = n*10;
            }
        }

        cout << "Step length:                      Maximum Error:" << endl;
        cout << "------------------------------------------------" << endl;
        for(int i = 0; i < numberOfRuns; i++) {
            cout << "h = " << h[i] << "             gives error = " << maxError[i] << endl;
        }

        delete[] maxError;
    }
    delete[] h, delete[] E;
    delete[] v, delete[] r;
}

double compareTime(int n, string scheme){
    //Comparing the cpu time for the two schemes.


    double* v;
    if(scheme == "specializedAlgorithm") {

        clock_t start, finish; //Declare start and stop time

        start = clock();

        v = specializedAlgorithm(n);

        finish = clock();

        delete[] v;

        return ((finish - start)/(double)CLOCKS_PER_SEC);

    } else if (scheme == "generalAlgorithm") {

        clock_t start, finish; //Declare start and stop time

        start = clock();

        v = generalAlgorithm(n, 2);


        finish = clock();

        delete[] v;

        return ((finish - start)/((double)CLOCKS_PER_SEC));

    } else {

        cout << "scheme name is incorrect. Send 'generalAlgorithm' or 'specializedAlgorithm' to the function.";
        return 0.0;
    }

}

void timeLU(int n) {

    clock_t start, finish; //Declare start and stop time

    start = clock();

    //Initialize LU decomposition
    mat L; mat U;
    mat A(n+2,n+2, fill::eye);
    A = 2*A;
    vec f = zeros<vec>(n+2);
    double h = 1.0/(n+1);

    for(int i = 0; i< n+1; i++) {
        A(i,i+1) = -1;
        f(i) = h*h*sourceTerm(i*h);
    }

    A = symmatu(A);

    lu(L,U,A);

    vec y = solve(L, f);
    vec x = solve(U, y);

    finish = clock();

    double timeLU = ((finish - start)/(double)CLOCKS_PER_SEC);

    cout << "LU decomposition solver time was: " << timeLU << endl;


    start = clock();

    double* v = specializedAlgorithm(n);

    finish = clock();

    double timeSA = ((finish - start)/(double)CLOCKS_PER_SEC);

    cout << "Specialized algorithm time was: " << timeSA << endl;

    delete[] v;


}

void oppgaveA() {
    int n = 10;
    int derivative = 2;

    double* v1;
    v1 = generalAlgorithm(n, derivative);
    double* v2;
    v2 = generalAlgorithm(n*10, derivative);
    double* v3;
    v3 = generalAlgorithm(n*100, derivative);


    for(int i = 0; i<n; i++) {
        cout << v1[i];
    }

    for(int i = 0; i<n*10; i++) {
        cout << v2[i];
    }

    for(int i = 0; i<n*100; i++) {
        cout << v3[i];
    }
}

void oppgaveB() {
    //Skriver verdier til en .txt fil og leser inn i python for å plotte.

    for(int n = 10; n < 10000; n=n*10) {
        string filename = "plotGeneralAlgorithm" + to_string(n) + ".txt";

        ofstream myfile;
        myfile.open (filename);

        double* v = generalAlgorithm(n, 2);

        for(int i = 0; i < n+2; i++) {

            myfile << to_string(v[i]) + " ";
        }
        myfile.close();
        delete[] v;
    }
}

void oppgaveC() {

    double* time = new double[6];


    string sA = "specializedAlgorithm";
    cout << "This is the cpu time for the specialized algorithm" << endl;
    for(int n=1000; n < 1000000000; n = n*10) {
        int j=0;
        time[j] = compareTime(n,sA);

        cout << "n = "<< n << ",  time = " << time[j] << endl;
        j++;
    }

    string gA = "generalAlgorithm";
    cout << "This is the cpu time for general algorithm" << endl;
    for(int n=1000; n < 1000000000; n = n*10) {
        int j=0;
        time[j] = compareTime(n, gA);

        cout << "n = "<< n << ",  time = " << time[j] << endl;
        j++;
    }
    delete[] time;
}

void oppgaveD() {
    int n = 100;
    int numberOfRuns = 5;
    //string scheme = "centeredDifferenceSolver";
    //string scheme = "specializedAlgorithm";
    string scheme = "generalAlgorithm";
    int returnMaximum = 0;

    convergenceRate(numberOfRuns, n, scheme, returnMaximum);
}

void oppgaveE() {
    //Teste tiden det tar for å kjøre LU decomposition og kjøre specialized algorithm.
    int n = 10000;

    timeLU(n);
}

int main() {
    //oppgaveA();

    //oppgaveB();

    //oppgaveC();

    //oppgaveD();

    //oppgaveE();

    return 0;

}
