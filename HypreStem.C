#include "HypreStem.h"

#include <iostream>

#define ARRAY2D(i,j,imin,jmin,ni) (i-(imin))+(((j)-(jmin))*(ni))
#define EXTERNAL_FACE -1

extern "C" {
    void setup_hypre_(
            int* lft,
            int* rght,
            int* bttm,
            int* tp,
            double* eps,
            int* mx_itrs,
            int* slvr_tp);

    void teardown_hypre_();

    void hypre_solve_(
            int* lft,
            int* rght,
            int* bttom,
            int* tp,
            int* xmin,
            int* xmax,
            int* ymin,
            int* ymax,
            int* globalxmin,
            int* globalxmax,
            int* globalymin,
            int* globalymax,
            double* rx,
            double* ry,
            double* Kx,
            double* Ky,
            double* u0,
            int* neighbours);
}

void setup_hypre_(
        int* lft,
        int* rght,
        int* bttm,
        int* tp,
        double* eps,
        int* mx_itrs,
        int* slvr_tp)
{
    int left = *lft;
    int right = *rght;
    int bottom = *bttm;
    int top = *tp;

    double epsilon = *eps;
    int max_iters = *mx_itrs;
    int solver_type = *slvr_tp;

    HypreStem::init(left,right,bottom,top,epsilon,max_iters,solver_type);
}

void teardown_hypre_()
{
    HypreStem::finalise();
}

void hypre_solve_(
        int* lft,
        int* rght,
        int* bttom,
        int* tp,
        int* xmin,
        int* xmax,
        int* ymin,
        int* ymax,
        int* globalxmin,
        int* globalxmax,
        int* globalymin,
        int* globalymax,
        double* rxp,
        double* ryp,
        double* Kx,
        double* Ky,
        double* u0,
        int* neighbours)
{
    int left = *lft;
    int right = *rght;
    int bottom = *bttom;
    int top = *tp;
    int x_min = *xmin;
    int x_max = *xmax;
    int y_min = *ymin;
    int y_max = *ymax;
    int global_xmin = *globalxmin;
    int global_xmax = *globalxmax;
    int global_ymin = *globalymin;
    int global_ymax = *globalymax;
    double rx = *rxp;
    double ry = *ryp;

    HypreStem::solve(
            left,
            right,
            bottom,
            top,
            x_min,
            x_max,
            y_min,
            y_max,
            global_xmin,
            global_xmax,
            global_ymin,
            global_ymax,
            rx,
            ry,
            Kx,
            Ky,
            u0,
            neighbours);
}

HYPRE_StructGrid HypreStem::grid;
HYPRE_StructStencil HypreStem::stencil;
HYPRE_StructMatrix HypreStem::A;
HYPRE_StructVector HypreStem::b;
HYPRE_StructVector HypreStem::x;
HYPRE_StructSolver HypreStem::solver;
HYPRE_StructSolver HypreStem::preconditioner;
double* HypreStem::coefficients;
double* HypreStem::values;
int HypreStem::d_solver_type;

void HypreStem::init(
        int left,
        int right,
        int bottom,
        int top,
        double eps,
        int max_iters,
        int solver_type)
{
    d_solver_type = solver_type;

    HYPRE_StructGridCreate(MPI_COMM_WORLD, 2, &grid);

    int ilower[2];
    int iupper[2];

    ilower[0] = left;
    ilower[1] = bottom;

    iupper[0] = right;
    iupper[1] = top;

    HYPRE_StructGridSetExtents(grid, ilower, iupper);

    HYPRE_StructGridAssemble(grid);

    HYPRE_StructStencilCreate(2, 5, &stencil);

    int offsets[5][2] = {{0,0}, {-1,0}, {1,0}, {0,-1}, {0,1}};

    for(int entry = 0; entry < 5; entry++) {
        HYPRE_StructStencilSetElement(stencil, entry, offsets[entry]);
    }

    HYPRE_StructMatrixCreate(MPI_COMM_WORLD, grid, stencil, &A);

    HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &b);
    HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &x);

    if(d_solver_type == SOLVER_TYPE_JACOBI) {
        std::cout << "Using JACOBI Solver" << std::endl;
        HYPRE_StructJacobiCreate(MPI_COMM_WORLD, &solver);
        HYPRE_StructJacobiSetTol(solver, eps);
        HYPRE_StructJacobiSetMaxIter(solver, max_iters);
    } else {
        HYPRE_StructPCGCreate(MPI_COMM_WORLD, &solver);
        HYPRE_StructPCGSetTol(solver, eps);
        HYPRE_StructPCGSetMaxIter(solver, max_iters);
        HYPRE_StructPCGSetPrecond(solver,
            HYPRE_StructDiagScale,
            HYPRE_StructDiagScaleSetup,
            preconditioner);
    }



    int nx = right - left + 1;
    int ny = top - bottom + 1;
    int nvalues = nx*ny;

    coefficients = new double[nvalues*5];
    values = new double[nvalues];
}

void HypreStem::finalise()
{
    HYPRE_StructMatrixDestroy(A);
    HYPRE_StructVectorDestroy(b);
    HYPRE_StructVectorDestroy(x);

    if(d_solver_type == SOLVER_TYPE_JACOBI) {
        HYPRE_StructJacobiDestroy(solver);
    } else {
        HYPRE_StructPCGDestroy(solver);
    }

    delete coefficients;
}

void HypreStem::solve(
        int left,
        int right,
        int bottom,
        int top,
        int x_min,
        int x_max,
        int y_min,
        int y_max,
        int global_xmin,
        int global_xmax,
        int global_ymin,
        int global_ymax,
        double rx,
        double ry,
        double* Kx,
        double* Ky,
        double* u0,
        int* neighbours)
{
    HYPRE_StructMatrixInitialize(A);

    int ilower[2], iupper[2];

    ilower[0] = left;
    ilower[1] = bottom;
    iupper[0] = right;
    iupper[1] = top;

    int nx = (x_max - x_min + 1) + 5;
    int ny = (y_max - y_min + 1) + 5;

    int nentries = 5;
    int stencil_indices[5] = {0,1,2,3,4};

    int n = 0;
    //for(int i = 0; i < nvalues; i += nentries) {
    for(int k = bottom; k <= top; k++) {
        for(int j = left; j <= right; j++) {

            /*
             * Stencil indices:
             *
             *   | 5 |
             * 2 | 1 | 3
             *   | 4 |
             */
            double c2 = Kx[ARRAY2D(j,k,left-2,bottom-2,nx)];
            double c3 = Kx[ARRAY2D(j+1,k,left-2,bottom-2,nx)];
            double c4 = Ky[ARRAY2D(j,k,left-2,bottom-2,nx)];
            double c5 = Ky[ARRAY2D(j,k+1,left-2,bottom-2,nx)];

            coefficients[n] = (1.0+(2.0*(0.5*(c2+c3))*rx)+(2.0*(0.5*(c4+c5))*ry));
            coefficients[n+1] = (-1.0*rx)*c2;
            coefficients[n+2] = (-1.0*rx)*c3;
            coefficients[n+3] = (-1.0*ry)*c4;
            coefficients[n+4] = (-1.0*ry)*c5;

            if(j == global_xmin) {
                coefficients[n+2] = (-2.0*rx)*c3;
            } 
            if(j == global_xmax) {
                coefficients[n+1] = (-2.0*rx)*c2;
            }

            if (k == global_ymin) {
                coefficients[n+4] = (-2.0*ry)*c5;
            } 
            if (k == global_ymax) {
                coefficients[n+3] = (-2.0*ry)*c4;
            }


            n += nentries;
        }
    }
    //}

    HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, nentries, stencil_indices, coefficients);

    for (int i = 0; i < 4; i++) {
        if(neighbours[i] == EXTERNAL_FACE) {
            switch(i) {
                case 0: 
                    {
                        ilower[0] = left;
                        ilower[1] = bottom;
                        iupper[0] = left;
                        iupper[1] = top;

                        int boundary_stencil_indices[1] = {1};

                        double* boundary_coefficients = new double[ny];

                        for(int j = 0; j < ny; j++) {
                            boundary_coefficients[j] = 0.0;
                        }

                        HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 1, boundary_stencil_indices, boundary_coefficients);

                        delete [] boundary_coefficients;
                    }
                    break;
                case 1: 
                    {
                        ilower[0] = right;
                        ilower[1] = bottom;
                        iupper[0] = right;
                        iupper[1] = top;

                        int boundary_stencil_indices[1] = {2};

                        double* boundary_coefficients = new double[ny];

                        for(int j = 0; j < ny; j++) {
                            boundary_coefficients[j] = 0.0;
                        }

                        HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 1, boundary_stencil_indices, boundary_coefficients);

                        delete [] boundary_coefficients;
                    }
                    break;
                case 2: 
                    {
                        ilower[0] = left;
                        ilower[1] = bottom;
                        iupper[0] = right;
                        iupper[1] = bottom;

                        int boundary_stencil_indices[1] = {3};

                        double* boundary_coefficients = new double[nx];

                        for(int j = 0; j < nx; j++) {
                            boundary_coefficients[j] = 0.0;
                        }

                        HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 1, boundary_stencil_indices, boundary_coefficients);

                        delete [] boundary_coefficients;
                    }
                    break;
                case 3:
                    {
                        ilower[0] = left;
                        ilower[1] = top;
                        iupper[0] = right;
                        iupper[1] = top;

                        int boundary_stencil_indices[1] = {4};

                        double* boundary_coefficients = new double[nx];

                        for(int j = 0; j < nx; j++) {
                            boundary_coefficients[j] = 0.0;
                        }

                        HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, 1, boundary_stencil_indices, boundary_coefficients);

                        delete [] boundary_coefficients;
                    }
                    break;
            }
        }
    }

    ilower[0] = left;
    ilower[1] = bottom;
    iupper[0] = right;
    iupper[1] = top;

    HYPRE_StructMatrixAssemble(A);
    HYPRE_StructMatrixPrint("A", A, 0);

    HYPRE_StructVectorInitialize(b);
    HYPRE_StructVectorInitialize(x);


    int xmn = left-2;
    int ymn = bottom-2;
    nx = (x_max - x_min + 1) + 4;

    n = 0;

    for (int j = bottom; j <= top; j++) {
        for (int i = left; i <= right; i++) {
            double c2 = Kx[ARRAY2D(i,j,xmn,ymn,nx+1)];
            double c3 = Kx[ARRAY2D(i+1,j,xmn,ymn,nx+1)];
            double c4 = Ky[ARRAY2D(i,j,xmn,ymn,nx+1)];
            double c5 = Ky[ARRAY2D(i,j+1,xmn,ymn,nx+1)];

            values[n] = u0[ARRAY2D(i,j,xmn,ymn,nx)];

//            if(i == global_xmin) {
//                values[n] += rx*c2*u0[ARRAY2D(i-1,j,xmn,ymn,nx)];
//            } 
//            if(i == global_xmax) {
//                values[n] += rx*c3*u0[ARRAY2D(i+1,j,xmn,ymn,nx)];
//            }
//
//            if (j == global_ymin) {
//                values[n] += ry*c4*u0[ARRAY2D(i,j-1,xmn,ymn,nx)];
//            } 
//            if (j == global_ymax) {
//                values[n] += ry*c5*u0[ARRAY2D(i,j+1,xmn,ymn,nx)];
//            }

            n++;
        }
    }

    HYPRE_StructVectorSetBoxValues(b, ilower, iupper, values);

    n = 0;

    for (int j = bottom; j <= top; j++) {
        for (int i = left; i <= right; i++) {
            values[n] = u0[ARRAY2D(i,j,xmn,ymn,nx)];
            n++;
        }
    }

    HYPRE_StructVectorSetBoxValues(x, ilower, iupper, values);

    HYPRE_StructVectorAssemble(b);
    HYPRE_StructVectorPrint("b", b, 0);
    HYPRE_StructVectorAssemble(x);

    if(SOLVER_TYPE_JACOBI == d_solver_type) {
        HYPRE_StructJacobiSetup(solver, A, b, x);
        HYPRE_StructJacobiSolve(solver, A, b, x);
    } else {
        HYPRE_StructPCGSetup(solver, A, b, x);
        HYPRE_StructPCGSolve(solver, A, b, x);
    }

    HYPRE_StructVectorPrint("x", x, 0);

    int iters = 0;
    if (SOLVER_TYPE_JACOBI == d_solver_type) {
        HYPRE_StructJacobiGetNumIterations(solver, &iters);
    } else {
        HYPRE_StructPCGGetNumIterations(solver, &iters);
    }

    HYPRE_StructVectorGetBoxValues(x, ilower, iupper, values);

    n = 0;

    for (int j = bottom; j <= top; j++) {
        for (int i = left; i <= right; i++) {
            u0[ARRAY2D(i,j,xmn,ymn,nx)] = values[n];
            n++;
        }
    }
}
