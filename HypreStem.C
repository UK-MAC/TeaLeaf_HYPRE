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
            int* print_info,
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
        int* print_info,
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
    int info = *print_info;
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
            info,
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
HYPRE_Real* HypreStem::coefficients;
HYPRE_Real* HypreStem::values;
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

    HYPRE_Init();
    HYPRE_StructGridCreate(MPI_COMM_WORLD, 2, &grid);

    int ilower[2];
    int iupper[2];

    ilower[0] = left;
    ilower[1] = bottom;

    iupper[0] = right;
    iupper[1] = top;

    HYPRE_StructGridSetExtents(grid, ilower, iupper);

    HYPRE_StructGridAssemble(grid);

    HYPRE_StructGridSetDataLocation(grid, HYPRE_MEMORY_DEVICE);

    HYPRE_StructStencilCreate(2, 5, &stencil);

    int offsets[5][2] = {{0,0}, {-1,0}, {1,0}, {0,-1}, {0,1}};

    for(int entry = 0; entry < 5; entry++) {
        HYPRE_StructStencilSetElement(stencil, entry, offsets[entry]);
    }

    HYPRE_StructMatrixCreate(MPI_COMM_WORLD, grid, stencil, &A);

    HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &b);
    HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &x);

    if(d_solver_type == SOLVER_TYPE_JACOBI) {
        //std::cout << "Using JACOBI Solver" << std::endl;
        HYPRE_StructJacobiCreate(MPI_COMM_WORLD, &solver);
        HYPRE_StructJacobiSetTol(solver, eps);
        HYPRE_StructJacobiSetMaxIter(solver, max_iters);
    } 
    else if ( d_solver_type == SOLVER_TYPE_PFMG ){
        HYPRE_StructPCGCreate(MPI_COMM_WORLD, &solver);
        /* Create the Struct PFMG solver for use as a preconditioner */
        HYPRE_StructPFMGCreate(MPI_COMM_WORLD, &preconditioner);
      	
        /* Set PFMG parameters */
        HYPRE_StructPFMGSetMaxIter(preconditioner, 1);
        HYPRE_StructPFMGSetTol(preconditioner, 0.0);
        HYPRE_StructPFMGSetZeroGuess(preconditioner);
        HYPRE_StructPFMGSetNumPreRelax(preconditioner, 1);
        HYPRE_StructPFMGSetNumPostRelax(preconditioner, 1);
      
        /* non-Galerkin coarse grid  */
        HYPRE_StructPFMGSetRAPType(preconditioner, 1);
        /* TODO: Default Jacobi GPU performance better than Red/Black? 
         * 0 : Jacobi
         * 1 : Weighted Jacobi (default)
         * 2 : Red/Black Gauss-Seidel (symmetric: RB pre-relaxation, BR post-relaxation)
         * 3 : Red/Black Gauss-Seidel (nonsymmetric: RB pre- and post-relaxation)
         */
         HYPRE_StructPFMGSetRelaxType(preconditioner, 1);

        /* skip relaxation on some levels (more efficient for this problem) */
        HYPRE_StructPFMGSetSkipRelax(preconditioner, 1);

        HYPRE_StructPCGSetTol(solver, eps);
        HYPRE_StructPCGSetMaxIter(solver, max_iters);

        HYPRE_StructPCGSetPrecond(solver,
            HYPRE_StructPFMGSolve,
            HYPRE_StructPFMGSetup,
            preconditioner); 
    }
    else if  ( d_solver_type == SOLVER_TYPE_SMG ){
        HYPRE_StructPCGCreate(MPI_COMM_WORLD, &solver);

        /* Create the SMG as a preconditioner */
        HYPRE_StructSMGCreate(MPI_COMM_WORLD, &preconditioner);
        HYPRE_StructSMGSetMemoryUse(preconditioner, 0);
        HYPRE_StructSMGSetMaxIter(preconditioner, 1);
        HYPRE_StructSMGSetTol(preconditioner, 0.0);
        HYPRE_StructSMGSetZeroGuess(preconditioner);

        HYPRE_StructSMGSetNumPreRelax(preconditioner, 1);
        HYPRE_StructSMGSetNumPostRelax(preconditioner, 1);

        HYPRE_StructPCGSetTol(solver, eps);
        HYPRE_StructPCGSetMaxIter(solver, max_iters);
        HYPRE_StructPCGSetPrecond(solver,
                      HYPRE_StructSMGSolve,
                      HYPRE_StructSMGSetup,
                      preconditioner);

    }
    else {
        /* use diagonal scaling as preconditioner */
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

    coefficients = hypre_CTAlloc(HYPRE_Real, nvalues*5, HYPRE_MEMORY_DEVICE);
    values = hypre_CTAlloc(HYPRE_Real, nvalues, HYPRE_MEMORY_DEVICE);
}

void HypreStem::finalise()
{
    HYPRE_StructMatrixDestroy(A);
    HYPRE_StructVectorDestroy(b);
    HYPRE_StructVectorDestroy(x);

    if(d_solver_type == SOLVER_TYPE_JACOBI) {
        HYPRE_StructJacobiDestroy(solver);
    }
    else if  ( d_solver_type == SOLVER_TYPE_PFMG ){
        HYPRE_StructPCGDestroy(solver);
        HYPRE_StructPFMGDestroy(preconditioner);
    }
    else if  ( d_solver_type == SOLVER_TYPE_SMG ){
        HYPRE_StructPCGDestroy(solver);
        HYPRE_StructSMGDestroy(preconditioner);
    }
    else {
        HYPRE_StructPCGDestroy(solver);
    }
    hypre_TFree(values, HYPRE_MEMORY_DEVICE);
    hypre_TFree(coefficients, HYPRE_MEMORY_DEVICE);
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
        int  info,
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

    int nx = (x_max - x_min + 1) + 4;

    int nentries = 5;
    int stencil_indices[5] = {0,1,2,3,4};

    int n = 0;
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

            //coefficients[n] = (1.0+(2.0*(0.5*(c2+c3))*rx)+(2.0*(0.5*(c4+c5))*ry));
            coefficients[n+1] = (-1.0*rx)*c2;
            coefficients[n+2] = (-1.0*rx)*c3;
            coefficients[n+3] = (-1.0*ry)*c4;
            coefficients[n+4] = (-1.0*ry)*c5;

            if(j == global_xmin) {
                coefficients[n+1] = 0;
            }

            if(j == global_xmax) {
                coefficients[n+2] = 0;
            } 

            if (k == global_ymin) {
                coefficients[n+3] = 0;
            }

            if (k == global_ymax) {
                coefficients[n+4] = 0;
            }

            coefficients[n] = 1.0-coefficients[n+1]-coefficients[n+2]-coefficients[n+3]-coefficients[n+4];
            n += nentries;
        }
    }
    //TODO: coefficients data input bottleneck
    HYPRE_StructMatrixSetBoxValues(A, ilower, iupper, nentries, stencil_indices, coefficients);

    ilower[0] = left;
    ilower[1] = bottom;
    iupper[0] = right;
    iupper[1] = top;

    HYPRE_StructMatrixAssemble(A);
    if(info == 1) {
       HYPRE_StructMatrixPrint("A", A, 0);
    }
    HYPRE_StructVectorInitialize(b);
    HYPRE_StructVectorInitialize(x);

    int xmn = left-2;
    int ymn = bottom-2;
    nx = (x_max - x_min + 1) + 4;

    n = 0;

    for (int j = bottom; j <= top; j++) {
        for (int i = left; i <= right; i++) {
            values[n] = u0[ARRAY2D(i,j,xmn,ymn,nx)];
            n++;
        }
    }
    //TODO: Vector data
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
    if(info == 1) {
       HYPRE_StructVectorPrint("b", b, 0);
    }
    HYPRE_StructVectorAssemble(x);

    if(SOLVER_TYPE_JACOBI == d_solver_type) {
        HYPRE_StructJacobiSetup(solver, A, b, x);
        HYPRE_StructJacobiSolve(solver, A, b, x);
    } else {
        HYPRE_StructPCGSetup(solver, A, b, x);
        HYPRE_StructPCGSolve(solver, A, b, x);
    }

    // TODO: synchronize before using the vector??
    //
    cudaDeviceSynchronize();
    if(info == 1) {
      HYPRE_StructVectorPrint("x", x, 0);
    }

    int iters = 0;
    double norm = 0.0;
    
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    
    if (SOLVER_TYPE_JACOBI == d_solver_type) {
        HYPRE_StructJacobiGetNumIterations(solver, &iters);
        HYPRE_StructJacobiGetFinalRelativeResidualNorm(solver, &norm);
        
        // If on master then print its and norm
        if (myid == 0){
            // print its
            printf("Iteration count       %d\n", iters);
            // print norm
            printf("Residual norm         %.17g\n", norm);
        } 
        
    } else {
        HYPRE_StructPCGGetNumIterations(solver, &iters);
        HYPRE_StructPCGGetFinalRelativeResidualNorm(solver, &norm);
        
        // If on master then print its and norm
        if (myid == 0){
            // print its
            printf("Iteration count       %d\n", iters);
            // print norm
            printf("Residual norm         %.17g\n", norm);
        } 
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
