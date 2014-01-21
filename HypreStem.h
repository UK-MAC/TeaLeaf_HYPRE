#ifndef HYPRE_LEAF_H_
#define HYPRE_LEAF_H_

#include "HYPRE_struct_ls.h"

#define SOLVER_TYPE_JACOBI 1

class HypreStem {
    public:
        HypreStem();
        virtual ~HypreStem();

        static void init(
                int left,
                int right,
                int bottom,
                int top,
                double eps,
                int max_iters,
                int solver_type);

        static void finalise();

        static void solve(
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
                int* neighbours);

        static HYPRE_StructGrid grid;
        static HYPRE_StructStencil stencil;
        static HYPRE_StructMatrix A;
        static HYPRE_StructVector b;
        static HYPRE_StructVector x;
        static HYPRE_StructSolver solver;
        static HYPRE_StructSolver preconditioner;

        static double* coefficients;
        static double* values;
        static int d_solver_type;
    private:
};
#endif
