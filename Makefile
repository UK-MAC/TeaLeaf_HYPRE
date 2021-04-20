#Crown Copyright 2014 AWE.
#
# This file is part of TeaLeaf.
#
# TeaLeaf is free software: you can redistribute it and/or modify it under 
# the terms of the GNU General Public License as published by the 
# Free Software Foundation, either version 3 of the License, or (at your option) 
# any later version.
#
# TeaLeaf is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
# details.
#
# You should have received a copy of the GNU General Public License along with 
# TeaLeaf. If not, see http://www.gnu.org/licenses/.

#  @brief Makefile for TeaLeaf
#  @author David Beckingsale, Wayne Gaudin
#  @details Agnostic, platform independent makefile for the TeaLeaf benchmark code.

# Agnostic, platform independent makefile for the TeaLeaf benchmark code.
# It is not meant to be clever in anyway, just a simple build out of the box script.
# Just make sure mpif90 is in your path. It uses mpif90 even for all builds because this abstracts the base
#  name of the compiler. If you are on a system that doesn't use mpif90, just replace mpif90 with the compiler name
#  of choice. The only mpi dependencies in this non-MPI version are mpi_wtime in timer.f90.

# There is no single way of turning OpenMP compilation on with all compilers.
# The known compilers have been added as a variable. By default the make
#  will use no options, which will work on Cray for example, but not on other
#  compilers.
# To select a OpenMP compiler option, do this in the shell before typing make:-
#
#  export COMPILER=INTEL       # to select the Intel flags
#  export COMPILER=SUN         # to select the Sun flags
#  export COMPILER=GNU         # to select the Gnu flags
#  export COMPILER=CRAY        # to select the Cray flags
#  export COMPILER=PGI         # to select the PGI flags
#  export COMPILER=PATHSCALE   # to select the Pathscale flags
#  export COMPILER=XL          # to select the IBM Xlf flags

# or this works as well:-
#
# make COMPILER=INTEL
# make COMPILER=SUN
# make COMPILER=GNU
# make COMPILER=CRAY
# make COMPILER=PGI
# make COMPILER=PATHSCALE
# make COMPILER=XL
#

# Don't forget to set the number of threads you want to use, like so
# export OMP_NUM_THREADS=4

# usage: make                     # Will make the binary
#        make clean               # Will clean up the directory
#        make DEBUG=1             # Will select debug options. If a compiler is selected, it will use compiler specific debug options
#        make IEEE=1              # Will select debug options as long as a compiler is selected as well
# e.g. make COMPILER=INTEL MPI_COMPILER=mpiifort C_MPI_COMPILER=mpiicc DEBUG=1 IEEE=1 # will compile with the intel compiler with intel debug and ieee flags included



HYPRE_DIR=${HYPRE_PATH}

ifndef COMPILER
  MESSAGE=select a compiler to compile in OpenMP, e.g. make COMPILER=INTEL
endif

OMP_INTEL     = -qopenmp
OMP_SUN       = -xopenmp=parallel -vpara
OMP_GNU       = -fopenmp
OMP_CRAY      = -e Z
OMP_PGI       = -mp=nonuma
OMP_PATHSCALE = -mp
OMP_XL        = -qsmp=omp -qthreaded
OMP=$(OMP_$(COMPILER))

FLAGS_INTEL     = -O3 -no-prec-div -fpp
FLAGS_SUN       = -fast -xipo=2 -Xlistv4
FLAGS_GNU       = -O3 -mcpu=native -funroll-loops
FLAGS_CRAY      = -em -ra -h acc_model=fast_addr:no_deep_copy:auto_async_all
FLAGS_PGI       = -fastsse -gopt -Mipa=fast -Mlist
FLAGS_PATHSCALE = -O3
FLAGS_XL       = -O5 -qarch=pwr8 -qtune=pwr8 -qextname -qipa=partition=large -g -qfullpath -Q -qsigtrap -qextname=flush:timer_c:unpack_top_bottom_buffers_c:pack_top_bottom_buffers_c:unpack_left_right_buffers_c:pack_left_right_buffers_c:field_summary_kernel_c:update_halo_kernel_c:generate_chunk_kernel_c:initialise_chunk_kernel_c:calc_dt_kernel_c -qlistopt -qattr=full -qlist -qreport -qxref=full -qsource -qsuppress=1506-224:1500-036
FLAGS_          = -O3
CFLAGS_INTEL     = -O3 -no-prec-div -restrict #-fno-alias
CFLAGS_SUN       = -fast -xipo=2
CFLAGS_GNU       = -O3 -mcpu=native -funroll-loops
CFLAGS_CRAY      = -em -h list=a
CFLAGS_PGI       = -fastsse -gopt -Mipa=fast -Mlist
CFLAGS_PATHSCALE = -O3
CFLAGS_XL       = -O5 -qarch=pwr8 -qtune=pwr8 -qipa=partition=large -g -qfullpath -Q -qlist -qreport -qsource
CFLAGS_          = -O3

ifdef DEBUG
  FLAGS_INTEL     = -O0 -g -debug all -check all -traceback -check noarg_temp_created
  FLAGS_SUN       = -g -xopenmp=noopt -stackvar -u -fpover=yes -C -ftrap=common
  FLAGS_GNU       = -O0 -g -O -Wall -Wextra -fbounds-check
  FLAGS_CRAY      = -O0 -g -em -eD
  FLAGS_PGI       = -O0 -g -C -Mchkstk -Ktrap=fp -Mchkfpstk -Mchkptr
  FLAGS_PATHSCALE = -O0 -g
  FLAGS_XL       = -O0 -g -qfullpath -qcheck -qflttrap=ov:zero:invalid:en -qsource -qinitauto=FF -qmaxmem=-1 -qinit=f90ptr -qsigtrap -qextname=flush:timer_c:unpack_top_bottom_buffers_c:pack_top_bottom_buffers_c:unpack_left_right_buffers_c:pack_left_right_buffers_c:field_summary_kernel_c:update_halo_kernel_c:generate_chunk_kernel_c:initialise_chunk_kernel_c:calc_dt_kernel_c
  FLAGS_          = -O0 -g
  CFLAGS_INTEL    = -O0 -g -debug all -traceback
  CFLAGS_SUN      = -g -O0 -xopenmp=noopt -stackvar -u -fpover=yes -C -ftrap=common
  CFLAGS_GNU       = -O0 -g -O -Wall -Wextra -fbounds-check
  CFLAGS_CRAY     = -O0 -g -em -eD
  CFLAGS_PGI      = -O0 -g -C -Mchkstk -Ktrap=fp -Mchkfpstk
  CFLAGS_PATHSCALE= -O0 -g
  CFLAGS_XL      = -O0 -g -qfullpath -qcheck -qflttrap=ov:zero:invalid:en -qsource -qinitauto=FF -qmaxmem=-1 -qsrcmsg
endif

ifdef IEEE
  I3E_INTEL     = -fp-model strict -fp-model source -prec-div -prec-sqrt
  I3E_SUN       = -fsimple=0 -fns=no
  I3E_GNU       = -ffloat-store
  I3E_CRAY      = -hflex_mp=intolerant
  I3E_PGI       = -Kieee
  I3E_PATHSCALE = -mieee-fp
  I3E_XL       = -qfloat=nomaf
  I3E=$(I3E_$(COMPILER))
endif

FLAGS=${FLAGS_$(COMPILER)} ${OMP} ${I3E} ${OPTIONS} ${PETSC_INC} $(REQ_LIB)
CFLAGS=${CFLAGS_$(COMPILER)} ${OMP} ${I3E} ${C_OPTIONS} ${PETSC_INC} -I$(HYPRE_DIR)/include -c

# flags for nvcc
# # set NV_ARCH to select the correct one
NV_ARCH=VOLTA
CODE_GEN_FERMI=-gencode arch=compute_20,code=sm_21
CODE_GEN_KEPLER=-gencode arch=compute_35,code=sm_35
CODE_GEN_KEPLER_CONSUMER=-gencode arch=compute_30,code=sm_30
CODE_GEN_MAXWELL=-gencode arch=compute_50,code=sm_50
CODE_GEN_PASCAL=-gencode arch=compute_60,code=sm_60
CODE_GEN_VOLTA=-gencode arch=compute_70,code=sm_70
LDLIBS+= -lcudart  -lcusparse -lcurand -lcudadevrt -lm -lstdc++ \
		 -L/opt/ibm/xlC/16.1.1/lib -L/opt/ibm/lib -libmc++  

MPI_COMPILER=mpif90
C_MPI_COMPILER=mpicc
CXX_MPI_COMPILER=mpic++
NVCC_COMPILER=nvcc

# requires CUDA_HOME to be set - not the same on all machines
NV_FLAGS=$(CODE_GEN_$(NV_ARCH)) -g -lineinfo -expt-extended-lambda -dc -std=c++11 --x cu -Xcompiler "-O2" -I$(CUDA_HOME)/include -I$(HYPRE_DIR)/include
NV_FLAGS+=-DNO_ERR_CHK
HYPRE_FLAGS=-DHYPRE_USING_CUDA
libdir.x86_64 = lib64
libdir.i686   = lib
libdir.ppc64le=lib64
MACHINE := $(shell uname -m)
libdir = $(libdir.$(MACHINE))
LDFLAGS+=-L$(CUDA_HOME)/$(libdir)

tea_leaf: HypreStem.o link.o timer_c.o  *.f90 Makefile
	$(MPI_COMPILER) $(FLAGS)	\
	data.f90			\
	definitions.f90			\
	pack_kernel.f90			\
	tea.f90				\
	report.f90			\
	timer.f90			\
	parse.f90			\
	read_input.f90			\
	initialise_chunk_kernel.f90	\
	initialise_chunk.f90		\
	build_field.f90			\
	update_halo_kernel.f90		\
	update_halo.f90			\
	start.f90			\
	generate_chunk_kernel.f90	\
	generate_chunk.f90		\
	initialise.f90			\
	field_summary_kernel.f90	\
	field_summary.f90		\
	calc_dt_kernel.f90		\
	calc_dt.f90			\
	timestep.f90			\
	set_field_kernel.f90		\
	set_field.f90			\
	tea_leaf_jacobi.f90             \
	tea_leaf_cg.f90			\
	tea_leaf_cheby.f90              \
	tea_leaf_ppcg.f90               \
	tea_solve.f90                   \
	visit.f90			\
	tea_leaf.f90			\
	diffuse.f90                     \
	timer_c.o                       \
	HypreStem.o                     \
	link.o							\
	$(HYPRE_DIR)/lib/libHYPRE.a     \
	$(LDFLAGS)			\
	$(LDLIBS)			\
	-o tea_leaf; echo $(MESSAGE)

timer_c.o: timer_c.c
	$(C_MPI_COMPILER) $(CFLAGS) timer_c.c

HypreStem.o: HypreStem.C HypreStem.h
	nvcc -O2  -ccbin=$(CXX_MPI_COMPILER) $(NV_FLAGS) $(HYPRE_FLAGS) -c $< -o $@

link.o: 
	nvcc -O2 -ccbin=$(CXX_MPI_COMPILER) $(CODE_GEN_$(NV_ARCH)) $(HYPRE_FLAGS) -dlink -o link.o HypreStem.o $(HYPRE_DIR)/lib/libHYPRE.a  $(LDFLAGS) $(LDLIBS) 

clean:
	rm -f *.o *.mod *genmod* *.lst *.cub *.ptx tea_leaf
