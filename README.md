This version of TeaLeaf requires HYPRE. We use version 2.9.0b.

## HYPRE

- Download `hypre-2.9.0b` from computation.llnl.gov/casc/hypre/software.html
  - Babel interfaces are _not_ required.
- Extract the files.
- Inside the HYPRE `src` directory, use configure to set up the install location.
  - If you are only building HYPRE for TeaLeaf, consider installing it in the `libs/hypre-2.9.0b` folder in the TeaLeaf_HYPRE directory
  - Other options can also be set using the `configure` command. Use `./configure --help` to see them all.
  - For example: `CC=mpiicc CXX=mpiicpc F77=mpiifort ./configure --prefix=$HOME/TeaLeaf_HYPRE/libs/hypre-2.9.0b`.
- We recommend that you use the same compilers for building both HYPRE and TeaLeaf.
- Once HYPRE is configured, use `make` and then `make install` to install HYPRE.

## TeaLeaf

## Compling

- If you didn't install HYPRE to `libs/hypre-2.9.0b' edit the TeaLeaf `Makefile` and specify the directory that you have installed HYPRE in.
- In many case just typing `make` in the required software version will work. 

If the MPI compilers have different names then the build process needs to 
notified of this by defining two environment variables, `MPI_COMPILER` and 
`C_MPI_COMPILER`. 

For example on some Intel systems:

`make MPI_COMPILER=mpiifort C_MPI_COMPILER=mpiicc`

Or on Cray systems:

`make MPI_COMPILER=ftn C_MPI_COMPILER=cc`

### OpenMP Build

All compilers use different arguments to invoke OpenMP compilation. A simple 
call to make will invoke the compiler with -O3. This does not usually include 
OpenMP by default. To build for OpenMP for a specific compiler a further 
variable must be defined, `COMPILER` that will then select the correct option 
for OpenMP compilation. 

For example with the Intel compiler:

`make COMPILER=INTEL`

Which then append the -openmp to the build flags.

Other supported compiler that will be recognise are:-

* CRAY
* SUN
* GNU
* IBM
* PATHSCALE
* PGI

The default flags for each of these is show below:-

* INTEL: -O3 -ipo
* SUN: -fast
* GNU: -ipo
* XL: -O5
* PATHSCLE: -O3
* PGI: -O3 -Minline
* CRAY: -em  _Note: that by default the Cray compiler with pick the optimum 
options for performance._

### Other Flags

The default compilation with the COMPILER flag set chooses the optimal 
performing set of flags for the specified compiler, but with no hardware 
specific options or IEEE compatability.

To produce a version that has IEEE compatiblity a further flag has to be set on 
the compiler line.

`make COMPILER=INTEL IEEE=1`

This flag has no effect if the compiler flag is not set because IEEE options 
are always compiler specific.

For each compiler the flags associated with IEEE are shown below:-

* INTEL: -fp-model strict –fp-model source –prec-div –prec-sqrt
* CRAY: -hpflex_mp=intolerant
* SUN: -fsimple=0 –fns=no
* GNU: -ffloat-store
* PGI: -Kieee
* PATHSCALE: -mieee-fp
* XL: -qstrict –qfloat=nomaf

Note that the MPI communications have been written to ensure bitwise identical 
answers independent of core count. However under some compilers this is not 
true unless the IEEE flags is set to be true. This is certainly true of the 
Intel and Cray compiler. Even with the IEEE options set, this is not guarantee 
that different compilers or platforms will produce the same answers. Indeed a 
Fortran run can give different answers from a C run with the same compiler, 
same options and same hardware.

Extra options can be added without modifying the makefile by adding two further 
flags, `OPTIONS` and `C_OPTIONS`, one for the Fortran and one for the C options.

`make COMPILER=INTEL OPTIONS=-xavx C_OPTIONS=-xavx`

Finally, a `DEBUG` flag can be set to use debug options for a specific compiler.

`make COMPILER=PGI DEBUG=1`

These flags are also compiler specific, and so will depend on the `COMPILER` 
environment variable.

So on a system without the standard MPI wrappers, for a build that requires 
OpenMP, IEEE and AVX this would look like so:-

```
make COMPILER=INTEL MPI_COMPILER=mpiifort C_MPI_COMPILER=mpiicc IEEE=1 \
OPTIONS="-xavx" C_OPTIONS="-xavx"
```
