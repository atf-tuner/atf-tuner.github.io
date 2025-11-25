---
layout: splash
---

![](/assets/images/atf_banner.png)

# Overview

The [Auto-Tuning Framework (ATF)](https://dl.acm.org/doi/abs/10.1145/3427093) is a general-purpose auto-tuning approach: given a program that is implemented as generic in performance-critical program parameters (a.k.a. *tuning parameters*), such as sizes of tiles and numbers of threads, ATF fully automatically determines a hardware- and data-optimized configuration of such parameters.

### Key Feature of ATF

A key feature of ATF is its support for [*Tuning Parameter Constraints*](/tp_constraints).
Parameter constraints allow auto-tuning programs whose tuning parameters have so-called *interdependencies* among them, e.g., the value of one tuning parameter has to evenly divide the value of another tuning parameter.

ATF's support for parameter constraints is important: modern parallel programs target novel parallel architectures, and such architectures typically have deep memory and core hierarchies thus requiring constraints on tuning parameters, e.g., the value of a tile size tuning parameter on an upper memory layer has to be a multiple of a tile size value on a lower memory layer.

For such parameters, ATF introduces novel concepts for [*Generating*](/tp_constraints/#generating-constrained-search-spaces) & [*Storing*](/tp_constraints/#storing-constrained-search-spaces) & [*Exploring*](/tp_constraints/#exploring-constrained-search-spaces) the search spaces of constrained tuning parameters, thereby contributing to a substantially more efficient overall auto-tuning process for such parameters, as confirmed in our [*Experiments*](/experiments).

### Generality of ATF

For wide applicability, ATF is designed as generic in:

  1. The target program's [Programming Language](/gen_pl), e.g., *C/C++*, *CUDA*, *OpenMP*, or *OpenCL*. ATF offers *pre-implemented cost functions* for conveniently auto-tuning C/C++ programs, as well as CUDA and OpenCL kernels which require host code for their execution which is automatically generated and executed by ATF's pre-implemented CUDA and OpenCL cost functions. ATF also offers a pre-implemented *generic* cost function that can be used for conveniently auto-tuning programs in any other programming language different from C/C++, CUDA, and OpenCL.

  2. The [Search Technique](/gen_st) to use. ATF offers different kinds of pre-implemented search techniques, such as *simulated annealing* and *AUC bandit* (inspired by [OpenTuner](https://opentuner.org)) which combines multiple techniques for exploration (such as differential evolution, Nelder-Mead, and Torczon hillclimbers). New techniques can be conveniently added to ATF, by implementing a straightforward interface.

  3. The [Tuning Objective](/gen_to), e.g., high *runtime performance* and/or low *energy consumption*. By default, ATF's pre-implemented cost functions auto-tune for high runtime performance. The user can choose arbitrary, self-defined tuning objectives.

  4. The [Abort Condition](/gen_ac) which specifies when to stop the auto-tuning process. ATF's pre-implemented abort conditions allow to stop the auto-tuning process dependent on the *tuning time* (e.g., after a user-defined time interval has been exceeded or a user-defined number of parameter configurations has been explored), but also dependent on the *tuning result* (e.g., when particular performance is already achieved) or dependent on *both time and result* (e.g., when within a particular time interval the auto-tuning run was not able to further improve the target program's performance). New conditions can be conveniently added to ATF by the user, by implementing a straightforward interface.




<br>
# Getting Started

ATF is open source on [GitHub](https://github.com/atf-tuner/), with both of its interface kinds, under a *commercially permissive MIT license*:

  1. [pyATF](https://github.com/atf-tuner/pyATF) -- ATF with its Python-based user interface
  2. [cppATF](https://github.com/atf-tuner/cppATF) -- ATF with its C++-based user interface

We encourage you to use ATF in other projects, open source, or commercial.

<br>
# Code Examples

ATF currently offers two kinds of user interfaces that can be freely chosen by the user:

  1. A [Python](https://www.python.org)-based user interface -- we call ATF with its Python-based interface [pyATF](https://github.com/atf-tuner/pyATF)
  2. A [C++](https://isocpp.org)-based user interface -- we call ATF with its C++-based interface [cppATF](https://github.com/atf-tuner/cppATF)

More interface kinds for ATF, in further mainstream programming languages, might follow.

## Using ATF for Constrained Search Spaces

We show how ATF is used for auto-tuning [CUDA SAXPY Routine](https://github.com/CNugteren/CLBlast/blob/master/src/kernels/level1/xaxpy.opencl) from the [CLBlast library](https://github.com/CNugteren/CLBlast).

The routine relies on two tuning parameters -- `WPT` and `LS`.
Both parameters have as their range of potential values `1,...,N` where `N` denotes the user-defined input size.
The two parameters are interdependent: parameter `LS` needs to divide `N/WPT` -- the input size `N` divided by the value of the `WPT` parameters -- for correctness of the CUDA routine.

<div class="tab-container">

{% tabs saxpy %}

{% tab saxpy pyATF %}

<hr>

The SAXPY CUDA routine is conveniently auto-tuned via [pyATF](https://github.com/atf-tuner/pyATF) using its pre-implemented CUDA cost function `cuda.CostFunction`, which automatically generates and executes the CUDA host code required for executing the routine.

~~~python
import numpy as np

from pyatf import TP, Interval, Tuner
from pyatf.cost_functions import cuda
from pyatf.search_techniques import AUCBandit
from pyatf.abort_conditions import Evaluations

# kernel code as string
saxpy_kernel_as_string = '''
extern "C" __global__ void saxpy( const int N, const float a, const float* x, float* y )
{
    for( int w = 0 ; w < WPT ; ++w )
    {
        const int index = w * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
        y[ index ] += a * x[ index ];
    }
}
'''

# input size
N = 1000

# Step 1: Generate the Search Space
WPT = TP('WPT', Interval( 1, N ), lambda WPT: N % WPT == 0           )
LS  = TP('LS',  Interval( 1, N ), lambda WPT, LS: (N / WPT) % LS == 0)

# Step 2: Implement a Cost Function
saxpy_kernel = cuda.Kernel( cuda.source(saxpy_kernel_as_string), 'saxpy' )  # kernel's code & name

cf_saxpy = cuda.CostFunction( saxpy_kernel ).device_id( 0 )                                       \
                                            .kernel_args( np.int32( N )                        ,
                                                          np.float32(np.random.random())       ,
                                                          np.random.rand(N).astype(np.float32) ,
                                                          np.random.rand(N).astype(np.float32) )  \
                                            .grid_dim( lambda WPT, LS: N/WPT/LS )                 \
                                            .block_dim( lambda LS: LS )

# Step 3: Explore the Search Space
tuning_result = Tuner().tuning_parameters( WPT, LS )       \
                       .search_technique( AUCBandit() )    \
                       .tune( cf_saxpy, Evaluations(50) )
~~~

{% endtab %}

{% tab saxpy cppATF %}

<hr>

The SAXPY CUDA routine is conveniently auto-tuned via [cppATF](https://github.com/atf-tuner/cppATF) using its pre-implemented CUDA cost function `atf::cuda::cost_function`, which automatically generates and executes the CUDA host code required for executing the routine.

~~~ c++
#define ENABLE_CUDA_COST_FUNCTION
#include <atf.hpp>

int main( int argc, char* argv[] )
{
    // kernel code as string
    auto saxpy_kernel_as_string = R"(
extern "C" __global__ void saxpy( const int N, const float a, const float* x, float* y )
{
    for( int w = 0 ; w < WPT ; ++w )
    {
        const int index = w * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
        y[ index ] += a * x[ index ];
    }
})";

    // input size
    int N = 1000;

    // Step 1: Generate the Search Space
    auto WPT = atf::tuning_parameter( "WPT"                        ,
                                      atf::interval<size_t>( 1,N ) ,
                                      atf::divides( N )            );

    auto LS  = atf::tuning_parameter( "LS"                         ,
                                      atf::interval<size_t>( 1,N ) ,
                                      atf::divides( N/WPT )        );

    // Step 2: Implement a Cost Function
    auto saxpy_kernel = atf::cuda::kernel< atf::scalar<int>   ,  // N
                                           atf::scalar<float> ,  // a
                                           atf::buffer<float> ,  // x
                                           atf::buffer<float> >  // y
                                         ( atf::source(saxpy_kernel_as_string), "saxpy" );  // kernel's code & name

    auto cf_saxpy = atf::cuda::cost_function( saxpy_kernel ).device_id( 0 )                     // CUDA device id
                                                            .inputs( atf::scalar<int>( N )   ,  // N
                                                                     atf::scalar<float>()    ,  // a
                                                                     atf::buffer<float>( N ) ,  // x
                                                                     atf::buffer<float>( N ) )  // y
                                                            .grid_dim( N/WPT/LS )               // CUDA grid dim
                                                            .block_dim( LS );                   // CUDA block dim

    // Step 3: Explore the Search Space
    auto tuning_result = atf::tuner().tuning_parameters( WPT, LS )
                                     .search_technique( atf::auc_bandit() )
                                     .tune( cf_saxpy, atf::evaluations(50) );
}
~~~

{% endtab %}

{% endtabs %}

</div>


## Using ATF for Conventional Search Spaces

We show how ATF is used for auto-tuning the [C++ Matrix Multiplication](https://github.com/jansel/opentuner/blob/master/examples/tutorials/mmm_block.cpp) example from [OpenTuner](http://opentuner.org) -- a state-of-the-art auto-tuning approach for programs relying on conventional search spaces, i.e., whose tuning parameters have no interdependencies among them.

The matrix multiplication implementation uses as tuning parameter the `BLOCK_SIZE` which is a value within the range `1,...,10` according to the [OpenTuner's tuning program for matrix multiplication](https://github.com/jansel/opentuner/blob/master/examples/tutorials/mmm_tuner.py).

The corresponding, equivalent OpenTuner program for auto-tuning the C++ matrix multiplication implementation is available [here](https://github.com/jansel/opentuner/blob/master/examples/tutorials/mmm_tuner.py).

Our [Experiments](/experiments) confirm that we achieve the same high auto-tuning quality as OpenTuner for programs with unconstrained search spaces.

<div class="tab-container">

{% tabs saxpy %}

{% tab saxpy pyATF %}

<hr>

We conveniently auto-tune the matrix multiplication implementation via [pyATF](https://github.com/atf-tuner/pyATF) using [pyATF](https://github.com/atf-tuner/pyATF)'s pre-implemented cost function `generic.CostFunction` which allows auto-tuning programs implemented in arbitrary programming languages. The full example is available [here](https://github.com/atf-tuner/pyATF/blob/main/examples/full_examples/bash__opentuner_matmul/bash__opentuner_matmul.py).

~~~python
from pyatf import TP, Interval, Tuner
from pyatf.cost_functions import generic
from pyatf.search_techniques import Exhaustive

# Step 1: Generate the Search Space
BLOCK_SIZE = TP('BLOCK_SIZE', Interval(1, 10))

# Step 2: Implement a Cost Function
run_command     = './tmp.bin'
compile_command = 'g++ ../mmm_block.cpp -DBLOCK_SIZE=$BLOCK_SIZE -o ./tmp.bin'

cf_matmul = generic.CostFunction(run_command).compile_command(compile_command)

# Step 3: Explore the Search Space
tuning_result = Tuner().tuning_parameters( BLOCK_SIZE )   \
                       .search_technique( Exhaustive() )  \
                       .tune( cf_matmul )
~~~

{% endtab %}

{% tab saxpy cppATF %}

<hr>

We conveniently auto-tune the matrix multiplication implementation via [cppATF](https://github.com/atf-tuner/cppATF) using [cppATF](https://github.com/atf-tuner/cppATF)'s pre-implemented cost function `atf::generic::cost_function` which allows auto-tuning programs implemented in arbitrary programming languages. The full example is available [here](https://github.com/atf-tuner/cppATF/blob/main/examples/full_examples/bash__opentuner_matmul/bash__opentuner_matmul.cpp).

~~~ c++
#include <atf.hpp>

int main()
{
  // Step 1: Generate the Search Space
  auto BLOCK_SIZE = atf::tuning_parameter( "BLOCK_SIZE", atf::interval<int>( 1,10 ) );

  // Step 2: Implement a Cost Function
  auto run_command     = "./run.bin";
  auto compile_command = "g++ mmm_block.cpp -DBLOCK_SIZE=$BLOCK_SIZE -o ./run.bin";

  auto cf_matmul = atf::generic::cost_function( run_command ).compile_command( compile_command );

  // Step 3: Explore the Search Space
  auto tuning_result = atf::tuner().tuning_parameters( BLOCK_SIZE )
                                   .search_technique( atf::auc_bandit() )
                                   .tune( cf_matmul, atf::duration<std::chrono::seconds>( 30 ) );
}
~~~

*Note:* Since [cppATF](https://github.com/atf-tuner/cppATF) is implemented in C++, the same as the matrix multiplication example to tune, we can use as cost function for this example also [cppATF](https://github.com/atf-tuner/cppATF)'s pre-implemented cost function `atf::cxx::cost_function` (as demonstrated [here](https://github.com/atf-tuner/cppATF/blob/main/examples/full_examples/cpp__opentuner_matmul/cpp__opentuner_matmul.cpp)). This cost function is more convenient to use and also more performant for auto-tuning C++ programs via [cppATF](https://github.com/atf-tuner/cppATF) than our cost function `atf::generic::cost_function` which is generic in the programming language the program to tune is implemented in. We use function `atf::generic::cost_function` in this example for demonstration.

{% endtab %}

{% endtabs %}

</div>

<br>
# Selected Publications

1.  R.Schulze, S. Gorlatch, A. Rasch \\
    [pyATF: Constraint-Based Auto-Tuning in Python](https://dl.acm.org/doi/10.1145/3708493.3712682) \\
    *ACM SIGPLAN International Conference on Compiler Construction (CC 2025)*\\
    <a href="assets/files/paper/cc25/paper.pdf"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em"></i></a> [Paper](assets/files/paper/cc25/paper.pdf)
   
3.  A. Rasch, R. Schulze, M. Steuwer, S. Gorlatch \\
    [Efficient Auto-Tuning of Parallel Programs With Interdependent Tuning Parameters via Auto-Tuning Framework (ATF)](https://dl.acm.org/doi/abs/10.1145/3427093) \\
    *ACM Transactions on Architecture and Code Optimization (TACO 2021) -- Presented at HiPEAC'21 conference*\\
    <a href="assets/files/paper/taco21/paper.pdf"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em"></i></a> [Paper](assets/files/paper/taco21/paper.pdf)
    <a href="assets/files/paper/taco21/slides.pdf"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em; padding-left: 1em"></i></a> [Slides](assets/files/paper/taco21/slides.pdf)
    <a href="https://www.youtube.com/watch?v=PRUbf1R-lZ0"><i class="fas fa-video" style="color: red; font-size: 2em; padding-top: .4em; padding-left: 1em"></i></a> [Talk](https://www.youtube.com/watch?v=PRUbf1R-lZ0)

4.  A. Rasch, S. Gorlatch \\
    [ATF: A Generic, Directive-Based Auto-Tuning Framework](https://onlinelibrary.wiley.com/doi/abs/10.1002/cpe.4423) \\
    *Concurrency and Computation: Practice and Experience (CCPE 2018)*\\
    <a href="assets/files/paper/ccpe18/paper.pdf"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em"></i></a> [Paper](assets/files/paper/ccpe18/paper.pdf)

5.  A. Rasch, M. Haidl, S. Gorlatch \\
    [ATF: A Generic Auto-Tuning Framework](https://ieeexplore.ieee.org/document/8291912) \\
    *IEEE International Conference on High Performance Computing and Communications (HPCC 2017)*\\
    <a href="assets/files/paper/hpcc17/paper.pdf"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em"></i></a> [Paper](assets/files/paper/hpcc17/paper.pdf)
    <a href="assets/files/paper/hpcc17/slides.pdf"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em; padding-left: 1em"></i></a> [Slides](assets/files/paper/hpcc17/slides.pdf)

### Talks

1.  A. Rasch, R. Schulze \\
    [Auto-Tuning Framework (ATF)](https://www.lorentzcenter.nl/generic-autotuning-technology-for-gpu-applications.html) \\
    *Generic Autotuning Technology for GPU Applications (Lorentz Center 2022), (invited talk)*\\
    <a href="assets/files/paper/lorentz22/slides.pdf"><i class="fas fa-file-pdf" style="color: red; font-size: 2em; padding-top: .4em"></i></a> [Slides](assets/files/paper/lorentz22/slides.pdf)


<br>
# Citations

Please use the following citation, when referring to ATF's:

1. *Internal Design & Implementation*
~~~
@article{10.1145/3427093,
  author = {Rasch, Ari and Schulze, Richard and Steuwer, Michel and Gorlatch, Sergei},
  title = {Efficient Auto-Tuning of Parallel Programs with Interdependent Tuning Parameters via Auto-Tuning Framework (ATF)},
  year = {2021},
  issue_date = {March 2021},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  volume = {18},
  number = {1},
  issn = {1544-3566},
  url = {https://doi.org/10.1145/3427093},
  doi = {10.1145/3427093},
  journal = {ACM Trans. Archit. Code Optim.},
  month = {jan},
  articleno = {1},
  numpages = {26},
  keywords = {parallel programs, Auto-tuning, interdependent tuning parameters}
}
~~~

2. *DSL-Based* User Interface
~~~
@article{https://doi.org/10.1002/cpe.4423,
  author = {Rasch, Ari and Gorlatch, Sergei},
  title = {ATF: A generic directive-based auto-tuning framework},
  journal = {Concurrency and Computation: Practice and Experience},
  volume = {31},
  number = {5},
  pages = {e4423},
  keywords = {auto-tuning, CLBlast, CLTune, CUDA, dependent tuning parameters, GEMM, many-core, multi-core, multi-objective auto-tuning, OpenCL, OpenTuner, tuning parameter constraints},
  doi = {https://doi.org/10.1002/cpe.4423},
  url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/cpe.4423},
  eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1002/cpe.4423},
  note = {e4423 cpe.4423},
  year = {2019}
}
~~~

3. *GPL-Based* User Interfaces:
   
    1. *Python*-Based Interface
    ~~~
    @inproceedings{10.1145/3708493.3712682, 
      author = {Schulze, Richard and Gorlatch, Sergei and Rasch, Ari}, 
      title = {pyATF: Constraint-Based Auto-Tuning in Python}, 
      year = {2025}, 
      isbn = {9798400714078}, 
      publisher = {Association for Computing Machinery}, 
      address = {New York, NY, USA}, 
      url = {https://doi.org/10.1145/3708493.3712682}, 
      doi = {10.1145/3708493.3712682},
      booktitle = {Proceedings of the 34th ACM SIGPLAN International Conference on Compiler Construction}, 
      pages = {35–47}, 
      numpages = {13}, 
      keywords = {CUDA, OpenCL, auto-tuning, constraints}, 
      location = {Las Vegas, NV, USA}, 
      series = {CC '25} 
    }
    ~~~
  
    2. *C++*-Based Interface
    ~~~
    @INPROCEEDINGS{8291912,
      author={Rasch, Ari and Haidl, Michael and Gorlatch, Sergei},
      booktitle={2017 IEEE 19th International Conference on High Performance Computing and Communications; IEEE 15th International Conference on Smart City; IEEE 3rd International Conference on Data Science and Systems (HPCC/SmartCity/DSS)},
      title={ATF: A Generic Auto-Tuning Framework},
      year={2017},
      volume={},
      number={},
      pages={64-71},
      doi={10.1109/HPCC-SmartCity-DSS.2017.9}
    }
    ~~~

<br>
# Contact

<div class="card_container">
  <div class="card">
    <div class="card_content">
      <a href="https://arirasch.net" style="color: black"><img src="assets/images/ari.JPG" alt="Avatar"></a>
      <a href="https://arirasch.net" style="color: black"><h4><b>Ari Rasch</b></h4></a>
      <table>
        <tr><td>Focus:</td><td>Project Lead</td></tr>
        <tr><td>Affiliation:</td><td><a href="https://www.uni-muenster.de/en/">University of Münster</a></td></tr>
        <tr><td>Email:</td><td><a href="mailto:a.rasch@uni-muenster.de?cc=r.schulze@uni-muenster.de">a.rasch@uni-muenster.de</a></td></tr>
        <tr><td>Website:</td><td><a href="https://www.arirasch.net">arirasch.net</a></td></tr>
      </table>
    </div>
  </div>
  <div class="card">
    <div class="card_content">
      <a href="https://richardschulze.net" style="color: black"><img src="assets/images/richard.PNG" alt="Avatar"></a>
      <a href="https://richardschulze.net" style="color: black"><h4><b>Richard Schulze</b></h4></a>
      <table>
        <tr><td>Focus:</td><td>Technical Lead</td></tr>
        <tr><td>Affiliation:</td><td><a href="https://www.uni-muenster.de/en/">University of Münster</a></td></tr>
        <tr><td>Email:</td><td><a href="mailto:r.schulze@uni-muenster.de?cc=a.rasch@uni-muenster.de">r.schulze@uni-muenster.de</a></td></tr>
        <tr><td>Website:</td><td><a href="https://www.richardschulze.net">richardschulze.net</a></td></tr>
      </table>
    </div>
  </div>
</div>

<br>

You can also find us on <a href="https://discord.gg/kVRZRyUDQY"><img src="assets/images/discord-logo.svg" alt="Discord" style="height: 20px"> Discord</a>.
