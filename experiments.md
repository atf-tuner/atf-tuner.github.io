---
layout: single
---

# Experimental Results

The experimental results presented in the following are described in detail [here](https://dl.acm.org/doi/abs/10.1145/3427093).

## Preliminary Remark
ATF is successfully used in literature for auto-tuning applications from different important domains, summarized in the following table:

![Applications Auto-Tuned via ATF](/assets/images/atf_application.png)

## Auto-Tuning Efficiency

ATF compared to state-of-the-art approaches, on NVIDIA GPU and Intel CPU, for application case studies:

1. `CONV`: Convolution,
2. `GEMM`: Matrix Multiplication,
3. `CCSD(T)`: Coupled Cluster,
4. `PRL`: Probabilistic Record Linkage

![Applications Auto-Tuned via ATF](/assets/images/atf_eval_at_efficiency_gpu.png)
![Applications Auto-Tuned via ATF](/assets/images/atf_eval_at_efficiency_cpu.png)

## Generating & Storing & Exploring Constrained Search Spaces

ATF analyzed for each individual phase of the auto-tuning process.

### Generating Constrained Search Spaces

Search space *generation time* (lower is better) of ATF compared to a [Constraint Solver (CS)](https://pypi.org/project/python-constraint/) and [CLTune](https://github.com/CNugteren/CLTune).
Here, we use the following abbreviations: `s` for *seconds*; `h` for *hours*; `m` for *months*; `c` for *centuries*.

![Applications Auto-Tuned via ATF](/assets/images/atf_eval_generation.png)

### Storing Constrained Search Spaces

Search space *memory footprint* (lower is better) of ATF compared to [CLTune](https://github.com/CNugteren/CLTune).

![Applications Auto-Tuned via ATF](/assets/images/atf_eval_storing.png)

### Exploring Constrained Search Spaces

Search space *exploration efficiency* (lower is better) of ATF compared to [CLTune](https://github.com/CNugteren/CLTune).

![Applications Auto-Tuned via ATF](/assets/images/atf_eval_exploring.png)

## ATF for CLTune's Target Application Class

ATF vs CLTune for CLTune's own running example [2D Convolution](https://github.com/CNugteren/CLTune/tree/master/samples/conv) (described in detail [here](https://onlinelibrary.wiley.com/doi/abs/10.1002/cpe.4423))

![ATF vs CLTune](/assets/images/atf_eval_atf_vs_cltune.png)

## ATF for OpenTuner's Target Application Class

ATF vs OpenTuner for OpenTuner's own running example [GCC Flags](https://github.com/jansel/opentuner/tree/master/examples/gccflags) (described in detail [here](https://onlinelibrary.wiley.com/doi/abs/10.1002/cpe.4423))

![ATF vs OpenTuner](/assets/images/atf_eval_atf_vs_ot.png)