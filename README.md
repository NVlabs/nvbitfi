# NVBitFI_for_ODC

Fork of [NVBitFI](https://github.com/NVlabs/nvbitfi) used in an MSc thesis on radiation effects in GPUs for orbital data centers. The fault injection results are combined with proton-beam data to map which GPU subsystems need protection.

Main changes with respect to NVBitFI (tested with NVBit 1.5.5, Volta/Turing):

* **NVBit 1.8, CUDA 12, Blackwell (sm_120).** Shared build settings in `common/build.mk` (`NVBIT_PATH`, `ARCH`, `DUMMY`). All kernel launch APIs are intercepted, including `cuLaunchKernelEx`, which CUDA 12 runtimes use.
* **Opcode table for current GPUs.** `common/arch.h` holds one table (opcode → group) generated with an X-macro. It covers the Blackwell instruction set (CUDA Binary Utilities 12.9) and every opcode in the sm_120 SASS of cuBLAS/cuBLASLt 12.9, which has zero unknown opcodes. The original table missed 49 opcodes used on sm_120 (e.g. `LDCU`, `UIADD3`, `R2UR`, `QMMA`, `DMMA`, `LDGSTS`, `LDSM`, `F2FP`) and counted them as `FADD`. Opcodes still missing go to `G_UNKNOWN`: they are reported at run time and never injected.
* **Destination registers from NVBit's decoded operands** (`common/dests.h`) instead of string parsing. It separates GPRs, predicate registers, uniform registers and uniform predicates. It ignores discard targets (`RZ`, `URZ`, `PT`, `UPT`). It computes the right width of 64/128-bit, `IMAD.WIDE` and tensor-core results. Tested offline on 11 M sm_120 instructions (`tests/dests_test.cpp`).
* **Injection into every register class** with the NVBit 1.8 device API: GPRs, **predicate registers**, **uniform registers** and **uniform predicates**.
* **New instruction groups**: `tc` (tensor cores), `sfu` (MUFU), `fp16` (packed FP16/BF16/FP8), `uniform` (uniform datapath), `unknown`.
* **Two-phase injection.** The target is selected BEFORE the instruction, with the same rules and the same instrumentation point as the profiler. It is corrupted AFTER the instruction.
* **Hardware location in the log**: SM id, hardware warp slot, lane, CTA, register class and number.
* **One JSON record per injection** (`results-*.jsonl`), ready for the ML pipeline.
* **Bug fixes**: `SUST` was in two groups, `HSETP2` was in the wrong group (assert crash with `G_GP`), off-by-one in injection-site sampling, `__ballot` illegal on sm_70+, pf_injector deadlock past `INSTRUMENTATION_LIMIT` and injection into predicated-out instructions, Python 3 issues (`map` is lazy so SDC outputs were never saved, `dmesg` bytes/empty output, timeout check).

## Requirements

| | |
|---|---|
| OS | Linux x86_64 (NVBit does not run on Windows; WSL is not supported) |
| NVBit | 1.8 (`nvbit-Linux-x86_64-1.8.tar.bz2` from the NVBit releases) |
| CUDA toolkit | 12.0 or newer; 12.8 or newer for sm_120 |
| NVIDIA driver | **575.xx or older** (NVBit 1.8 requirement) |
| GCC | 8.5 or newer |

## Build

```console
wget https://github.com/NVlabs/NVBit/releases/download/v1.8/nvbit-Linux-x86_64-1.8.tar.bz2
tar xvfj nvbit-Linux-x86_64-1.8.tar.bz2
cd nvbit_release_x86_64/tools/
git clone https://github.com/tommasolanzini/NVBitFI_for_ODC nvbitfi
cd nvbitfi
make -C injector ARCH=sm_120     # ARCH=all (default) builds for every supported GPU
make -C profiler ARCH=sm_120
make -C pf_injector ARCH=sm_120
```

If the repository is somewhere else, pass `NVBIT_PATH=/path/to/nvbit_release_x86_64/core`.

## Instruction groups

Each opcode belongs to one exclusive group, given by the unit that produces its result. `gppr` and `gp` are aggregates defined by the destination operands.

| id | group | what | injectable destinations |
|---|---|---|---|
| 0 | fp64 | DADD, DFMA, DMUL, ... | all destinations (GPR + PR) |
| 1 | fp32 | FADD, FFMA, FMUL, FMNMX, ... | all destinations |
| 2 | ld | LDG, LDS, LDSM, LDC, LDL, ... | all destinations |
| 3 | pr | ISETP, FSETP, DSETP, PLOP3, R2P, ... | predicate registers |
| 4 | nodest | stores, branches, barriers, LDGSTS, REDG, ... | none |
| 5 | others | integer, conversion, move, shuffle, atomics, ... | all destinations |
| 6 | tc | HMMA, IMMA, DMMA, QMMA, OMMA | all destinations |
| 7 | sfu | MUFU | all destinations |
| 8 | fp16 | HADD2, HFMA2, HMUL2, QFMA4, ... | all destinations |
| 9 | uniform | UIADD3, ULEA, LDCU, S2UR, R2UR, VOTEU, ... | uniform registers + uniform predicates |
| 10 | unknown | opcodes missing from `arch.h` | none |
| 11 | gppr | any non-uniform instruction writing a GPR or PR | GPRs + predicate registers |
| 12 | gp | any non-uniform instruction writing a GPR | GPRs (original NVBitFI default) |

Profile counts per group are the **injectable** dynamic instances, i.e. those with at least one destination of the right class and the guard predicate true. Uniform-datapath instructions execute once per warp and are counted once per warp. For 1-bit destinations (predicates), `FLIP_SINGLE_BIT` and `FLIP_TWO_BITS` invert the predicate, `RANDOM_VALUE` writes a random 0/1 and `ZERO_VALUE` writes 0.

## Injection log and results

`nvbitfi-injection-log-temp.txt` now also contains `regType` (GPR/PR/UR/UP), `smID`, `warpID` (hardware warp slot), `laneID`, `ctaID` and `groupInstrs`. `groupInstrs` is the number of injectable instances of the group in the kernel; it must match the profile.

Next to every `results-*.txt` (same format as NVBitFI), `run_one_injection.py` writes `results-*.jsonl` with one JSON object per injection: app, group, model, kernel, PC, opcode, thread, CTA, SM, warp slot, lane, register class and number, mask, before/after value, outcome category, runtime and dmesg.

## First run on a new GPU (checklist)

1. `make -C tests` and `./tests/dests_test --check`, the offline destination analysis (no GPU needed).
2. Profile with `TOOL_VERBOSE=1`. Every static instruction is printed with its group and destinations. Check that there are no `opcode ... is not in common/arch.h` warnings.
3. Build the injector with `DUMMY=1` and run a short campaign. Every outcome must be *Masked* and `groupInstrs` must match the profile.
4. Run a few injections per group with `TOOL_VERBOSE=1`. The log must show `regType`/`regNo`, and no `Expected reg value doesn't match` (sanity check of the NVBit register-read path).
5. Uniform registers and predicates are written by a single lane, the warp leader. Check with a few `G_UNIFORM` and `G_PR` injections that the corrupted value is visible to the whole warp (UR/UP) or to the thread (PR).

---

# Original NVBitFI README

# NVBitFI: An Architecture-level Fault Injection Tool for GPU Application Resilience Evaluations

NVBitFI provides an automated framework to perform error injection campaigns for GPU application resilience evaluation.  NVBitFI builds on top of [**NV**IDIA **Bi**nary **I**nstrumentation **T**ool (NVBit)](https://github.com/NVlabs/NVbit), which is a research prototype of a dynamic binary instrumentation library for NVIDIA GPUs. NVBitFI offers functionality that is similar to a prior tool called [SASSIFI](https://github.com/NVlabs/sassifi).  

Please refer to our [NVBitFI paper](https://ieeexplore.ieee.org/abstract/document/9505068) for additional details about the tool and some experimental results.

# Summary of NVBitFI's capabilities 

NVBitFI injects errors into the destination register values of a dynamic thread-instruction by instrumenting instructions after they are executed.  A dynamic instruction is selected at random from all dynamic kernels of a program for error injection.  Only one error is injected per run.  This mode was referred to as IOV in SASSIFI.  As of now (4/1/2020), NVBitFI allows us to select the following instruction groups to study how errors in them can propagate to the application output.

 * Instructions that write to general purpose registers
 * Single precision floating point instructions
 * Double precision floating point instructions 
 * Load instructions 

NVBitFI can be extended to include custom instruction groups. See below for more details. 

For a selected destination register, following errors can be injected. 

 * Single bit-flip: one bit-flip in one register in one thread
 * Double bit-flip: bit-flips in two adjacent bits in one register in one thread
 * Random value: random value in one register in one thread
 * Zero value: zero out the value of one register in one thread

New bit-flip models can be added by modifying common/arch.h and injector/inject\_func.cu and scripts/params.py. 


# Prerequisites
 * [NVBit v1.5.5](https://github.com/NVlabs/NVBit/releases/tag/1.5.5) or newer
 * [System requirements](https://github.com/NVlabs/NVbit#requirements)

# Getting started on a Linux x86\_64 PC
The following commands are tested on an x86 system with Ubuntu 18.04 using CUDA-11.2 and NVBit version 1.5.5.
```console
# NVBit-v1.5.5
wget https://github.com/NVlabs/NVBit/releases/download/1.5.5/nvbit-Linux-x86_64-1.5.5.tar.bz2
tar xvfj nvbit-Linux-x86_64-1.5.5.tar.bz2
cd nvbit_release/tools/

# NVBitFI 
git clone https://github.com/NVlabs/nvbitfi
cd nvbitfi
find . -name "*.sh" | xargs chmod +x
./test.sh
```
On an ARM-based device (e.g., Jetson Nano)
```console
# NVBit-1.5.5
wget https://github.com/NVlabs/NVBit/releases/download/1.5.5/nvbit-Linux-aarch64-1.5.5.tar.bz2
tar xvfj nvbit-Linux-aarch64-1.5.5.tar.bz2
cd nvbit_release/tools/

# NVBitFI 
git clone https://github.com/NVlabs/nvbitfi
cd nvbitfi
find . -name "*.sh" | xargs chmod +x
./test.sh
```

If these commands complete without errors, you just completed your first error injection campaign using NVBitFI. The printed output should say where the results are stored. Summary of the campaign is stored in a tab-separated file, `results_*NVbitFI_details.tsv`. It can be opened using a spreadsheet program (e.g., Excel) for visualization and analysis. 

# Detailed steps 

There are three main steps to run NVBitFI. We provide a sample script (test.sh) that automates nearly all these steps.

## Step 0: Setup

 * One-time only: Copy NVBitFI package to tool directory in NVBit installation (see the above commands) 
 * Every time we run an injection campaign: Setup environment (see Step 0 (2) in test.sh)
 * One-time only: Build the injector and profiler tools (see Step 0 (3) in test.sh)
 * One-time only: Run and collect golden stdout and stderr files for each of the applications (see Step 0 (4) in test.sh). 
    * Record fault-free outputs: Record golden output file (as golden.txt), stdout (as golden\_stdout.txt), and stderr (as golden\_stderr.txt) in the workload directory (e.g., nvbitfi/test-apps/simple\_add).
    * Create application-specific scripts: Create run.sh and sdc\_check.sh scripts in workload directory. Instead of using absolute paths, please use environment variables for paths such as BIN\_DIR, APP\_DIR, and DATASET\_DIR. These variables are set in set\_env function in scripts/common\_functions.py. See the scripts in the nvbitfi/test-apps/simple\_add directory for examples.
    * Workloads will be run from logs/workload-name/run-name directory. It would be great if the workload can run from this directory. If the program requires input files to be in a specific location, either update the workload or provide soft links to the input files in appropriate locations. 
    * The program output should be deterministic. Please exclude non-deterministic values (e.g., runtimes) from the file if they are present in one of the output files (see test-apps/simple\_add/sdc\_check.sh for more details).

## Step 1: Profile and generate injection list

 * Profile the application: Run the program once by using profiler/profiler.so. We provide scripts/run\_profiler.py script for this step. A new file named nvbitfi-igprofile.txt will be generated in logs/workload-name directory. This file contains the instruction counts for all the instruction groups and opcodes defined in common/arch.h. One line is created per dynamic kernel invocation.
   Profiling is often slow as it instruments every instruction in every dynamic kernel. Using an approximate profile can speed it up by orders of magnitude. There are many ways to approximate a profile and trade-off accuracy for speed. In this release we implement a method that approximates the profiles of all dynamic invocations of a static kernel with the profile of the first invocation of the static kernel. It essentially profiles all static kernels just ones, which can make the profiling very fast if a program has few static kernels and many dynamic involutions per kernel. This approximation can be enabled by using the `SKIP_PROFILED_KERNELS` flag while building the profiler. 
 * Generate injection sites:
    * Ensure that the parameters are set correctly in scripts/params.py.  Following are some of the parameters that need user attention: 
		* Setting maximum number of error injections to perform per instruction group and bit-flip model combination. See NUM\_INJECTION and THRESHOLD\_JOBS in scripts/params.py. 
		* Selecting instruction groups and bit-flip models (more details in scripts/params.py). 
		* Listing the applications, benchmark suite name, application binary file name, and the expected runtime on the system where the injection job will be run. See the apps dictionary in scripts/params.py for an example. The expected runtime defined here is used later to determine when to timeout injection runs (based on the TIMEOUT\_THRESHOLD defined in scripts/params.py).
    * Run scripts/generate\_injection\_list.py to generate a file that contains a list of errors to be injected during the injection campaign. Instructions are selected randomly from the instructions of the selected instruction group. 

## Step 2: Run the error injection campaign

Run scripts/run\_injections.py to launch the error injection campaign. This script will run one injection run at a time in the standalone mode.  If you plan to run multiple injection runs in parallel, please take special care to ensure that the output file is not clobbered. As of now, we support running multiple jobs on a multi-GPU system. Please see scripts/run\_one\_injection.py for more details. 

Tip: Perform a few dummy injections before proceeding with full injection campaign (by setting DUMMY flag in injector/Makefile. Setting this flag will allow you to go through most of the SASSI handler code but skip the error injection. This is to ensure that you are not seeing crashes/SDCs that you should not see.

## Step 3: Parse the results

Use the scripts/parse\_results.py script to parse the results. This script generates three tab-separated values (tsv) files. The first file shows the fraction of executed instructions for different instruction groups and opcodes. The second file shows the outcomes of the error injections.  Refer to CAT\_STR in scripts/params.py for the list of error outcome categories. The third file shows the average runtime for the injection runs for different applications and selected error models. These files can be opened using a spreadsheet program (e.g., Excel) for plotting and analysis.



# NVBitFI vs. SASSIFI

NVBitFI benefits from the featured offered by NVBit. It can run on newer GPUs (e.g., Turing and Volta GPUs). It works with pre-compiled libraries also, unlike SASSIFI. NVBitFI is expected to be faster than SASSIFI as it instruments just a single chosen dynamic kernel (SASSIFI, as it was implemented, instrumented all dynamic kernels) for the injection runs.  As of now (April 14, 2020), NVBitFI implements a subset of the error injection models and we may be expanding this over time (users are more than welcome to contribute). 



# Contributing to NVBitFI

If you are interested in contributing to NVBitFI, please initialize a [Pull Request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests) and complete the [Contributor License Agreement](https://www.apache.org/licenses/icla.pdf).
