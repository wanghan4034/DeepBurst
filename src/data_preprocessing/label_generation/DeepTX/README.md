# DeepTX

This repository contains the implementation of **DeepTX**.

## Environment

DeepTX was tested with:

```text
Julia 1.7.3
Conda
```

## Installation

Create a conda environment:

```bash
conda create -n deeptx -c conda-forge julia=1.7.3
conda activate deeptx
```

Check Julia version:

```bash
julia --version
```

Expected output:

```text
julia version 1.7.3
```

## Julia packages

The following Julia packages are required:

```text
BlackBoxOptim v0.6.2
CSV v0.10.8
Catalyst v12.3.1
CodecBzip2 v0.7.2
Colors v0.12.11
DataFrames v1.4.4
DataFramesMeta v0.13.0
DataVoyager v1.0.2
DelaySSAToolkit v0.2.3
DiffEqBase v6.110.1
DiffEqJump v8.6.3
Distances v0.10.7
Distributions v0.25.79
FiniteStateProjection v0.2.1
Flux v0.13.9
FreqTables v0.4.6
GLM v1.8.3
GR v0.71.1
HTTP v0.9.17
Images v0.25.2
Interp1d v0.1.0
JLD2 v0.4.29
JSON v0.21.4
JSON3 v1.14.2
JuliaFormatter v1.0.34
JuliaZH v1.6.0
KernelDensity v0.6.9
LaTeXStrings v1.3.0
MAT v0.10.3
MLDatasets v0.7.8
MatrixMarket v0.5.2
OptimalTransport v0.3.19
OrdinaryDiffEq v6.35.1
Pardiso v0.5.4
PkgMirrors v1.3.0
PlotlyBase v0.8.19
PlotlyJS v0.18.10
Plots v1.37.2
ProgressMeter v1.7.2
PyPlot v2.11.0
Setfield v1.1.1
Sobol v1.5.0
SpecialFunctions v2.1.7
StaticArrays v1.5.11
StatsBase v0.33.21
Sundials v4.11.4
VegaDatasets v2.1.1
VegaLite v2.6.0
Zygote v0.6.55
ZygoteRules v0.2.2
```

## Setup

In the project root directory:

```bash
julia --project=.
```

Then run:

```julia
using Pkg
Pkg.instantiate()
Pkg.precompile()
```

If the environment files are not provided, install packages manually:

```julia
using Pkg

Pkg.add([
    PackageSpec(name="BlackBoxOptim", version="0.6.2"),
    PackageSpec(name="CSV", version="0.10.8"),
    PackageSpec(name="Catalyst", version="12.3.1"),
    PackageSpec(name="CodecBzip2", version="0.7.2"),
    PackageSpec(name="Colors", version="0.12.11"),
    PackageSpec(name="DataFrames", version="1.4.4"),
    PackageSpec(name="DataFramesMeta", version="0.13.0"),
    PackageSpec(name="DataVoyager", version="1.0.2"),
    PackageSpec(name="DelaySSAToolkit", version="0.2.3"),
    PackageSpec(name="DiffEqBase", version="6.110.1"),
    PackageSpec(name="DiffEqJump", version="8.6.3"),
    PackageSpec(name="Distances", version="0.10.7"),
    PackageSpec(name="Distributions", version="0.25.79"),
    PackageSpec(name="FiniteStateProjection", version="0.2.1"),
    PackageSpec(name="Flux", version="0.13.9"),
    PackageSpec(name="FreqTables", version="0.4.6"),
    PackageSpec(name="GLM", version="1.8.3"),
    PackageSpec(name="GR", version="0.71.1"),
    PackageSpec(name="HTTP", version="0.9.17"),
    PackageSpec(name="Images", version="0.25.2"),
    PackageSpec(name="JLD2", version="0.4.29"),
    PackageSpec(name="JSON", version="0.21.4"),
    PackageSpec(name="JSON3", version="1.14.2"),
    PackageSpec(name="JuliaFormatter", version="1.0.34"),
    PackageSpec(name="JuliaZH", version="1.6.0"),
    PackageSpec(name="KernelDensity", version="0.6.9"),
    PackageSpec(name="LaTeXStrings", version="1.3.0"),
    PackageSpec(name="MAT", version="0.10.3"),
    PackageSpec(name="MLDatasets", version="0.7.8"),
    PackageSpec(name="MatrixMarket", version="0.5.2"),
    PackageSpec(name="OptimalTransport", version="0.3.19"),
    PackageSpec(name="OrdinaryDiffEq", version="6.35.1"),
    PackageSpec(name="Pardiso", version="0.5.4"),
    PackageSpec(name="PkgMirrors", version="1.3.0"),
    PackageSpec(name="PlotlyBase", version="0.8.19"),
    PackageSpec(name="PlotlyJS", version="0.18.10"),
    PackageSpec(name="Plots", version="1.37.2"),
    PackageSpec(name="ProgressMeter", version="1.7.2"),
    PackageSpec(name="PyPlot", version="2.11.0"),
    PackageSpec(name="Setfield", version="1.1.1"),
    PackageSpec(name="Sobol", version="1.5.0"),
    PackageSpec(name="SpecialFunctions", version="2.1.7"),
    PackageSpec(name="StaticArrays", version="1.5.11"),
    PackageSpec(name="StatsBase", version="0.33.21"),
    PackageSpec(name="Sundials", version="4.11.4"),
    PackageSpec(name="VegaDatasets", version="2.1.1"),
    PackageSpec(name="VegaLite", version="2.6.0"),
    PackageSpec(name="Zygote", version="0.6.55"),
    PackageSpec(name="ZygoteRules", version="0.2.2")
])

Pkg.add(url="https://github.com/AtsushiSakai/Interp1d.jl.git")
```

## Run

Run the main DeepTX script from the project root directory:

```bash
julia --project=. 
julia --project=. src/data_preprocessing/label_generation/DeepTX/TX_inferrer.jl extra/datasets/burst/raw_data/gm12878/processed_adata.csv inferred_results.csv
```