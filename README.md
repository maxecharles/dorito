# dorito

![Io rotating on it's axis!](assets/io.gif)

`dorito` is an image reconstruction framework designed to interface with [`amigo`](https://github.com/LouisDesdoigts/amigo), an end-to-end differentiable forward-modelling pipeline for analysing data from the Aperture Masking Interferometer (AMI) observing mode of the *James Webb Space Telescope* (JWST).

Seen in the gif above are five successive images of Jupiter's moon Io that have been deconvolved with `dorito`, with the moon's axial rotation clearly visible. The right image shows the expected positions of known volcanic features on Io's hellscape of a surface!

Please refer to our publications for more detail: [amigo paper here](https://arxiv.org/abs/2510.09806), [dorito paper here](https://arxiv.org/abs/2510.10924).
<!-- 
- Differentiable
- Optics for
- Reconstructing
- Images from
- Telescope
- Observations -->

We are working hard to create more documentation and example notebooks to share with the community here. Thank you for your patience, and please do not hesitate to reach out with any questions or queries. You can reach me (Max) at
```text
max.charles@sydney.edu.au
```

## Installation

`dorito` is an extension of `amigo` and depends heavily on its runtime and modelling APIs, so you will first need to install `amigo` from source.

Basic walk-through to install `amigo` and `dorito` from source (recommended):

1. Create and activate a clean environment (example uses `conda` but `venv` works too):

```bash
conda create -n your_env
conda activate your_env
```

2. Clone and install `amigo` from source:

```bash
git clone https://github.com/LouisDesdoigts/amigo.git
cd amigo
pip install .
cd ..
```

3. Clone and install `dorito` (from this repository) into the same environment:

```bash
git clone https://github.com/maxecharles/dorito.git
cd dorito
pip install .
```
