[//]: # (to add: arxiv; pypi; ci)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue)

---

# SteerX

The SteerX toolkit is an extensible library for general purpose steering of LLMs. The toolkit allows for
the implementation of steering methods across a range of model control surfaces (input, structural, state, and output),
functionality to compose steering methods (into a `SteeringPipeline`), and the ability to compare steering methods
(and pipelines) on custom tasks/metrics.

## Installation

The toolkit uses [uv](https://docs.astral.sh/uv/) as the package manager (Python 3.11+). After installing `uv`, install
the toolkit by running:

```commandline
uv venv --python 3.11 && uv pip install .
```
Activate by running `source .venv/bin/activate`. Note that on Windows, you may need to split the above script into two separate commands (instead of chained via `&&`).

Inference is facilitated by Hugging Face. Before steering, create a `.env` file in the root directory for your Hugging
Face API key in the following format:
```
HUGGINGFACE_TOKEN=hf_***
```

Some Hugging Face models (e.g. `meta-llama/Meta-Llama-3.1-8B-Instruct`) are behind an access gate. To gain access:

1. Request access on the model's Hub page with the same account whose token you'll pass to the toolkit.
2. Wait for approval (you'll receive an email).
3. (Re-)authenticate locally by running `huggingface-cli login` if your token has expired or was never saved.


## Example library

> [!NOTE]
> SteerX runs the model inside your process. For efficient inference, please run the toolkit from a machine that
> has enough GPU memory for both the base checkpoint and the extra overhead your steering method/pipeline adds.

Notebook examples for each of the supported steering methods (and wrappers) can be found in the `examples/notebooks/` directory.


## Contributing

We invite community contributions primarily on broadening the set of steering methods (via new controls) and evaluations
(via use cases and metrics). We additionally welcome reporting of any bugs/issues, improvements to the documentation,
and new features). Specifics on how to contribute can be found in our [contribution guidelines](CONTRIBUTING.md).
To make contributing easier, we have prepared the following tutorials.


### Adding a new steering method

If there is an existing steering method that is not yet in the toolkit, or you have developed a new steering method of
your own, the toolkit has been designed to enable relatively easy contribution of new steering methods. Please see the
tutorial on [adding your own steering method](./docs/tutorials/add_new_steering_method.md) for a detailed guide


### Adding a new use case / benchmark

Use cases enable comparison of different steering methods on a common task. The `UseCase`
(`steerx/evaluation/use_cases/`) and `Benchmark` classes (`steerx/evaluation/benchmark.py`) enable this
comparison. If you'd like to compare various steering methods/pipelines on a novel use case, please see the tutorial on
[adding your own use case](./docs/tutorials/add_new_use_case.md).


### Adding a new metric

Metrics are used by a given benchmark to quantify model performance across steering pipelines in a comparable way. We've
included a selection of generic metrics (see `steerx/evaluation/metrics/`). If you'd like to add new generic metrics
or custom metrics (for a new use case), please see the tutorial on
[adding your own metric](./docs/tutorials/add_new_metric.md).
