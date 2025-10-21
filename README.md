[![Stars](https://img.shields.io/github/stars/laminlabs/modlyn?logo=GitHub)](https://github.com/laminlabs/modlyn)
[![codecov](https://codecov.io/gh/laminlabs/modlyn/branch/main/graph/badge.svg?token=VKMRJ7OWR3)](https://codecov.io/gh/laminlabs/modlyn)
[![Docs](https://img.shields.io/badge/docs-humans-yellow)](https://modlyn.lamin.ai)
[![DocsLLMs](https://img.shields.io/badge/docs-LLMs-yellow)](https://modlyn.lamin.ai/summary.md)
[![pypi](https://img.shields.io/pypi/v/modlyn?color=blue&label=pypi%20package)](https://pypi.org/project/modlyn)
[![PyPI Downloads](https://img.shields.io/pepy/dt/modlyn?logo=pypi)](https://pepy.tech/project/modlyn)

# Modlyn: Feature selection for large count matrices

This project scales model-based feature selection techniques to very large datasets.
It is co-developed with the [arrayloaders](https://github.com/laminlabs/arrayloaders) package.

## Features

- **Multiple feature selection methods**: Simple logistic regression, ElasticNet, RandomForest, Mutual Information
- **Scalable training**: Supports in-memory and Dask-backed datasets for large-scale data (100k+ cells)
- **PyTorch Lightning integration**: GPU-ready, flexible training workflows
- **Quantitative comparison**: Built-in tools to compare methods using Jaccard overlap and visualizations
- **AnnData native**: Seamless integration with scanpy and the single-cell ecosystem

## Quick Links

- [Quickstart](https://modlyn.lamin.ai/quickstart): End-to-end example comparing multiple methods
- [Benchmarks](https://modlyn.lamin.ai/benchmarks): Performance comparisons and recommendations
- [API Reference](https://modlyn.lamin.ai/reference): Complete API documentation

## Contributing

Please run `pre-commit install` and `gitmoji -i` on the CLI before starting to work on this repository!
