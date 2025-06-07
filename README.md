# City-Scale Path Finder

## Features

- Compare Dijkstra, Dial, A*, and Bidirectional A* on real city road data
- FastAPI backend with REST API
- Streamlit UI for interactive routing and benchmarking
- CLI tool for headless benchmarking and plotting

## Setup

```bash
pip install -r requirements.txt
```

## Run API

```bash
python api.py
```

## Run Streamlit UI

```bash
streamlit run streamlit_ui.py
```

## Run CLI Benchmark

```bash
python cli_benchmark.py --tests 50 --generate-plots
```

## Data

- Place your `central_bengaluru.graphml` (or your own GraphML file) in the project root.
- The API will load the GraphML file directly for all features.
