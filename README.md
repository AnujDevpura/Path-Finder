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
python cli_benchmark.py --data bengaluru_roads.csv --tests 50 --generate-plots
```

## Data

- Place your `bengaluru_roads.csv` in the project root.
- The API will auto-convert it to JSON for faster future loads.
