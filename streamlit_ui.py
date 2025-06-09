import streamlit as st
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import numpy as np
from streamlit_plotly_events import plotly_events  # type: ignore
# Page config
st.set_page_config(
    page_title="City-Scale Path Finder",
    page_icon="🗺️",
    layout="wide"
)

# Default API base URL (can be overridden)
DEFAULT_API_BASE_URL = "http://localhost:8000"
API_BASE_URL = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)

# Sidebar input for API URL override
with st.sidebar.expander("API Settings", expanded=True):
    API_BASE_URL = st.text_input("API Base URL", value=API_BASE_URL)

def main():
    st.title("🗺️ City-Scale Path Finder")
    st.markdown("Compare pathfinding algorithms on real city-scale data")

    # Check API connection
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            st.sidebar.success("✅ API Connected")
            st.sidebar.metric("Nodes", stats.get("nodes", "N/A"))
            st.sidebar.metric("Edges", stats.get("edges", "N/A"))
        else:
            st.sidebar.error("❌ API Connection Failed")
            st.error("API returned error: " + response.text)
            return
    except requests.exceptions.RequestException:
        st.sidebar.error("❌ API Not Available")
        st.error("Please start the FastAPI server first or check API URL.")
        return

    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Route Finder", "📊 Benchmark", "📈 Analysis"])

    with tab1:
        route_finder_tab()

    with tab2:
        benchmark_tab()

    with tab3:
        analysis_tab()


def get_nodes(api_base_url):
    resp = requests.get(f"{api_base_url}/nodes")
    resp.raise_for_status()
    return resp.json()

def route_finder_tab():
    st.header("Find Optimal Route")

    # Fetch available nodes from API
    nodes = get_nodes(API_BASE_URL)
    df_nodes = pd.DataFrame(nodes)

    # Plot all nodes on map
    fig = go.Figure(go.Scattermapbox(
        lat=df_nodes["lat"],
        lon=df_nodes["lon"],
        mode="markers",
        marker=dict(size=8, color="blue"),
        text=df_nodes["id"].astype(str),
        customdata=df_nodes["id"],
        name="Nodes"
    ))
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=13,
        mapbox_center={"lat": df_nodes["lat"].mean(), "lon": df_nodes["lon"].mean()},
        height=500,
        margin={"r":0,"t":0,"l":0,"b":0}
    )

    st.markdown("**Click on the map to select source and destination nodes.**")
    selected_points = plotly_events(fig, click_event=True, select_event=False, override_height=500)

    # Store selections in session state
    if "src_id" not in st.session_state:
        st.session_state["src_id"] = None
    if "dst_id" not in st.session_state:
        st.session_state["dst_id"] = None

    if selected_points:
        point_idx = selected_points[0].get("pointIndex")
        if point_idx is not None:
            node_id = int(df_nodes.iloc[point_idx]["id"])
            if st.session_state["src_id"] is None:
                st.session_state["src_id"] = node_id
                st.success(f"Source node selected: {node_id}")
            elif st.session_state["dst_id"] is None and node_id != st.session_state["src_id"]:
                st.session_state["dst_id"] = node_id
                st.success(f"Destination node selected: {node_id}")

    # Show selected nodes
    src_id = st.session_state["src_id"]
    dst_id = st.session_state["dst_id"]
    st.write(f"Source: {src_id}, Destination: {dst_id}")

    # Add Reset Selection button
    if st.button("Reset Selection"):
        st.session_state["src_id"] = None
        st.session_state["dst_id"] = None

    # Algorithm selection and route finding
    algorithm = st.selectbox(
        "Select Algorithm",
        ["dijkstra", "dial", "astar", "bidirectional_astar"],
        index=0,
    )

    if src_id and dst_id and st.button("Find Route", type="primary"):
        with st.spinner("Finding optimal route..."):
            try:
                # Get node details for lat/lon
                src_node = next(n for n in nodes if n["id"] == src_id)
                dst_node = next(n for n in nodes if n["id"] == dst_id)

                payload = {
                    "src_lat": src_node["lat"],
                    "src_lon": src_node["lon"],
                    "dst_lat": dst_node["lat"],
                    "dst_lon": dst_node["lon"],
                    "algo": algorithm,
                }

                response = requests.post(f"{API_BASE_URL}/route", json=payload, timeout=10)

                if response.status_code == 200:
                    result = response.json()

                    if result.get("success"):
                        # Show metrics
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Distance", f"{result['distance']:.0f} m")
                        with col2:
                            st.metric("Runtime", f"{result['runtime_ms']:.2f} ms")
                        with col3:
                            st.metric("Nodes Expanded", result["nodes_expanded"])
                        with col4:
                            st.metric("Path Length", len(result["path"]))

                        # Build path DataFrame: Expecting 'path' as list of dicts with lat, lon, id
                        if result["path"]:
                            df_path = pd.DataFrame(result["path"])
                            if not {"lat", "lon", "id"}.issubset(df_path.columns):
                                st.warning("Path data incomplete: 'lat', 'lon', or 'id' missing.")
                            else:
                                fig = px.scatter_mapbox(
                                    df_path,
                                    lat="lat",
                                    lon="lon",
                                    hover_data=["id"],
                                    zoom=13,
                                    height=500,
                                    title=f"Route found using {algorithm.upper()}",
                                    center={"lat": df_path["lat"].mean(), "lon": df_path["lon"].mean()},
                                )

                                # Add line for the route
                                fig.add_trace(
                                    go.Scattermapbox(
                                        mode="lines",
                                        lon=df_path["lon"],
                                        lat=df_path["lat"],
                                        line=dict(width=4, color="red"),
                                        name="Route",
                                    )
                                )

                                # Start and end markers
                                fig.add_trace(
                                    go.Scattermapbox(
                                        mode="markers",
                                        lon=[src_node["lon"], dst_node["lon"]],
                                        lat=[src_node["lat"], dst_node["lat"]],
                                        marker=dict(size=15, color=["green", "red"]),
                                        text=["Start", "End"],
                                        name="Start/End",
                                    )
                                )

                                fig.update_layout(
                                    mapbox_style="open-street-map",
                                    margin={"r": 0, "t": 0, "l": 0, "b": 0},
                                )

                                st.plotly_chart(fig, use_container_width=True)

                                with st.expander("Path Details"):
                                    st.dataframe(df_path)
                        else:
                            st.warning("Empty path returned from API.")

                    else:
                        st.error("No route found between the specified points.")
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")

            except requests.exceptions.RequestException as e:
                st.error(f"Request error: {str(e)}")


def benchmark_tab():
    st.header("Algorithm Benchmark")

    num_tests = st.slider("Number of test cases", 5, 100, 20)

    if st.button("Run Benchmark", type="primary"):
        with st.spinner("Running benchmark tests..."):
            try:
                response = requests.post(f"{API_BASE_URL}/benchmark?num_tests={num_tests}", timeout=30)

                if response.status_code == 200:
                    results = response.json()

                    df_results = []
                    for algo, metrics in results.items():
                        df_results.append(
                            {
                                "Algorithm": algo.replace("_", " ").title(),
                                "Avg Runtime (ms)": metrics["avg_runtime_ms"],
                                "Std Runtime (ms)": metrics["std_runtime_ms"],
                                "Avg Nodes Expanded": metrics["avg_nodes_expanded"],
                                "Std Nodes Expanded": metrics["std_nodes_expanded"],
                                "Avg Path Length (m)": metrics["avg_path_length"],
                                "Success Rate (%)": metrics["success_rate"] * 100,
                            }
                        )

                    df = pd.DataFrame(df_results)

                    st.subheader("Benchmark Results")
                    st.dataframe(df, use_container_width=True)

                    col1, col2 = st.columns(2)

                    with col1:
                        fig_runtime = px.bar(
                            df,
                            x="Algorithm",
                            y="Avg Runtime (ms)",
                            error_y="Std Runtime (ms)",
                            title="Average Runtime Comparison",
                            color="Algorithm",
                        )
                        st.plotly_chart(fig_runtime, use_container_width=True)

                        fig_success = px.bar(
                            df,
                            x="Algorithm",
                            y="Success Rate (%)",
                            title="Success Rate",
                            color="Algorithm",
                        )
                        st.plotly_chart(fig_success, use_container_width=True)

                    with col2:
                        fig_nodes = px.bar(
                            df,
                            x="Algorithm",
                            y="Avg Nodes Expanded",
                            error_y="Std Nodes Expanded",
                            title="Average Nodes Expanded",
                            color="Algorithm",
                        )
                        st.plotly_chart(fig_nodes, use_container_width=True)

                        fig_path = px.bar(
                            df,
                            x="Algorithm",
                            y="Avg Path Length (m)",
                            title="Average Path Length",
                            color="Algorithm",
                        )
                        st.plotly_chart(fig_path, use_container_width=True)

                else:
                    st.error(f"Benchmark Error: {response.status_code} - {response.text}")

            except requests.exceptions.RequestException as e:
                st.error(f"Request error: {str(e)}")


def analysis_tab():
    st.header("Algorithm Analysis")

    st.markdown(
        """
    ## Theoretical Complexity Analysis

    ### Time Complexity:
    - **Dijkstra's Algorithm**: O(E log V) - Uses binary heap priority queue
    - **Dial's Algorithm**: O(E + C·V) - Uses bucket queues (C = max weight / bucket size)
    - **A* Search**: O(E log V) - Similar to Dijkstra but with heuristic guidance
    - **Bidirectional A***: ~O(√V) node expansions - Searches from both ends

    ### Space Complexity:
    - **Dijkstra's**: O(V) - Stores distances and previous nodes
    - **Dial's**: O(V + C) - Additional bucket storage
    - **A***: O(V) - Similar to Dijkstra plus heuristic values
    - **Bidirectional A***: O(V) - Maintains two search frontiers

    ## Performance Characteristics:

    ### Dijkstra's Algorithm
    - ✅ **Pros**: Guaranteed optimal solution, well-tested
    - ❌ **Cons**: No heuristic guidance, explores many unnecessary nodes
    - 🎯 **Best for**: Dense graphs, when all shortest paths are needed

    ### Dial's Algorithm
    - ✅ **Pros**: Better performance for small integer weights
    - ❌ **Cons**: Limited to specific weight ranges, more memory usage
    - 🎯 **Best for**: Road networks with distance-based weights

    ### A* Search
    - ✅ **Pros**: Heuristic guidance reduces search space
    - ❌ **Cons**: Requires good heuristic function
    - 🎯 **Best for**: Point-to-point queries with good heuristics

    ### Bidirectional A*
    - ✅ **Pros**: Dramatically reduces search space for long paths
    - ❌ **Cons**: Complex implementation, overhead for short paths
    - 🎯 **Best for**: Long-distance routing in large graphs
    """
    )

    # Theoretical performance visualization
    st.subheader("Theoretical vs Empirical Performance")

    graph_sizes = [100, 500, 1000, 2000, 5000]

    dijkstra_theory = [n * np.log(n) * 0.001 for n in graph_sizes]
    astar_theory = [n * np.log(n) * 0.0005 for n in graph_sizes]  # heuristic helps
    bidirectional_theory = [np.sqrt(n) * 0.01 for n in graph_sizes]

    fig_complexity = go.Figure()

    fig_complexity.add_trace(
        go.Scatter(
            x=graph_sizes,
            y=dijkstra_theory,
            mode="lines+markers",
            name="Dijkstra (theoretical)",
            line=dict(dash="dash"),
        )
    )

    fig_complexity.add_trace(
        go.Scatter(
            x=graph_sizes,
            y=astar_theory,
            mode="lines+markers",
            name="A* (theoretical)",
            line=dict(dash="dash"),
        )
    )

    fig_complexity.add_trace(
        go.Scatter(
            x=graph_sizes,
            y=bidirectional_theory,
            mode="lines+markers",
            name="Bidirectional A* (theoretical)",
            line=dict(dash="dash"),
        )
    )

    fig_complexity.update_layout(
        title="Theoretical Performance Scaling",
        xaxis_title="Graph Size (nodes)",
        yaxis_title="Relative Runtime",
        showlegend=True,
    )

    st.plotly_chart(fig_complexity, use_container_width=True)


if __name__ == "__main__":
    main()
