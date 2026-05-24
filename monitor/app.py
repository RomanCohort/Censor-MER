"""
Censor Training Monitor
=====================
Streamlit dashboard for real-time training monitoring.

Features:
    - Training metrics (Loss, Accuracy, F1)
    - Loss curves with Altair
    - GPU monitoring (memory, utilization, temperature)
    - Sample predictions visualization
    - Training control (start/pause/stop)

Usage:
    streamlit run monitor/app.py

    # Or run with custom port:
    streamlit run monitor/app.py --server.port 8502
"""

import os
import sys
import time
import json
import glob
import threading
import tempfile
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import torch

# Try to import pynvml
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


# =============================================================================
# Session State
# =============================================================================

if 'training_status' not in st.session_state:
    st.session_state.training_status = 'stopped'  # stopped, running, paused

if 'current_epoch' not in st.session_state:
    st.session_state.current_epoch = 0

if 'total_epochs' not in st.session_state:
    st.session_state.total_epochs = 50

if 'metrics_history' not in st.session_state:
    st.session_state.metrics_history = []

if 'control_lock' not in st.session_state:
    st.session_state.control_lock = threading.Lock()


# =============================================================================
# GPU Monitoring
# =============================================================================

class GPUMonitor:
    """Monitor GPU stats."""

    def __init__(self):
        if not PYNVML_AVAILABLE:
            self.available = False
            return

        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            self.handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i)
                for i in range(self.device_count)
            ]
            self.available = True
        except Exception as e:
            print(f"[WARN] NVML init failed: {e}")
            self.available = False

    def get_stats(self, device_id=None):
        """Get GPU stats for specified device."""
        if not self.available:
            return {}

        stats = {}
        devices = [device_id] if device_id is not None else range(self.device_count)

        for i in devices:
            try:
                handle = self.handles[i]
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                clock_sm = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                clock_mem = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)

                stats[i] = {
                    'name': pynvml.nvmlDeviceGetName(handle).decode(),
                    'memory_used_mb': mem_info.used / 1024**2,
                    'memory_total_mb': mem_info.total / 1024**2,
                    'memory_used_pct': 100 * mem_info.used / max(mem_info.total, 1),
                    'utilization_pct': util.gpu,
                    'memory_utilization_pct': util.memory,
                    'temperature_c': temp,
                    'clock_sm_mhz': clock_sm,
                    'clock_mem_mhz': clock_mem,
                }
            except Exception as e:
                stats[i] = {'error': str(e)}

        return stats

    def __del__(self):
        if self.available:
            pynvml.nvmlShutdown()


# =============================================================================
# Metrics Reader
# =============================================================================

class MetricsReader:
    """Read training metrics from CSV file."""

    def __init__(self, csv_path='./checkpoints/metrics.csv'):
        self.csv_path = csv_path
        self.last_mtime = 0

    def read(self):
        """Read metrics from CSV."""
        if not os.path.exists(self.csv_path):
            return pd.DataFrame()

        try:
            # Check if file changed
            mtime = os.path.getmtime(self.csv_path)
            if mtime == self.last_mtime:
                return None  # No change

            self.last_mtime = mtime
            df = pd.read_csv(self.csv_path)
            return df
        except Exception as e:
            print(f"[WARN] Failed to read metrics: {e}")
            return None

    def get_latest(self):
        """Get latest metrics row."""
        df = self.read()
        if df is None or len(df) == 0:
            return {}
        return df.iloc[-1].to_dict()


# =============================================================================
# Sidebar - Configuration
# =============================================================================

def render_sidebar():
    """Render sidebar configuration."""
    st.sidebar.title("⚙️ Configuration")

    # Paths
    st.sidebar.subheader("Paths")
    checkpoint_dir = st.sidebar.text_input(
        "Checkpoint Directory",
        value="./checkpoints"
    )
    csv_path = os.path.join(checkpoint_dir, "metrics.csv")

    # Training settings
    st.sidebar.subheader("Training")
    total_epochs = st.sidebar.number_input(
        "Total Epochs",
        min_value=1,
        value=50
    )
    batch_size = st.sidebar.number_input(
        "Batch Size",
        min_value=1,
        value=2
    )

    # GPU settings
    st.sidebar.subheader("GPU")
    gpu_id = st.sidebar.selectbox(
        "GPU Device",
        options=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [0],
        index=0 if torch.cuda.is_available() else 0
    )

    # Display settings
    st.sidebar.subheader("Display")
    refresh_rate = st.sidebar.slider(
        "Refresh Rate (seconds)",
        min_value=1,
        max_value=30,
        value=5
    )

    # Update session state
    st.session_state.csv_path = csv_path
    st.session_state.total_epochs = total_epochs
    st.session_state.batch_size = batch_size
    st.session_state.gpu_id = gpu_id
    st.session_state.refresh_rate = refresh_rate

    return {
        'checkpoint_dir': checkpoint_dir,
        'csv_path': csv_path,
        'total_epochs': total_epochs,
        'batch_size': batch_size,
        'gpu_id': gpu_id,
        'refresh_rate': refresh_rate,
    }


# =============================================================================
# Main Panel - Training Status
# =============================================================================

def render_status_panel(config):
    """Render training status panel."""
    st.title("🎯 Censor Training Monitor")

    # Get current metrics
    reader = MetricsReader(config['csv_path'])
    current_metrics = reader.get_latest()

    # Update session
    if current_metrics:
        epoch = current_metrics.get('epoch', 0)
    else:
        epoch = st.session_state.current_epoch

    st.session_state.current_epoch = epoch

    # Status row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Status",
            st.session_state.training_status.upper(),
            delta=None
        )

    with col2:
        st.metric(
            "Epoch",
            f"{epoch}/{config['total_epochs']}",
            delta=None
        )

    with col3:
        loss = current_metrics.get('train_loss_total', 0.0) if current_metrics else 0.0
        st.metric(
            "Total Loss",
            f"{loss:.4f}",
            delta=None
        )

    with col4:
        acc = current_metrics.get('train_me_acc', 0.0) if current_metrics else 0.0
        st.metric(
            "ME Accuracy",
            f"{acc:.4f}",
            delta=None
        )

    with col5:
        f1 = current_metrics.get('train_me_f1', 0.0) if current_metrics else 0.0
        st.metric(
            "ME F1",
            f"{f1:.4f}",
            delta=None
        )

    st.divider()


# =============================================================================
# Loss Curves Panel
# =============================================================================

def render_loss_curves(config):
    """Render loss curves."""
    st.subheader("📈 Loss Curves")

    # Read data
    reader = MetricsReader(config['csv_path'])
    df = reader.read()

    if df is None or len(df) == 0:
        st.info("No training data yet. Start training to see curves.")
        return

    # Create chart
    base = alt.Chart(df).mark_line(point=True).encode(
        tooltip=['epoch', alt.Tooltip('loss_me', format='.4f')]
    )

    # Loss components
    cols = st.columns(2)

    with cols[0]:
        chart_me = base.encode(
            x=alt.X('epoch', title='Epoch'),
            y=alt.Y('train_loss_me', title='ME Loss')
        ).properties(
            title='Micro-Expression Loss'
        )
        st.altair_chart(chart_me, use_container_width=True)

    with cols[1]:
        chart_au = base.encode(
            x=alt.X('epoch', title='Epoch'),
            y=alt.Y('train_loss_au', title='AU Loss')
        ).properties(
            title='Action Unit Loss'
        )
        st.altair_chart(chart_au, use_container_width=True)

    # Additional charts
    cols2 = st.columns(2)

    with cols2[0]:
        chart_moe = base.encode(
            x=alt.X('epoch', title='Epoch'),
            y=alt.Y('train_loss_moe', title='MoE Loss')
        ).properties(
            title='MoE Auxiliary Loss'
        )
        st.altair_chart(chart_moe, use_container_width=True)

    with cols2[1]:
        chart_total = base.encode(
            x=alt.X('epoch', title='Epoch'),
            y=alt.Y('train_loss_total', title='Total Loss')
        ).properties(
            title='Total Loss'
        )
        st.altair_chart(chart_total, use_container_width=True)


# =============================================================================
# GPU Monitoring Panel
# =============================================================================

def render_gpu_panel(config):
    """Render GPU monitoring panel."""
    st.subheader("🔥 GPU Monitoring")

    gpu_monitor = GPUMonitor()

    if not gpu_monitor.available:
        st.warning("GPU monitoring not available. Install nvidia-ml-py3:")
        st.code("pip install nvidia-ml-py3")
        return

    stats = gpu_monitor.get_stats()

    if not stats:
        st.warning("No GPU detected.")
        return

    # Display GPU cards
    for gpu_id, gpu_stats in stats.items():
        with st.expander(f"GPU {gpu_id}: {gpu_stats.get('name', 'Unknown')}", expanded=True):
            cols = st.columns(4)

            with cols[0]:
                mem_used = gpu_stats.get('memory_used_mb', 0)
                mem_total = gpu_stats.get('memory_total_mb', 0)
                mem_pct = gpu_stats.get('memory_used_pct', 0)
                st.metric(
                    "Memory",
                    f"{mem_used:.0f}MB / {mem_total:.0f}MB",
                    delta=f"{mem_pct:.1f}%"
                )

            with cols[1]:
                util = gpu_stats.get('utilization_pct', 0)
                st.metric(
                    "GPU Utilization",
                    f"{util}%",
                    delta=None
                )

            with cols[2]:
                mem_util = gpu_stats.get('memory_utilization_pct', 0)
                st.metric(
                    "Memory Utilization",
                    f"{mem_util}%",
                    delta=None
                )

            with cols[3]:
                temp = gpu_stats.get('temperature_c', 0)
                st.metric(
                    "Temperature",
                    f"{temp}°C",
                    delta=None
                )

            # Progress bars
            st.progress(mem_pct / 100, text="Memory Usage")
            st.progress(util / 100, text="GPU Utilization")


# =============================================================================
# Training Control Panel
# =============================================================================

def render_control_panel(config):
    """Render training control panel."""
    st.subheader("🎮 Training Control")

    cols = st.columns(3)

    with cols[0]:
        if st.session_state.training_status != 'running':
            if st.button("▶️ Start Training", use_container_width=True):
                st.session_state.training_status = 'running'
                st.rerun()

    with cols[1]:
        if st.session_state.training_status == 'running':
            if st.button("⏸️ Pause", use_container_width=True):
                st.session_state.training_status = 'paused'
                st.rerun()

    with cols[2]:
        if st.button("⏹️ Stop", use_container_width=True):
            st.session_state.training_status = 'stopped'
            st.rerun()

    # Quick actions
    st.subheader("⚡ Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📂 Open Checkpoints", use_container_width=True):
            checkpoint_dir = config['checkpoint_dir']
            if os.path.exists(checkpoint_dir):
                os.startfile(checkpoint_dir)

    with col2]:
        if st.button("📊 Open Metrics", use_container_width=True):
            csv_path = config['csv_path']
            if os.path.exists(csv_path):
                os.startfile(csv_path)

    with col3]:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()


# =============================================================================
# Validation Metrics Panel
# =============================================================================

def render_validation_panel(config):
    """Render validation metrics panel."""
    st.subheader("📋 Validation Metrics")

    reader = MetricsReader(config['csv_path'])
    df = reader.read()

    if df is None or len(df) == 0:
        st.info("No validation data yet.")
        return

    # Filter validation columns
    val_cols = [c for c in df.columns if c.startswith('val_')]

    if not val_cols:
        st.info("No validation data yet.")
        return

    # Display validation metrics
    latest = df.iloc[-1]

    cols = st.columns(4)

    with cols[0]:
        val_acc = latest.get('val_me_acc', 0)
        st.metric("Validation Accuracy", f"{val_acc:.4f}")

    with cols[1]:
        val_f1 = latest.get('val_me_f1', 0)
        st.metric("Validation F1", f"{val_f1:.4f}")

    with cols[2]:
        val_au_f1 = latest.get('val_au_f1', 0)
        st.metric("AU F1", f"{val_au_f1:.4f}")

    with cols[3]:
        val_ham = latest.get('val_au_hamming', 0)
        st.metric("AU Hamming", f"{val_ham:.4f}")

    # Validation curves
    base = alt.Chart(df).mark_line(point=True)

    cols = st.columns(2)

    with cols[0]:
        chart = base.encode(
            x='epoch',
            y=alt.Y('val_me_acc', title='Validation Accuracy')
        ).properties(title='Validation Accuracy')
        st.altair_chart(chart, use_container_width=True)

    with cols[1]:
        chart = base.encode(
            x='epoch',
            y=alt.Y('val_me_f1', title='Validation F1')
        ).properties(title='Validation F1')
        st.altair_chart(chart, use_container_width=True)


# =============================================================================
# Model Info Panel
# =============================================================================

def render_model_info():
    """Render model information panel."""
    st.subheader("🤖 Model Information")

    try:
        from main import Censor
        model = Censor()

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        cols = st.columns(2)

        with cols[0]:
            st.metric("Total Parameters", f"{total_params:,}")

        with cols[1]:
            st.metric("Trainable Parameters", f"{trainable_params:,}")

    except Exception as e:
        st.warning(f"Could not load model info: {e}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Main application."""

    # Render sidebar
    config = render_sidebar()

    # Auto-refresh
    if st.session_state.training_status == 'running':
        time.sleep(config['refresh_rate'])

    # Render main panels
    render_status_panel(config)

    # Tab layout
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Loss Curves",
        "🔥 GPU",
        "🎮 Control",
        "🤖 Model"
    ])

    with tab1:
        render_loss_curves(config)

    with tab2:
        render_gpu_panel(config)

    with tab3:
        render_control_panel(config)

    with tab4:
        render_model_info()

    # Footer
    st.markdown("---")
    st.caption(
        f"Censor Training Monitor | "
        f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


if __name__ == '__main__':
    main()