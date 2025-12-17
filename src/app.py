
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import sys
# Add project root to path for Streamlit Cloud
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets.cifar10 import load_cifar10
from src.shifts.corruptions import SHIFTS, denormalize
from src.models.build_model import build_model
from src.explainers.gradcam import GradCAMExplainer
from src.explainers.integrated_gradients import IGExplainer
from src.explainers.utils import normalize_attribution
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="XAI Robustness Benchmark")

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    h1 {
        color: #1f77b4;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f3ff;
        color: #1f77b4;
        font-weight: bold;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    /* Global Typography */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif !important;
        color: #1e293b !important;
    }
    
    /* Metrics Cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 15px 25px;
        border-radius: 12px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
    }
    div[data-testid="stMetricValue"] {
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Prediction Box Styling */
    .pred-box {
        padding: 10px;
        min-height: 80px;
        border-radius: 8px;
        background-color: #f8fafc;
        color: #334155;
        border-left: 4px solid #6366f1;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        margin-top: 5px;
        margin-bottom: 10px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }
    
    /* Remove top padding */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 2rem;
    }
    
    /* Custom Button Styling */
    div.stButton > button {
        background-color: #ffffff;
        color: #4f46e5;
        font-weight: 600;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        padding: 0.5rem 1rem;
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        border-color: #6366f1;
        color: #4338ca;
        background-color: #f8fafc;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ XAI Robustness Benchmark")
st.markdown("### Analyzing Explanation Stability Under Distribution Shift")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    run_dir = st.text_input("Run Directory", "./outputs/runs/run_002")
    ckpt_path = st.text_input("Model Checkpoint", "./outputs/checkpoints/resnet18_cifar10_baseline.pth")
    
    if not os.path.exists(run_dir):
        st.error(f"Run Directory not found.")
        st.stop()
        
    st.info(f"Run: {os.path.basename(run_dir)}")

# Load Resources
@st.cache_resource
def load_resources(checkpoint_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model('resnet18', num_classes=10)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        st.warning(f"Checkpoint not found at {checkpoint_path}. Using random weights.")
    
    model.to(device)
    model.eval()
    
    # Explainers
    gradcam = GradCAMExplainer(model, target_layer=model.layer4[1].conv2)
    return model, gradcam, device

model, gradcam, device = load_resources(ckpt_path)

csv_path = os.path.join(run_dir, "metrics.csv")
demo_csv_path = os.path.join("demo_data", "benchmark_results.csv")

if not os.path.exists(csv_path):
    if os.path.exists(demo_csv_path):
        st.info("⚠️ Local run not found. Switching to **Demo Mode** using pre-computed benchmark results.")
        csv_path = demo_csv_path
    else:
        st.error("metrics.csv not found and no demo data available.")
        st.stop()

df = pd.read_csv(csv_path)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📉 Shift Explorer", "👁️ Example Viewer", "🧪 Live Lab"])

with tab1:
    # Kpi row
    # Calculate metrics first
    consistent_df = df[df['is_consistent'] == True]
    robustness = consistent_df.groupby(['explainer', 'shift'])['ssim'].mean().unstack()
    
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    avg_acc = (df['shift_pred'] == df['true_label']).mean() * 100
    avg_ssim = df[df['is_consistent'] == True]['ssim'].mean()
    
    col_kpi1.metric("Overall Accuracy", f"{avg_acc:.1f}%")
    col_kpi2.metric("Overall Stability (SSIM)", f"{avg_ssim:.3f}")
    col_kpi3.metric("Samples Evaluated", len(df) // len(df['shift'].unique()) // len(df['explainer'].unique()) // 5)
    
    # Side-by-Side Layout
    col_chart, col_table = st.columns(2)
    
    with col_chart:
        st.subheader("Accuracy vs Stability")
        shifts = df['shift'].unique()
        selected_shift_ov = st.selectbox("Select Corruption Type", shifts)
        
        # Plot Logic
        shift_df = df[df['shift'] == selected_shift_ov]
        explainers = df['explainer'].unique()
        
        fig = go.Figure()
        # High Contrast Palette for Distinct Visibility
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        # 1. Plot Accuracy Once
        exp_df_0 = shift_df[shift_df['explainer'] == explainers[0]]
        acc_series_0 = exp_df_0.groupby('severity').apply(lambda x: (x['shift_pred'] == x['true_label']).mean())
        grouped_0 = exp_df_0.groupby('severity').agg({'ssim': 'mean'}).reset_index()
        
        fig.add_trace(go.Scatter(
            x=grouped_0['severity'],
            y=acc_series_0,
            mode='lines+markers',
            name='Model Accuracy',
            line=dict(color='#333333', dash='dash', width=2),
            hovertemplate='<b>Model Accuracy</b><br>Severity: %{x}<br>Acc: %{y:.2%}<extra></extra>'
        ))
        
        # 2. Plot SSIM
        for i, explainer in enumerate(explainers):
            exp_df = shift_df[shift_df['explainer'] == explainer]
            grouped = exp_df.groupby('severity').agg({'ssim': 'mean'}).reset_index()
            color = colors[i % len(colors)]
            
            fig.add_trace(go.Scatter(
                x=grouped['severity'],
                y=grouped['ssim'],
                mode='lines+markers',
                name=f'Stability ({explainer})',
                line=dict(color=color, width=3),
                hovertemplate=f'<b>{explainer}</b><br>Severity: %{{x}}<br>SSIM: %{{y:.3f}}<extra></extra>'
            ))
            
        fig.update_layout(
            title=dict(text=f"Impact of {selected_shift_ov}", x=0.5, xanchor='center', font=dict(color='#334155', size=24)),
            xaxis_title="Severity Level",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1.05], gridcolor='#f1f5f9'),
            xaxis=dict(tickmode='linear', tick0=1, dtick=1, gridcolor='#f1f5f9'),
            hovermode="x unified",
            legend=dict(orientation="h", y=-0.25, x=0.5, xanchor='center', font=dict(color='#334155', size=16)),
            template="plotly_white",
            height=450,
            margin=dict(l=20, r=20, t=60, b=60)
        )
        
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.subheader("Robustness Rankings")
        st.caption("Avg SSIM (Higher is better)")
        st.dataframe(
            robustness.style.background_gradient(cmap='Purples', axis=None).format("{:.3f}"), 
            use_container_width=True,
            height=400
        )

with tab2:
    st.header("Deep Dive: Metric Distributions")
    
    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown("#### Filter")
        sel_shift = st.selectbox("Shift", shifts, key='shift_ex')
        sel_metric = st.selectbox("Metric", ['ssim', 'cosine', 'topk', 'drift'])
        
    with col2:
        subset = df[df['shift'] == sel_shift]
        # Standard matplotlib style
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=subset, x='severity', y=sel_metric, hue='explainer', ax=ax2, palette="Purples")
        ax2.set_title(f"Distribution of {sel_metric} for {sel_shift}", color='#334155')
        
        # Clean up matplot
        for spine in ax2.spines.values():
            spine.set_edgecolor('#e2e8f0')
        ax2.tick_params(colors='#64748b')
        ax2.xaxis.label.set_color('#64748b')
        ax2.yaxis.label.set_color('#64748b')
        
        st.pyplot(fig2)

with tab3:
    st.header("Qualitative Analysis")
    st.info("Visual comparison of explanations for clean vs shifted inputs (From Pre-computed Runs).")
    
    fig_dir = os.path.join(run_dir, "figures/qualitative_panels")
    if os.path.exists(fig_dir):
        files = [f for f in os.listdir(fig_dir) if f.endswith('.png')]
        if files:
            selected_file = st.selectbox("Select Example", files)
            img = Image.open(os.path.join(fig_dir, selected_file))
            st.image(img, caption=selected_file, use_container_width=True)
        else:
            st.warning("No images found in figures/qualitative_panels")
    else:
        st.warning("figures/qualitative_panels directory does not exist.")

with tab4:
    st.header("🧪 Live Lab: Model & Explanation Playground")
    st.markdown("Run the model and Grad-CAM **live** on data to see how drift breaks confidence and reasoning.")
    
    data_root = "./data"
    
    # --- Top Row: Controls (Side-by-Side) ---
    c_ctrl1, c_ctrl2 = st.columns(2)
    
    with c_ctrl1:
        st.subheader("1. Input Data")
        if st.button("🎲 Load Random Sample", use_container_width=True):
            if not os.path.exists(os.path.join(data_root, 'cifar-10-batches-py')):
                 st.error("CIFAR-10 data not found.")
            else:
                _, testloader, classes = load_cifar10(data_root, batch_size=64, num_workers=0, download=False)
                data_iter = iter(testloader)
                images, labels = next(data_iter)
                idx = np.random.randint(0, len(images))
                st.session_state['sample_img'] = images[idx]
                st.session_state['sample_label'] = labels[idx].item()
                st.session_state['classes'] = classes

    with c_ctrl2:
        st.subheader("2. Shift Configuration")
        c_s1, c_s2 = st.columns(2)
        with c_s1:
            shift_name = st.selectbox("Shift Type", list(SHIFTS.keys()), key='live_shift')
        with c_s2:
            severity = st.slider("Severity", 1, 5, 3, key='live_sev')
            
    st.divider()

    # --- Bottom Row: Results (Horizontal Stack) ---
    if 'sample_img' in st.session_state:
        img_tensor = st.session_state['sample_img']
        label_idx = st.session_state['sample_label']
        classes = st.session_state['classes']
        true_class_name = classes[label_idx]
        
        # Apply Shift (CPU)
        shift_obj = SHIFTS[shift_name]
        shifted_tensor = shift_obj.apply(img_tensor.clone(), severity)
        
        # Inference (GPU)
        with torch.no_grad():
            # Prepare batch
            input_batch = torch.stack([img_tensor, shifted_tensor]).to(device)
            outputs = model(input_batch)
            probs = F.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)
            
            clean_prob = probs[0, preds[0]].item()
            shift_prob = probs[1, preds[1]].item()
            
            clean_pred_name = classes[preds[0]]
            shift_pred_name = classes[preds[1]]
            
        # Explanation (GradCAM)
        clean_attr = gradcam.explain(input_batch[0].unsqueeze(0), target_class=preds[0].item())
        clean_attr = normalize_attribution(clean_attr)
        
        shift_attr = gradcam.explain(input_batch[1].unsqueeze(0), target_class=preds[1].item())
        shift_attr = normalize_attribution(shift_attr)
        
        # Display - 4 Columns Layout
        # Display Images
        # Helper for displaying
        def get_display_img(t):
            d = denormalize(t.unsqueeze(0).cpu()).squeeze(0).permute(1,2,0).numpy()
            return np.clip(d, 0, 1)
        
        clean_disp = get_display_img(img_tensor)
        shift_disp = get_display_img(shifted_tensor)
        
        c_res1, c_res2, c_res3, c_res4 = st.columns(4, gap="medium")
        
        # -- Col 1: Original Image --
        with c_res1:
            st.markdown(f"**Original**")
            st.image(clean_disp, use_container_width=True)
            # Pred Box
            st.markdown(f"""
            <div class="pred-box" style="font-size: 0.8rem; padding: 5px;">
                <strong>Pred:</strong> {clean_pred_name}<br>
                <strong>Conf:</strong> {clean_prob*100:.1f}%
            </div>
            """, unsafe_allow_html=True)

        # -- Col 2: Original Heatmap --
        with c_res2:
            st.markdown("**Explanation**")
            fig_hm, ax_hm = plt.subplots(figsize=(3, 3)) # Small size
            ax_hm.imshow(clean_disp)
            ax_hm.imshow(clean_attr.squeeze(), cmap='jet', alpha=0.5)
            ax_hm.axis('off')
            st.pyplot(fig_hm, use_container_width=True)

        # -- Col 3: Corrupted Image --
        with c_res3:
            st.markdown(f"**Corrupted**")
            st.image(shift_disp, use_container_width=True)
            # Pred Box
            color = "#d4edda" if clean_pred_name == shift_pred_name else "#f8d7da"
            st.markdown(f"""
            <div class="pred-box" style="background-color: {color}; border-left-color: {'#28a745' if clean_pred_name == shift_pred_name else '#dc3545'}; font-size: 0.8rem; padding: 5px;">
                <strong>Pred:</strong> {shift_pred_name}<br>
                <strong>Conf:</strong> {shift_prob*100:.1f}%
            </div>
            """, unsafe_allow_html=True)

        # -- Col 4: Corrupted Heatmap --
        with c_res4:
            st.markdown("**Explanation**")
            fig_hm2, ax_hm2 = plt.subplots(figsize=(3, 3)) # Small size
            ax_hm2.imshow(shift_disp)
            ax_hm2.imshow(shift_attr.squeeze(), cmap='jet', alpha=0.5)
            ax_hm2.axis('off')
            st.pyplot(fig_hm2, use_container_width=True)

    else:
        st.info("👈 Click 'Load Random Sample' to start the live analysis.")
