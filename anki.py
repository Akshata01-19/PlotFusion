import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from io import BytesIO
import base64

# Custom background with gradient and pattern
def set_bg_color():
    st.markdown(
        """ <style>
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            background-attachment: fixed;
        }
        
        .stApp::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
                radial-gradient(circle at 25% 25%, rgba(255,255,255,0.1) 0%, transparent 50%),
                radial-gradient(circle at 75% 75%, rgba(255,255,255,0.1) 0%, transparent 50%);
            z-index: -1;
        }
        
        .stButton>button {
            background: linear-gradient(45deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        .css-1aumxhk {
            background-color: rgba(255,255,255,0.9);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Reduced spacing in markdown sections */
        .stMarkdown p {
            margin-bottom: 0.2rem !important;
            line-height: 1.2 !important;
        }
        
        .stMarkdown div {
            margin-bottom: 0.5rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

st.set_page_config(page_title="PlotFusion", layout="wide", page_icon="📊")
set_bg_color()

# Sample data for download
def get_sample_file(filename):
    base_size = 100  # Base sample size
    np.random.seed(42)  # For reproducibility
    df = pd.DataFrame({
        'Category': np.random.choice(['A','B','C'], base_size),
        'Value': np.random.normal(0,1, base_size)
    })
    return base64.b64encode(df.to_csv(index=False).encode()).decode()

def fig_to_download(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf

def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                return pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.tsv'):
                return pd.read_csv(uploaded_file, sep='\t')
            else:
                return pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None

# -- Home Page --
def show_home():
    st.title("Welcome to PlotFusion - Data Visualization App")

    st.header("📊 About This App")

    st.markdown("""
    <div style="font-size:20px; line-height:1.4;">
      <strong>Data Visualization Made Easy</strong><br>
      This application helps researchers and analysts explore their data through interactive visualizations.<br><br>

      <strong>Key Features:</strong><br>
      - <em>Volcano Plots</em>: Visualize statistical significance vs magnitude of change<br>
      - <em>Bar/Box Plots</em>: Compare categorical data distributions<br>
      - <em>Heatmaps</em>: Explore correlation matrices<br>
      - <em>Scatter/Dot Plots</em>: Examine relationships between variables<br>
      - <em>Ramachandran Plots</em>: Analyze protein backbone dihedral angles<br><br>

      <strong>Technology Stack:</strong><br>
      - Streamlit<br>
      - Matplotlib / Seaborn<br>
      - Plotly
    </div>
    """, unsafe_allow_html=True)

# -- Team Page --
def show_team():
    st.title("👥 Our Team")
    st.subheader("Author")
    st.image(
        "https://media.licdn.com/dms/image/v2/D4E03AQEHejsw-whX6Q/"
        "profile-displayphoto-shrink_200_200/B4EZavA0LyHkAY-/"
        "0/1746692953130?e=1752105600&v=beta&t=eY9gy-4nxSM5-"
        "hsV-DvALVVWDEO30CqBqyTHl0k-ess",
        width=150
    )
    st.markdown("""
    <div style="font-size:16px; line-height:1.2;">
    <strong>Akshata Ranjeet Nachare</strong><br>
    M.Sc. Bioinformatics Student, DES Pune University<br>
    This application is a part of my academic and personal interest in data-driven
    biological analysis. Developed with the aim to simplify and visualize complex
    bioinformatics data, the app reflects both my learning journey and practical
    skills in the field. By integrating both statistical concepts and visualization techniques, this project reflects my growing understanding of the field and my commitment to building tools that enhance scientific discovery. Developing this application has been a meaningful part of my journey in bioinformatics—helping me apply theoretical knowledge in a practical setting, and deepening my appreciation for the intersection of biology, data science, and technology.

<br>
    LinkedIn- https://www.linkedin.com/in/akshata-nachare-740858319
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Mentor")
    st.image(
        "https://media.licdn.com/dms/image/v2/D5603AQF9gsU7YBjWVg/"
        "profile-displayphoto-shrink_400_400/B56ZZI.WrdH0Ag-/"
        "0/1744981029051?e=1752105600&v=beta&t=F4QBDSEgjUvnBS00xPkKqPTLI0jQaMpYefaOzARY1Yg",
        width=150
    )
    st.markdown("""
    <div style="font-size:16px; line-height:1.2;">
    <strong>Dr. Kushagra Kashyap</strong><br>
    Assistant Professor (Bioinformatics), DES Pune University<br>
    Special thanks to Dr. Kushagra Kashyap under whose guidance this project was developed. His invaluable support, encouragement, and mentorship have played a pivotal role in shaping the direction and quality of this application. His insightful teaching has not only guided this work but has also deeply influenced my understanding and passion for the subject.<br>
    LinkedIn- https://www.linkedin.com/in/dr-kushagra-kashyap-b230a3bb
    </div>
    """, unsafe_allow_html=True)
# -- Sample Files Page --
def show_samples():
    st.title("📁 Sample Files")
    st.markdown("""
    <style>
      .stExpander > .streamlit-expanderHeader {
        font-size: 24px !important;
        font-weight: 600 !important;
      }
      [data-testid="stExpander"] summary {
        font-size: 24px !important;
        font-weight: 600 !important;
      }
    </style>
    """, unsafe_allow_html=True)

    samples = [
      ("Volcano Plot", "volcano_sample.csv", "Differential expression data"),
      ("Box/Bar Plots", "box_bar_sample.csv", "Categorical comparisons"),
      ("Scatter Plot", "scatter_sample.csv", "Height/Weight correlation"),
      ("Heatmap", "heatmap_sample.csv", "Gene expression correlations"),
      ("Dot Plot", "dot_sample.csv", "Individual data points visualization"),
      ("Ramachandran", "ramachandran_sample.csv", "Protein dihedral angles"),
      ("Data Processing", "data_processing_sample.csv", "Dataset with missing values")
    ]
    for name, file, desc in samples:
        with st.expander(name):
            st.markdown(desc)
            href = f'<a href="data:file/csv;base64,{get_sample_file(file)}" download="{file}">Download {file}</a>'
            st.markdown(href, unsafe_allow_html=True)

# -- Data Processing Tools --
def show_data_processing_tools():
    st.title("🛠️ Data Processing Tools")
    
    # Custom, larger label for the uploader
    st.markdown(
        "<h3 style='font-size:20px; font-weight:600;'>Upload your data file</h3>",
        unsafe_allow_html=True
    )
    # Hide the built-in label to avoid an empty-label warning
    uploaded_file = st.file_uploader(
        "data_processing_upload", 
        type=["csv", "tsv", "xlsx"], 
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        df = load_data(uploaded_file)
        if df is not None:
            st.success("Data loaded successfully!")
            st.dataframe(df.head())
            
            st.header("Data Cleaning Tools")
            
            # Remove NA values
            if st.checkbox("Remove rows with missing values"):
                df = df.dropna()
                st.success(f"Removed rows with NA values. New shape: {df.shape}")
            
            # Filter data
            st.subheader("Filter Data")
            col1, col2 = st.columns(2)
            with col1:
                filter_col = st.selectbox("Select column to filter", df.columns)
            with col2:
                if pd.api.types.is_numeric_dtype(df[filter_col]):
                    min_val, max_val = st.slider(
                        "Select range",
                        float(df[filter_col].min()),
                        float(df[filter_col].max()),
                        (float(df[filter_col].min()), float(df[filter_col].max()))
                    )
                    df = df[(df[filter_col] >= min_val) & (df[filter_col] <= max_val)]
                else:
                    selected_values = st.multiselect(
                        "Select values to keep",
                        df[filter_col].unique()
                    )
                    if selected_values:
                        df = df[df[filter_col].isin(selected_values)]
            
            # Normalize data
            st.subheader("Normalize Data")
            norm_col = st.selectbox("Select column to normalize", df.select_dtypes(include='number').columns)
            if st.button("Normalize (Z-score)"):
                df[norm_col] = (df[norm_col] - df[norm_col].mean()) / df[norm_col].std()
                st.success(f"Column '{norm_col}' normalized using Z-score")
            
            # Correlation matrix
            st.subheader("Correlation Matrix")
            if st.checkbox("Show correlation matrix"):
                numeric_df = df.select_dtypes(include=['number'])
                if len(numeric_df.columns) > 1:
                    fig, ax = plt.subplots(figsize=(5, 4))  # Reduced size
                    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
                    st.pyplot(fig)
                else:
                    st.warning("Need at least 2 numeric columns for correlation matrix")
            
            # Show processed data
            st.subheader("Processed Data")
            st.dataframe(df.head())
            
            # Download processed data
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="processed_data.csv">Download Processed Data</a>'
            st.markdown(href, unsafe_allow_html=True)

            

# -- Interactive Plot Viewer --
def show_plot_viewer():
    st.title("📊 Interactive Plot Viewer")

    # Instead of empty labels, we collapse them after giving a placeholder
    st.markdown("<h5><b>Upload CSV/Excel for Volcano/Scatter/Dot Plots</b></h5>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "volcano_scatter_dot_upload",
        type=["csv", "xlsx", "tsv"],
        label_visibility="collapsed"
    )

    st.markdown("<h5><b>Upload CSV/TSV for Bar/Box Plots</b></h5>", unsafe_allow_html=True)
    bar_box_file = st.file_uploader(
        "bar_box_upload",
        type=["csv", "tsv"],
        key="bar_box",
        label_visibility="collapsed"
    )

    st.markdown("<h5><b>Upload CSV/TSV for Heatmap</b></h5>", unsafe_allow_html=True)
    heatmap_file = st.file_uploader(
        "heatmap_upload",
        type=["csv", "tsv"],
        key="heatmap",
        label_visibility="collapsed"
    )

    st.markdown("<h5><b>Upload CSV/TSV for Ramachandran Plot</b></h5>", unsafe_allow_html=True)
    ramachandran_file = st.file_uploader(
        "rama_upload",
        type=["csv", "tsv"],
        key="rama",
        label_visibility="collapsed"
    )

    st.markdown("<h4><b>🔍 Choose the plot to display</b></h4>", unsafe_allow_html=True)
    plot_type = st.selectbox(
        "plot_selector",
        ("Volcano Plot", "Bar Plot", "Box Plot", "Heatmap", "Scatter Plot", "Dot Plot", "Ramachandran Plot"),
        label_visibility="collapsed"
    )

    st.markdown("**Note:** For Box and Bar plot, Y-axis should be numeric and for better visualization of plots click on ⬜")
    

    # Volcano Plot
    if plot_type == "Volcano Plot":
        st.subheader("🌋 Volcano Plot")
        if uploaded_file:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

            if 'logFC' in df.columns and 'adj.P.Val' in df.columns:
                df["-log10(adj.P.Val)"] = -np.log10(df["adj.P.Val"])
                
                df["Significance"] = "Not Significant"
                df.loc[(df["logFC"] > 0) & (df["adj.P.Val"] < 0.05), "Significance"] = "Upregulated"
                df.loc[(df["logFC"] < 0) & (df["adj.P.Val"] < 0.05), "Significance"] = "Downregulated"

                colors = df["Significance"].map({"Upregulated": "red", "Downregulated": "blue", "Not Significant": "black"})

                fig, ax = plt.subplots(figsize=(4, 3))  # Reduced size
                scatter = ax.scatter(df["logFC"], df["-log10(adj.P.Val)"], c=colors, s=2)

                ax.set_xlabel("Log2 Fold Change, fontsize=9")
                ax.set_ylabel("-Log10 P-value, fontsize=9")
                ax.set_title("Volcano Plot, fontsize=10")

                # Create legend
                legend_elements = [
                    mpatches.Patch(color="red", label="Upregulated"),
                    mpatches.Patch(color="blue", label="Downregulated"),
                    mpatches.Patch(color="black", label="Non Significant")
                ]
                ax.legend(handles=legend_elements, loc="upper right")
                
                # Display in Streamlit
                st.pyplot(fig)

                # Interpretation
                sig_count = (df["Significance"] != "Not Significant").sum()
                total = len(df)
                st.markdown(f"🧠 *Interpretation*: {sig_count} out of {total} features are statistically significant (p < 0.05).")

            else:
                st.error("Missing 'logFC' or 'adj.P.Val' columns.")
        else:
            st.info("Please upload a file for the volcano plot.")

    # Scatter Plot
    elif plot_type == "Scatter Plot":
        st.subheader("🔵 Scatter Plot")
        df = load_data(uploaded_file)
        if df is not None:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) >= 2:
                x_col = st.selectbox("Select X-axis", numeric_cols)
                y_col = st.selectbox("Select Y-axis", numeric_cols)
                color_col = st.selectbox("Select Color Column (optional)", [None] + df.columns.tolist())
                
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col, 
                               title=f"Scatter Plot: {x_col} vs {y_col}",
                               trendline="ols" if st.checkbox("Show trendline") else None,
                               width=200, height=400)  # Reduced size
                st.plotly_chart(fig, use_container_width=True)
                
                # Improved interpretation
                if x_col and y_col:
                    corr = df[[x_col, y_col]].corr().iloc[0,1]
                    x_mean = df[x_col].mean()
                    y_mean = df[y_col].mean()
                    
                    st.markdown(f"""
                    🧠 *Interpretation*:
                    - *Correlation*: {corr:.2f} ({'positive' if corr > 0 else 'negative'} {'and strong' if abs(corr) > 0.7 else 'and moderate' if abs(corr) > 0.3 else 'but weak'} relationship)
                    - *Average {x_col}*: {x_mean:.2f}
                    - *Average {y_col}*: {y_mean:.2f}
                    - Points clustered in specific regions may indicate relationships
                    - Outliers may represent interesting cases for further investigation
                    """)
            else:
                st.error("Need at least 2 numeric columns for scatter plot")
        else:
            st.info("Please upload a file for the scatter plot")

    # Dot Plot
    elif plot_type == "Dot Plot":
        st.subheader("⚫ Dot Plot")
        df = load_data(uploaded_file)
        if df is not None:
            cols = df.columns.tolist()
            x_col = st.selectbox("Select X-axis", cols)
            y_col = st.selectbox("Select Y-axis", cols)
            
            fig, ax = plt.subplots(figsize=(4, 2.5))  # Reduced size
            ax.plot(df[x_col], df[y_col], 'o', markersize=6)
            ax.set(xlabel=x_col, ylabel=y_col, title=f"Dot Plot: {x_col} vs {y_col}")
            st.pyplot(fig)
            
            # Interpretation with type checking
            if x_col and y_col:
                x_min = f"{df[x_col].min():.2f}" if pd.api.types.is_numeric_dtype(df[x_col]) else str(df[x_col].min())
                x_max = f"{df[x_col].max():.2f}" if pd.api.types.is_numeric_dtype(df[x_col]) else str(df[x_col].max())
                y_min = f"{df[y_col].min():.2f}" if pd.api.types.is_numeric_dtype(df[y_col]) else str(df[y_col].min())
                y_max = f"{df[y_col].max():.2f}" if pd.api.types.is_numeric_dtype(df[y_col]) else str(df[y_col].max())
                
                st.markdown(f"""
                🧠 *Interpretation*:
                - Shows individual data points for {x_col} vs {y_col}
                - *Range of {x_col}*: {x_min} to {x_max}
                - *Range of {y_col}*: {y_min} to {y_max}
                - Reveals distribution patterns and potential outliers
                - Useful for small to medium-sized datasets
                """)
        else:
            st.info("Please upload a file for the dot plot")

    # Ramachandran Plot
    elif plot_type == "Ramachandran Plot":
        st.subheader("🔄 Ramachandran Plot")
        
        if ramachandran_file:
            df = load_data(ramachandran_file)
            if df is not None:
                if 'phi' in df.columns and 'psi' in df.columns:
                    # Create custom colormap
                    cmap = LinearSegmentedColormap.from_list('rama', ['blue', 'cyan', 'green', 'yellow', 'red'])
                    
                    fig, ax = plt.subplots(figsize=(6, 6))  # Reduced size
                    hb = ax.hexbin(df['phi'], df['psi'], 
                                  gridsize=100, 
                                  cmap=cmap, 
                                  bins='log',
                                  mincnt=1,
                                  extent=(-180, 180, -180, 180))
                    
                    # Add reference lines and labels
                    ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
                    ax.axvline(0, color='black', linestyle='--', linewidth=0.5)
                    
                    # Highlight favored and disallowed regions
                    from matplotlib.patches import Polygon
                    
                    # Favored regions (approximate)
                    favored_regions = [
                        [(-180, -100), (-100, -45), (-65, -45), (-65, 50), (-180, 50)],  # Beta-sheet
                        [(-120, 30), (-120, 100), (-30, 100), (-30, 30)]  # Alpha-helix
                    ]
                    
                    # Disallowed regions (approximate)
                    disallowed_regions = [
                        [(50, 180), (50, -180), (180, -180), (180, 180)]  # Right-handed
                    ]
                    
                    for region in favored_regions:
                        poly = Polygon(region, alpha=0.2, color='green', label='Favored')
                        ax.add_patch(poly)
                    
                    for region in disallowed_regions:
                        poly = Polygon(region, alpha=0.2, color='red', label='Disallowed')
                        ax.add_patch(poly)
                    
                    # Set labels and title
                    ax.set_xlabel('Phi angle (φ)', fontsize=12)
                    ax.set_ylabel('Psi angle (ψ)', fontsize=12)
                    ax.set_title('Ramachandran Plot with Favored/Disallowed Regions', fontsize=14)
                    
                    # Set axis limits and ticks
                    ax.set_xlim(-180, 180)
                    ax.set_ylim(-180, 180)
                    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
                    ax.set_yticks([-180, -120, -60, 0, 60, 120, 180])
                    
                    # Add legend
                    from matplotlib.lines import Line2D
                    legend_elements = [
                        Line2D([0], [0], marker='o', color='w', label='Favored Regions',
                              markerfacecolor='green', markersize=10, alpha=0.3),
                        Line2D([0], [0], marker='o', color='w', label='Disallowed Regions',
                              markerfacecolor='red', markersize=10, alpha=0.3)
                    ]
                    ax.legend(handles=legend_elements, loc='upper right')
                    
                    # Add colorbar
                    cb = fig.colorbar(hb, ax=ax)
                    cb.set_label('log10(count)', fontsize=10)
                    
                    st.pyplot(fig)
                    
                    # Interpretation
                    st.markdown("""
                    🧠 *Interpretation*:
                    - *Green regions*: Favored conformations (α-helices, β-sheets)
                    - *Red regions*: Disallowed conformations (steric clashes)
                    - *Dense regions* correspond to common secondary structures:
                      * *Top-left quadrant*: β-sheets
                      * *Bottom-left quadrant*: α-helices
                      * *Bottom-right quadrant*: left-handed helices
                    - *Outliers* (points in red regions) may indicate:
                      * Errors in protein structure determination
                      * Unusual conformations or rare structural motifs
                      * Highlighted in red in 3D view
                    - *Expected coverage*: ~90% of residues should fall in allowed regions
                    """)
                    
                    # Download button for plot
                    st.download_button(
                        label="Download Ramachandran Plot",
                        data=fig_to_download(fig),
                        file_name="ramachandran_plot.png",
                        mime="image/png"
                    )
                    
                else:
                    st.error("Data must contain 'phi' and 'psi' columns")
        else:
            st.info("Please upload a file with phi/psi angles for Ramachandran plot")

    # Bar Plot
    elif plot_type == "Bar Plot":
        st.subheader("📶 Bar Plot")
        df = load_data(bar_box_file)
        if df is not None:
            columns = df.columns.tolist()
            x_col = st.selectbox("Select X-axis", columns)
            y_col = st.selectbox("Select Y-axis", columns)
            
            # Check if y_col is numeric
            if not np.issubdtype(df[y_col].dtype, np.number):
                st.error(f"Selected Y-axis column '{y_col}' must be numeric")
            else:
                fig = px.bar(df, x=x_col, y=y_col, title=f"Bar Plot: {y_col} by {x_col}",
                            width=300, height=400)  # Reduced size
                st.plotly_chart(fig, use_container_width=True)
                
                # Enhanced interpretation with statistics
                max_idx = df[y_col].idxmax()
                min_idx = df[y_col].idxmin()
                
                top = df.loc[max_idx, [x_col, y_col]]
                bottom = df.loc[min_idx, [x_col, y_col]]
                avg_val = df[y_col].mean()
                
                st.markdown(f"""
                🧠 *Interpretation*: 
                - *Highest value*: {top[y_col]:.2f} ({x_col} = {top[x_col]})
                - *Lowest value*: {bottom[y_col]:.2f} ({x_col} = {bottom[x_col]})
                - *Average value*: {avg_val:.2f}
                - *Total observations*: {len(df)}
                """)
        else:
            st.info("Please upload a file for the bar plot.")

    # Box Plot
    elif plot_type == "Box Plot":
        st.subheader("📦 Box Plot")
        df = load_data(bar_box_file)
        if df is not None:
            columns = df.columns.tolist()
            x_col = st.selectbox("Select Category Column (X-axis)", columns)
            y_col = st.selectbox("Select Value Column (Y-axis)", columns)
            
            # Check if y_col is numeric
            if not np.issubdtype(df[y_col].dtype, np.number):
                st.error(f"Selected Y-axis column '{y_col}' must be numeric")
            else:
                fig = px.box(df, x=x_col, y=y_col, title=f"Box Plot: {y_col} by {x_col}",
                            width=300, height=500)  # Reduced size
                st.plotly_chart(fig, use_container_width=True)
                
                group_stats = df.groupby(x_col)[y_col].agg(['median', 'mean', 'std', 'count']).sort_values('median')
                overall_stats = df[y_col].agg(['median', 'mean', 'std'])
                
                st.markdown(f"""
                🧠 *Interpretation*: 
                - *Group with highest median*: {group_stats.index[-1]} ({group_stats['median'].iloc[-1]:.2f})
                - *Group with lowest median*: {group_stats.index[0]} ({group_stats['median'].iloc[0]:.2f})
                - *Overall statistics*:
                  * Median: {overall_stats['median']:.2f}
                  * Mean: {overall_stats['mean']:.2f} ± {overall_stats['std']:.2f}
                - *Total observations*: {len(df)}
                """)
        else:
            st.info("Please upload a file for the box plot.")

    # Heatmap
    elif plot_type == "Heatmap":
        st.subheader("🔥 Heatmap")
        df = load_data(heatmap_file)
        if df is not None:
            numeric_df = df.select_dtypes(include=['number'])
            if len(numeric_df.columns) > 1:
                fig, ax = plt.subplots(figsize=(6, 4))  # Reduced size
                sns.heatmap(numeric_df.corr(), 
                           cmap="coolwarm", 
                           annot=True, 
                           fmt=".2f", 
                           linewidths=0.5, 
                           vmin=-1, 
                           vmax=1,
                           ax=ax)
                ax.set_title("Correlation Heatmap", fontsize=14)
                st.pyplot(fig)
                
                # Find strongest correlations
                corr_matrix = numeric_df.corr().abs()
                np.fill_diagonal(corr_matrix.values, 0)  # Ignore diagonal
                if not corr_matrix.empty:
                    max_corr_pair = corr_matrix.stack().idxmax()
                    max_corr = corr_matrix.loc[max_corr_pair[0], max_corr_pair[1]]
                    
                    st.markdown(f"""
                    🧠 *Interpretation*: 
                    - *Strongest correlation*: Between {max_corr_pair[0]} and {max_corr_pair[1]} ({max_corr:.2f})
                    - Color scale:
                      * Red: Strong positive correlation (1.0)
                      * Blue: Strong negative correlation (-1.0)
                      * White: No correlation (0.0)
                    - *Note*: Only shows linear relationships between variables
                    """)
            else:
                st.error("Need at least 2 numeric columns for heatmap")
        else:
            st.info("Please upload a file for the heatmap.")


# -- About Page --
def show_about():
    st.title("📚 About PlotFusion")
    st.header("📝 User Guide")

    st.subheader("Using the Interactive Plot Viewer")
    st.markdown("""
    <div style="font-size:18px; line-height:1.3;">
    1. Upload your file via the uploader in that tab.<br>
    2. Choose a plot from the "Select plot" dropdown.<br>
    3. Configure axes.<br>
    4. (Optional) Enable features like trendlines.<br>
    5. View your plot.<br>
    6. Download it via right-click or the "Download" button.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Using Data Processing Tools")
    st.markdown("""
    <div style="font-size:18px; line-height:1.3;">
    1. Upload your dataset.<br>
    2. Remove missing values with one click.<br>
    3. Filter rows by numeric range or categorical values.<br>
    4. Normalize any numeric column to a Z-score.<br>
    5. View a Pearson correlation matrix (when ≥ 2 numeric columns).<br>
    6. Download your processed dataset as a CSV.
    </div>
    """, unsafe_allow_html=True)

    st.header("🙏 Acknowledgements")
    st.markdown("""
    <div style="font-size:18px; line-height:1.3;">
    Akshata Ranjeet Nachare (Developer)<br>
    Dr. Kushagra Kashyap (Mentor)
    </div>
    """, unsafe_allow_html=True)
def main():
    # Inject CSS to enlarge tab labels
    st.markdown(""" <style>
    /* Targets the tab buttons in Streamlit's tab bar */
    [data-testid="stHorizontalBlock"] button[role="tab"] {
        font-size: 18px !important;
        font-weight: 600 !important;
    } </style>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏠 Home Page",
        "📚 About",
        "👥 Team",
        "📁 Sample Files",
        "📈 Interactive Plot Viewer",
        "🛠️ Data Processing Tools"
    ])

    with tab1:
        show_home()
    with tab2:
        show_about()
    with tab3:
        show_team()
    with tab4:
        show_samples()
    with tab5:
        show_plot_viewer()
    with tab6:
        show_data_processing_tools()

if __name__ == "__main__":
    main()
