import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="US Tariff Data Visualization",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("US Tariff Data Visualization Dashboard")
st.markdown("Explore tariff rates across countries and product categories based on the 2025 Harmonized Tariff Schedule.")

# Function to preprocess tariff data
def preprocess_tariff_data(file):
    """
    Preprocess the tariff database to create datasets for visualization
    
    Parameters:
    -----------
    file : uploaded file object or file path
    
    Returns:
    --------
    raw_df : pandas.DataFrame
        The original data with added categorizations
    heatmap_df : pandas.DataFrame
        Processed data suitable for heatmap visualization
    category_map : dict
        Mapping between category codes and descriptive names
    """
    # Define mapping of category codes to descriptions
    category_map = {
        "01": "Live Animals",
        "02": "Meat",
        "03": "Fish & Seafood",
        "04": "Dairy Products",
        "07": "Vegetables",
        "08": "Fruits & Nuts",
        "10": "Cereals",
        "16": "Prepared Meat/Fish",
        "22": "Beverages",
        "27": "Mineral Fuels",
        "29": "Organic Chemicals",
        "30": "Pharmaceuticals",
        "39": "Plastics",
        "40": "Rubber",
        "52": "Cotton",
        "61": "Apparel (Knitted)",
        "62": "Apparel (Not Knitted)",
        "64": "Footwear",
        "72": "Iron & Steel",
        "84": "Machinery",
        "85": "Electronics",
        "87": "Vehicles",
        "90": "Optical/Medical",
        "94": "Furniture",
        "95": "Toys & Sports Equipment"
    }
    
    # Load the Excel file
    if isinstance(file, str):
        # File path provided
        raw_df = pd.read_excel(file, sheet_name="trade_tariff_database_2025")
    else:
        # Streamlit uploaded file
        raw_df = pd.read_excel(file, sheet_name="trade_tariff_database_2025")
    
    # Extract the first 2 digits of HTS code to categorize products
    raw_df['category_code'] = raw_df['hts8'].astype(str).str[:2]
    
    # Map category codes to descriptions
    raw_df['category'] = raw_df['category_code'].map(category_map)
    
    # Define countries and their corresponding columns
    countries = [
        {"name": "China", "val_column": "mfn_ad_val_rate"},  # Using MFN for China
        {"name": "EU", "val_column": "mfn_ad_val_rate"},     # Using MFN for EU
        {"name": "Canada", "val_column": "mfn_ad_val_rate", "ind_column": "nafta_canada_ind"},
        {"name": "Mexico", "val_column": "mexico_ad_val_rate", "ind_column": "nafta_mexico_ind"},
        {"name": "Japan", "val_column": "japan_ad_val_rate", "ind_column": "japan_indicator"},
        {"name": "South Korea", "val_column": "korea_ad_val_rate", "ind_column": "korea_indicator"},
        {"name": "Australia", "val_column": "australia_ad_val_rate", "ind_column": "australia_indicator"},
        {"name": "Brazil", "val_column": "mfn_ad_val_rate"}  # Using MFN for Brazil
    ]
    
    # Create a new dataframe for the heatmap
    heatmap_data = []
    
    # Process each selected product category
    for category_code, category_name in category_map.items():
        # Filter products in this category
        products_in_category = raw_df[raw_df['category_code'] == category_code]
        
        if len(products_in_category) > 0:
            country_rates = {"category": category_name}
            
            # Process each country
            for country in countries:
                # Filter out invalid rates or extremely high values
                products_with_rates = products_in_category[
                    products_in_category[country["val_column"]].notna() &
                    (products_in_category[country["val_column"]] <= 1)  # Filter out extremely high values
                ]
                
                if len(products_with_rates) > 0:
                    # Calculate average tariff rate and convert to percentage
                    avg_rate = products_with_rates[country["val_column"]].mean() * 100
                else:
                    avg_rate = 0
                
                country_rates[country["name"]] = round(avg_rate, 2)
            
            heatmap_data.append(country_rates)
    
    # Convert to DataFrame
    heatmap_df = pd.DataFrame(heatmap_data)
    
    # Calculate average tariff across all countries
    heatmap_df['avg_tariff'] = heatmap_df[[c["name"] for c in countries]].mean(axis=1)
    
    # Sort by average tariff rate
    heatmap_df = heatmap_df.sort_values('avg_tariff', ascending=False)
    
    return raw_df, heatmap_df, category_map

# File upload or use sample data
st.sidebar.title("Data Input")
data_option = st.sidebar.radio(
    "Choose a data source:",
    ["Upload your Tariff Database", "Use Sample Dataset (if available)"]
)

# Initialize session state for storing data
if 'raw_df' not in st.session_state:
    st.session_state.raw_df = None
if 'heatmap_df' not in st.session_state:
    st.session_state.heatmap_df = None
if 'category_map' not in st.session_state:
    st.session_state.category_map = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

if data_option == "Upload your Tariff Database":
    uploaded_file = st.sidebar.file_uploader("Upload your tariff database Excel file", type=['xlsx'])
    
    if uploaded_file is not None:
        # Process the file
        with st.spinner("Processing tariff data..."):
            try:
                raw_df, heatmap_df, category_map = preprocess_tariff_data(uploaded_file)
                st.session_state.raw_df = raw_df
                st.session_state.heatmap_df = heatmap_df
                st.session_state.category_map = category_map
                st.session_state.data_loaded = True
                st.sidebar.success("✅ Data processed successfully!")
            except Exception as e:
                st.sidebar.error(f"Error processing file: {e}")
else:
    # Check if sample data file exists in the current directory
    sample_file_path = "tariff_database_2025.xlsx"
    if os.path.exists(sample_file_path):
        if not st.session_state.data_loaded:
            with st.spinner("Loading sample data..."):
                try:
                    raw_df, heatmap_df, category_map = preprocess_tariff_data(sample_file_path)
                    st.session_state.raw_df = raw_df
                    st.session_state.heatmap_df = heatmap_df
                    st.session_state.category_map = category_map
                    st.session_state.data_loaded = True
                    st.sidebar.success("✅ Sample data loaded successfully!")
                except Exception as e:
                    st.sidebar.error(f"Error loading sample data: {e}")
    else:
        st.sidebar.warning("Sample dataset not found. Please upload your own data.")

# Download processed data
if st.session_state.data_loaded:
    st.sidebar.subheader("Download Processed Data")
    
    # Create a CSV for download
    csv = st.session_state.heatmap_df.to_csv(index=False).encode('utf-8')
    
    st.sidebar.download_button(
        label="Download Processed Data as CSV",
        data=csv,
        file_name=f"tariff_heatmap_data_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# Navigation
st.sidebar.title("Navigation")
if st.session_state.data_loaded:
    page = st.sidebar.radio(
        "Select a page:",
        ["Heatmap Visualization", "Tariff Comparisons", "Detailed Analysis", "Raw Data Explorer"]
    )
else:
    page = "Data Input Required"
    st.warning("Please upload a tariff database file or use the sample dataset to continue.")

# Only show the app if data is loaded
if st.session_state.data_loaded:
    raw_df = st.session_state.raw_df
    heatmap_df = st.session_state.heatmap_df
    category_map = st.session_state.category_map
    
    if page == "Heatmap Visualization":
        st.header("Tariff Rate Heatmap by Country and Product Category")
        
        # Filter options
        view_option = st.radio(
            "Select view",
            ["All Categories", "Top 10 by Tariff Rate"]
        )
        
        # Filter data based on selection
        if view_option == "Top 10 by Tariff Rate":
            display_df = heatmap_df.head(10).copy()
        else:
            display_df = heatmap_df.copy()
        
        # Drop the average column for visualization
        display_df = display_df.drop(columns=['avg_tariff'])
        
        # Melt the dataframe for plotly
        melted_df = pd.melt(
            display_df, 
            id_vars=['category'], 
            var_name='country', 
            value_name='tariff_rate'
        )
        
        # Create heatmap with Plotly
        fig = px.imshow(
            display_df.set_index('category'),
            labels=dict(x="Country", y="Product Category", color="Tariff Rate (%)"),
            x=display_df.columns[1:],
            y=display_df['category'],
            color_continuous_scale='Blues',
            aspect="auto",
            title="US Tariff Rates by Country and Product Category (%)"
        )
        
        fig.update_layout(
            height=600,
            width=1000,
            coloraxis_colorbar=dict(title="Tariff Rate (%)"),
        )
        
        # Add text annotations
        for i, category in enumerate(display_df['category']):
            for j, country in enumerate(display_df.columns[1:]):
                value = display_df.iloc[i][country]
                fig.add_annotation(
                    x=country,
                    y=category,
                    text=f"{value:.1f}%",
                    showarrow=False,
                    font=dict(
                        color="white" if value > 7 else "black",
                        size=10
                    )
                )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### Key Insights:
        - **Trade Agreements Impact**: Mexico shows 0% tariffs due to USMCA, while countries without trade agreements face higher tariffs.
        - **Protected Sectors**: Footwear, apparel, and textiles have the highest tariff protection.
        - **MFN Rates**: China, EU, and Brazil typically face the same tariff levels, reflecting their MFN status.
        """)
    
    elif page == "Tariff Comparisons":
        st.header("Country Tariff Comparisons")
        
        # Select countries to compare
        selected_countries = st.multiselect(
            "Select countries to compare:",
            options=heatmap_df.columns[1:-1],
            default=["China", "Mexico", "Japan", "EU"]
        )
        
        if not selected_countries:
            st.warning("Please select at least one country to compare.")
        else:
            # Prepare data for bar chart
            compare_df = heatmap_df[['category', 'avg_tariff'] + selected_countries].head(10)
            
            # Create bar chart
            fig = px.bar(
                compare_df,
                x='category',
                y=selected_countries,
                title=f"Top 10 Categories by Tariff Rate - Country Comparison",
                labels={"value": "Tariff Rate (%)", "category": "Product Category", "variable": "Country"},
                barmode='group',
                height=600
            )
            
            fig.update_layout(xaxis={'categoryorder':'total descending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Create radar chart for overall tariff profile
            categories = compare_df['category'].tolist()
            
            fig = go.Figure()
            
            for country in selected_countries:
                fig.add_trace(go.Scatterpolar(
                    r=compare_df[country].tolist(),
                    theta=categories,
                    fill='toself',
                    name=country
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(compare_df[selected_countries].max()) + 2]
                    )
                ),
                title="Tariff Profile Comparison (Radar Chart)",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Detailed Analysis":
        st.header("Detailed Tariff Analysis")
        
        # Category selection for detailed analysis
        selected_category = st.selectbox(
            "Select a product category to analyze:",
            options=heatmap_df['category'].tolist()
        )
        
        # Get the category code
        category_code = [code for code, name in category_map.items() if name == selected_category][0]
        
        # Filter raw data for the selected category
        category_data = raw_df[raw_df['category_code'] == category_code]
        
        # Display basic stats
        st.subheader(f"Statistics for {selected_category} (HTS Code: {category_code})")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Number of Tariff Lines", len(category_data))
        col2.metric("Avg MFN Rate", f"{category_data['mfn_ad_val_rate'].mean()*100:.2f}%")
        col3.metric("Max MFN Rate", f"{category_data['mfn_ad_val_rate'].max()*100:.2f}%")
        
        # Distribution of MFN rates
        st.subheader("Distribution of MFN Tariff Rates")
        
        # Filter out extreme values for better visualization
        filtered_data = category_data[category_data['mfn_ad_val_rate'] <= 1]
        
        fig = px.histogram(
            filtered_data, 
            x='mfn_ad_val_rate',
            nbins=20,
            labels={"mfn_ad_val_rate": "MFN Tariff Rate"},
            title=f"Distribution of MFN Tariff Rates for {selected_category}"
        )
        
        fig.update_layout(
            xaxis_title="MFN Tariff Rate",
            yaxis_title="Count",
            xaxis=dict(tickformat=".0%")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Sample products from this category
        st.subheader("Sample Products in this Category")
        sample_products = category_data[['hts8', 'brief_description', 'mfn_ad_val_rate']].sample(min(10, len(category_data)))
        sample_products['mfn_ad_val_rate'] = sample_products['mfn_ad_val_rate'] * 100
        sample_products.columns = ['HTS Code', 'Description', 'MFN Rate (%)']
        st.dataframe(sample_products, use_container_width=True)
    
    else:  # Raw Data Explorer
        st.header("Raw Data Explorer")
        
        # Search functionality
        search_term = st.text_input("Search by product description:")
        
        if search_term:
            filtered_df = raw_df[raw_df['brief_description'].str.contains(search_term, case=False, na=False)]
            st.write(f"Found {len(filtered_df)} items matching '{search_term}'")
            
            if not filtered_df.empty:
                # Display search results
                results_df = filtered_df[['hts8', 'brief_description', 'category', 'mfn_ad_val_rate']]
                results_df['mfn_ad_val_rate'] = results_df['mfn_ad_val_rate'] * 100
                results_df.columns = ['HTS Code', 'Description', 'Category', 'MFN Rate (%)']
                st.dataframe(results_df, use_container_width=True)
            else:
                st.info("No results found. Try a different search term.")
        
        # Display data summary
        st.subheader("Data Summary")
        
        # Category distribution
        category_counts = raw_df['category'].value_counts().reset_index()
        category_counts.columns = ['Category', 'Count']
        
        fig = px.bar(
            category_counts, 
            x='Category', 
            y='Count',
            title="Number of Tariff Lines by Product Category",
            height=500
        )
        
        fig.update_layout(xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data statistics
        st.subheader("Tariff Database Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Tariff Lines", len(raw_df))
        col2.metric("Categories", len(raw_df['category'].dropna().unique()))
        col3.metric("Avg. MFN Rate", f"{raw_df['mfn_ad_val_rate'].mean()*100:.2f}%")
        col4.metric("Non-Zero Rates", f"{(raw_df['mfn_ad_val_rate'] > 0).sum()}")

# Footer
st.markdown("---")
st.markdown("Data source: 2025 Harmonized Tariff Schedule database")
st.markdown("Created with Streamlit, Pandas, and Plotly")
