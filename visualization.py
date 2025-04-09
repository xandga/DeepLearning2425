# import all packages needed
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



def plot_family_distribution_by_phylum(data, phylum_column='phylum', family_column='family', 
                                       selected_phylum=None, top_n=20, figsize=(6, 4)):
    '''
    Plot the distribution of families within each phylum. 
    For a specific phylum (selected_phylum), show top N families, otherwise, show all families.
    
    Parameters:
    data (DataFrame): The DataFrame containing the data to plot.
    phylum_column (str, optional): The column for the phylum. Default is 'phylum'.
    family_column (str, optional): The column for the family. Default is 'family'.
    selected_phylum (str, optional): The phylum for which to display top N families. Default is None (show all).
    top_n (int, optional): Number of top families to display for the selected phylum. Default is 20.
    figsize (tuple, optional): The size of the figure (width, height). Default is (12, 8).
    
    Returns:
    None (displays the plot)
    '''
    # Filter data for the selected phylum (if provided)
    if selected_phylum:
        selected_phylum_data = data[data[phylum_column] == selected_phylum]
        family_counts = selected_phylum_data[family_column].value_counts()

        # Get top N families for the selected phylum
        top_families = family_counts.head(top_n).index
        
        # Filter data for top N families of selected phylum
        filtered_data = selected_phylum_data[selected_phylum_data[family_column].isin(top_families)]
        
        # Plot top N families for selected phylum
        plt.figure(figsize=figsize)
        sns.countplot(data=filtered_data, x=family_column, order=top_families, color='skyblue')
        
        # Add count labels above each bar
        for p in plt.gca().patches:
            height = p.get_height()
            if height > 0:
                plt.gca().annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                                   ha='center', va='center', xytext=(0, 5), textcoords='offset points')

        plt.title(f'Families in Phylum {selected_phylum}')
        plt.xlabel(family_column)
        plt.ylabel('Count')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
    
    else:
        # For all phyla, plot distribution without filtering top N
        plt.figure(figsize=figsize)
        sns.countplot(data=data, x=family_column, hue=phylum_column, color='skyblue')

        # Add count labels above each bar
        for p in plt.gca().patches:
            height = p.get_height()
            if height > 0:
                plt.gca().annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                                   ha='center', va='center', xytext=(0, 5), textcoords='offset points')

        plt.title('Family Distribution Across All Phyla')
        plt.xlabel(family_column)
        plt.ylabel('Count')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

def plot_cumulative_family_distribution(df, selected_phylum, top_n=50):
    """
    Plots the CUMULATIVE FRACTION of rows per family within a selected phylum,
    ordered by the most frequent families.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataset containing at least 'Phylum' and 'Family' columns.
    selected_phylum : str
        The name of the phylum to filter by.
    top_n : int, optional (default=50)
        The maximum number of families to include in the plot (in descending order).
    """
    # 1) Filter rows to the selected phylum
    phylum_df = df[df['phylum'] == selected_phylum].copy()
    
    # 2) Count the number of rows per family, then sort in descending order
    family_counts = phylum_df['family'].value_counts().sort_values(ascending=False)
    
    # 3) Optionally select only the top N families
    top_families = family_counts.iloc[:top_n]
    
    # 4) Compute the fraction of rows each family represents
    total_rows = family_counts.sum()
    top_families_fraction = top_families / total_rows
    
    # 5) Compute the cumulative fraction
    cumulative_fraction = top_families_fraction.cumsum()
    
    
    # 6) Plot the cumulative fraction
    plt.figure(figsize=(15, 6))
    plt.plot(range(len(cumulative_fraction)), cumulative_fraction, marker='o')
    
    plt.title(f'Cumulative Family Distribution (Relative) in {selected_phylum}')
    plt.xlabel('Family (descending frequency)')
    plt.ylabel('Cumulative fraction of rows')
    plt.xticks(range(len(cumulative_fraction)), cumulative_fraction.index, rotation=90)
    plt.ylim(0, 1.05)  # A bit above 1 for padding
    plt.tight_layout()
    plt.show()