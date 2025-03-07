import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_bmm_small():
    # Read the CSV files
    df_naive = pd.read_csv('performance_bmm_naive_comparison.csv', header=0,
                           names=['function', 'BD', 'ARows', 'ACols', 'BCols', 'B1', 'B2', 'B3', 'Correctness', 'time'])
    df_naive.columns = df_naive.columns.str.strip()
    df_comparison = pd.read_csv('performance_bmm_comparison.csv', header=0,
                                names=['function', 'BD', 'ARows', 'ACols', 'BCols', 'B1', 'B2', 'B3', 'Correctness', 'time'])
    df_comparison.columns = df_comparison.columns.str.strip()
    df_omp = pd.read_csv('performance_bmm_omp_comparison.csv', header=0,
                         names=['function', 'BD', 'ARows', 'ACols', 'BCols', 'B1', 'B2', 'B3', 'Correctness', 'time'])
    df_omp.columns = df_omp.columns.str.strip()

    # Combine the DataFrames
    df_combined = pd.concat([df_naive, df_comparison, df_omp], ignore_index=True)

    # Define the functions to plot
    parallel_functions = ['blas parallel', 'kernel parallel', 'packing parallel', 'naive parallel']#, 'packing omp parallel']
    single_threaded_functions = ['kernel', 'packing', 'naive']

    # Filter the DataFrame to include only the selected functions
    df_filtered = df_combined[df_combined['function'].isin(parallel_functions + single_threaded_functions)]

    # Filter for size (ARows) smaller than 300
    df_filtered = df_filtered[df_filtered['ARows'] < 1000]

    # Define custom styles and colors
    styles = {
        'naive parallel': '',
        'blas parallel': '',
        'kernel parallel': '',
        'packing parallel': '',
        # 'packing omp parallel': '',
        'naive': (5, 5),
        # 'blas': (5, 5),
        'kernel': (5, 5),
        'packing': (5, 5)
    }
    colors = {
        'naive parallel': 'C0',
        'blas parallel': 'C1',
        'kernel parallel': 'C2',
        'packing parallel': 'C3',
        # 'packing omp parallel': 'C4',
        'naive': 'C0',
        'blas': 'C1',
        'kernel': 'C2',
        'packing': 'C3'
    }

    # Plot
    plt.figure(figsize=(12, 12))
    sns.set_palette("colorblind")

    # Plot time over size for the selected functions with bullet point markers
    sns.lineplot(x='ARows', y='time', hue='function', data=df_filtered, style='function', dashes=styles, markers=['o']*len(parallel_functions + single_threaded_functions), palette=colors)

    plt.xlabel(r'$n$', fontsize=30)
    plt.ylabel(r'Time (s)', fontsize=30)
    plt.title('Time over Size for Parallel and\n Single-Threaded BMM Functions', fontsize=35)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.yscale('log')  # Set y-axis to log scale
    # plt.xscale('log')  # Set x-axis to log scale
    plt.grid(True)
    plt.tight_layout()

    # Customize the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    legend_labels = ['naive (single-threaded)', 'naive (parallel)', 'blas', 'kernel (AVX2) (single-threaded)',
                     'kernel (AVX2) (parallel)', 'blocking and kernel\n(AVX2) (single-threaded)', 'blocking and kernel\n(AVX2) (parallel)']
    plt.legend(handles=handles, labels=legend_labels, title='BMM Function', fontsize=20, title_fontsize=25)

    # Save the plot
    plot_dir = 'plot'
    os.makedirs(plot_dir, exist_ok=True)
    filename = 'time_over_size_small'
    plt.savefig(os.path.join(plot_dir, '{}.pdf'.format(filename)))
    plt.savefig(os.path.join(plot_dir, '{}.png'.format(filename)))
    plt.close()

# Example usage
plot_bmm_small()