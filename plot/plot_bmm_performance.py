import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_parallel_functions():
    # Read the CSV files
    df_naive = pd.read_csv('results/performance_bmm_naive_comparison.csv', header=0,
                           names=['function', 'BD', 'ARows', 'ACols', 'BCols', 'B1', 'B2', 'B3', 'Correctness', 'time'])
    df_naive.columns = df_naive.columns.str.strip()
    df_comparison = pd.read_csv('results/performance_bmm_comparison.csv', header=0,
                                names=['function', 'BD', 'ARows', 'ACols', 'BCols', 'B1', 'B2', 'B3', 'Correctness', 'time'])
    df_comparison.columns = df_comparison.columns.str.strip()
    df_omp = pd.read_csv('results/performance_bmm_omp_comparison.csv', header=0,
                         names=['function', 'BD', 'ARows', 'ACols', 'BCols', 'B1', 'B2', 'B3', 'Correctness', 'time'])
    df_omp.columns = df_omp.columns.str.strip()

    # Combine the DataFrames
    df_combined = pd.concat([df_naive, df_comparison, df_omp], ignore_index=True)

    # Filter the DataFrame to include only parallel functions
    parallel_functions = ['blas parallel', 'kernel parallel', 'packing parallel', 'naive parallel', 'packing omp parallel']
    df_filtered = df_combined[df_combined['function'].isin(parallel_functions)]

    # Filter out y values greater than 50
    df_filtered = df_filtered[df_filtered['time'] < 50]

    # Define custom styles and colors
    styles = {
        'naive parallel': '',
        'blas parallel': '',
        'kernel parallel': '',
        'packing parallel': '',
        'packing omp parallel': ''
    }
    colors = {
        'naive parallel': 'C0',
        'blas parallel': 'C1',
        'kernel parallel': 'C2',
        'packing parallel': 'C3',
        'packing omp parallel': 'C4'
    }

    # Plot
    plt.figure(figsize=(14, 8))
    sns.set_palette("colorblind")

    # Plot time over size for the selected functions with bullet point markers
    sns.lineplot(x='ARows', y='time', hue='function', data=df_filtered, style='function', dashes=styles, markers=['o']*len(parallel_functions), palette=colors)

    # Fill area below 'blas parallel'
    blas_parallel = df_filtered[df_filtered['function'] == 'blas parallel']
    plt.fill_between(blas_parallel['ARows'], blas_parallel['time'], color='lightgrey', alpha=0.5)

    # Fill area above 'naive parallel'
    naive_parallel = df_filtered[df_filtered['function'] == 'naive parallel']
    plt.fill_between(naive_parallel['ARows'], naive_parallel['time'], y2=30, color='lightgrey', alpha=0.5)

    # # Fill the space below time = 0
    # plt.fill_between(df_filtered['ARows'], 0, color='lightgrey', alpha=0.5)
    #
    # Fill the space below n = 50 and up to y = 30
    plt.fill_betweenx(naive_parallel['time'], 0, 50, color='lightgrey', alpha=0.5)

    plt.xlabel(r'$n$', fontsize=30)
    plt.ylabel(r'Time (s)', fontsize=30)
    plt.title('Time over Size for Parallel BMM Functions', fontsize=35)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(0, 30)  # Set y-axis limit to 30
    plt.xlim(0, 8000)  # Set x-axis limit to 8000
    plt.grid(True)
    plt.tight_layout()

    # Customize the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    legend_labels = ['naive', 'blas', 'kernel (AVX2)', 'blocking and kernel (AVX2)', 'blocking and kernel (OMP)']
    plt.legend(handles=handles, labels=legend_labels, title='BMM Function (parallel)', fontsize=20, title_fontsize=25)

    # Save the plot
    plot_dir = 'plot'
    os.makedirs(plot_dir, exist_ok=True)
    filename = 'time_over_size_plot_parallel_functions'
    plt.savefig(os.path.join(plot_dir, '{}.pdf'.format(filename)))
    plt.savefig(os.path.join(plot_dir, '{}.png'.format(filename)))
    plt.close()

# Example usage
plot_parallel_functions()

