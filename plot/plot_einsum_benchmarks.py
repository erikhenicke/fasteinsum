import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# barplot
def plot_einsum_benchmarks(exclude=True, func_3=True):
    # Read the CSV file
    df = pd.read_csv('results/benchmark_einsum_comparison.csv', header=0,
                     names=['function', 'einsum_string', 'shape1', 'shape2', 'time'])

    # Strip any leading or trailing spaces from column names
    df.columns = df.columns.str.strip()

    if func_3:
        # Filter the DataFrame to include only the specified functions
        df = df[df['function'].isin(['einsum_bmm_avx2', 'einsum_bmm_blas', 'numpy_einsum'])]
        # Define the order of the functions
        function_order = ['numpy_einsum', 'einsum_bmm_avx2', 'einsum_bmm_blas']
        # Define the legend labels
        legend_labels = ['numpy', 'bmm optimized', 'blas']
    else:
        # Define the order of all functions
        function_order = ['numpy_einsum', 'einsum_bmm_avx2', 'einsum_bmm_blas', 'einsum_bmm_omp', 'einsum_bmm_naive']
        # Define the legend labels
        legend_labels = ['numpy', 'bmm optimized', 'blas', 'omp', 'naive']

    if exclude:
        # Define the test cases to exclude
        exclude_cases = [
            'zabcdef-zdfgb-zgeac', 'zabcdef-zdfgc-zgeab', 'zabcdef-zefga-zgdbc',
            'zabcdef-zefgb-zgdac', 'zabcdef-zefgc-zgdab', 'zabcdef-zgdab-zefgc',
            'zabcdef-zgdac-zefgb', 'zabcdef-zgdbc-zefga', 'zabcdef-zgeab-zdfgc',
            'zabcdef-zgeac-zdfgb', 'zabcdef-zgebc-zdfga', 'zabcdef-zgfab-zdegc',
            'zabcdef-zgfac-zdegb', 'zabcdef-zgfbc-zdega', 'zabcd-zaebf-zdfce',
            'zabcd-zaebf-zfdec', 'zabcd-zaecf-zbfde', 'zabcd-zaecf-zfbed',
            'zabcd-zaedf-zbfce', 'zabcd-zaedf-zfbec', 'zabcd-zaefb-zfdce'
        ]
        # Exclude the specified test cases
        df = df[~df['einsum_string'].isin(exclude_cases)]

    # Plot
    plt.figure(figsize=(24, 10))
    sns.set_palette("colorblind")
    barplot = sns.barplot(x='einsum_string', y='time', hue='function', data=df, hue_order=function_order)
    plt.xlabel('Einsum String', fontsize=30)
    plt.ylabel('Time (s)', fontsize=30)
    plt.title('Einsum Benchmark', fontsize=35)
    plt.xticks(rotation=45, fontsize=20, ha='right')
    plt.yticks(fontsize=20)
    plt.yscale('log')
    plt.grid(True, axis='y')
    plt.tight_layout()

    # Customize the legend
    handles, labels = barplot.get_legend_handles_labels()
    barplot.legend(handles, legend_labels, title='Einsum Function', fontsize=20, title_fontsize=25)

    # Save the plot
    plot_dir = 'plot'
    os.makedirs(plot_dir, exist_ok=True)
    filename = 'einsum_benchmark_comparison'
    if exclude:
        filename += '_exclude_cases'
    if func_3:
        filename += '_only_np_avx2_blas'
    plt.savefig(os.path.join(plot_dir, '{}.pdf'.format(filename)))
    plt.savefig(os.path.join(plot_dir, '{}.png'.format(filename)))
    plt.close()

# Example usage
# plot_einsum_benchmarks(exclude=True, func_3=True)
# plot_einsum_benchmarks(exclude=False, func_3=True)
# plot_einsum_benchmarks(exclude=True, func_3=False)
# plot_einsum_benchmarks(exclude=False, func_3=False)

# scatterplot


def plot_einsum_benchmarks_scatter(exclude=True, func_3=True):
    # Read the CSV file
    df = pd.read_csv('results/benchmark_einsum_comparison.csv', header=0,
                     names=['function', 'einsum_string', 'shape1', 'shape2', 'time'])

    # Strip any leading or trailing spaces from column names
    df.columns = df.columns.str.strip()

    if func_3:
        # Filter the DataFrame to include only the specified functions
        df = df[df['function'].isin(['einsum_bmm_avx2', 'einsum_bmm_blas', 'numpy_einsum'])]
        # Define the order of the functions
        function_order = ['numpy_einsum', 'einsum_bmm_avx2', 'einsum_bmm_blas']
        # Define the legend labels
        legend_labels = ['numpy', 'bmm optimized', 'blas']
    else:
        # Define the order of all functions
        function_order = ['numpy_einsum', 'einsum_bmm_avx2', 'einsum_bmm_blas', 'einsum_bmm_omp', 'einsum_bmm_naive']
        # Define the legend labels
        legend_labels = ['numpy', 'bmm optimized', 'blas', 'omp', 'naive']

    if exclude:
        # Define the test cases to exclude
        exclude_cases = [
            'zabcdef-zdfgb-zgeac', 'zabcdef-zdfgc-zgeab', 'zabcdef-zefga-zgdbc',
            'zabcdef-zefgb-zgdac', 'zabcdef-zefgc-zgdab', 'zabcdef-zgdab-zefgc',
            'zabcdef-zgdac-zefgb', 'zabcdef-zgdbc-zefga', 'zabcdef-zgeab-zdfgc',
            'zabcdef-zgeac-zdfgb', 'zabcdef-zgebc-zdfga', 'zabcdef-zgfab-zdegc',
            'zabcdef-zgfac-zdegb', 'zabcdef-zgfbc-zdega', 'zabcd-zaebf-zdfce',
            'zabcd-zaebf-zfdec', 'zabcd-zaecf-zbfde', 'zabcd-zaecf-zfbed',
            'zabcd-zaedf-zbfce', 'zabcd-zaedf-zfbec', 'zabcd-zaefb-zfdce'
            #
            , "zabcdef-zdegc-zgfab", "zabcdef-zdfga-zgebc",

        ]
        # Exclude the specified test cases
        df = df[~df['einsum_string'].isin(exclude_cases)]

    # Plot
    plt.figure(figsize=(24, 10))
    sns.set_palette("colorblind")
    scatterplot = sns.scatterplot(x='einsum_string', y='time', hue='function', data=df, hue_order=function_order, marker='_', s=200, linewidth=5)  # Set marker size and linewidth
    plt.xlabel('Einsum String', fontsize=30)
    plt.ylabel('Time (s)', fontsize=30)
    plt.title('Einsum Benchmark', fontsize=35)
    plt.xticks(rotation=45, fontsize=20)
    plt.yticks(fontsize=20)
    plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()

    # Customize the legend
    handles, labels = scatterplot.get_legend_handles_labels()
    scatterplot.legend(handles, legend_labels, title='Einsum Function', fontsize=20, title_fontsize=25)

    # Save the plot
    plot_dir = 'plot'
    os.makedirs(plot_dir, exist_ok=True)
    filename = 'einsum_scatter'
    if exclude:
        filename += '_exclude_cases'
    if func_3:
        filename += '_only_np_avx2_blas'
    plt.savefig(os.path.join(plot_dir, '{}.pdf'.format(filename)))
    plt.savefig(os.path.join(plot_dir, '{}.png'.format(filename)))
    plt.close()

# Example usage
# plot_einsum_benchmarks_scatter(exclude=True, func_3=True)
# plot_einsum_benchmarks_scatter(exclude=False, func_3=True)
# plot_einsum_benchmarks_scatter(exclude=True, func_3=False)
# plot_einsum_benchmarks_scatter(exclude=False, func_3=False)


# custom plot


def plot_einsum_benchmarks_custom(exclude=True):
    # Read the CSV file
    df = pd.read_csv('results/benchmark_einsum_comparison.csv', header=0,
                     names=['function', 'einsum_string', 'shape1', 'shape2', 'time'])

    # Strip any leading or trailing spaces from column names
    df.columns = df.columns.str.strip()

    # Filter the DataFrame to include only the specified functions
    df_numpy = df[df['function'] == 'numpy_einsum']
    df_avx2 = df[df['function'] == 'einsum_bmm_avx2']
    df_blas = df[df['function'] == 'einsum_bmm_blas']

    s_ = 1000 # blas linewidth

    if exclude:
        s_ = 2000
        # Define the test cases to exclude
        exclude_cases = [
            'zabcdef-zdfgb-zgeac', 'zabcdef-zdfgc-zgeab', 'zabcdef-zefga-zgdbc',
            'zabcdef-zefgb-zgdac', 'zabcdef-zefgc-zgdab', 'zabcdef-zgdab-zefgc',
            'zabcdef-zgdac-zefgb', 'zabcdef-zgdbc-zefga', 'zabcdef-zgeab-zdfgc',
            'zabcdef-zgeac-zdfgb', 'zabcdef-zgebc-zdfga', 'zabcdef-zgfab-zdegc',
            'zabcdef-zgfac-zdegb', 'zabcdef-zgfbc-zdega', 'zabcd-zaebf-zdfce',
            'zabcd-zaebf-zfdec', 'zabcd-zaecf-zbfde', 'zabcd-zaecf-zfbed',
            'zabcd-zaedf-zbfce', 'zabcd-zaedf-zfbec', 'zabcd-zaefb-zfdce'
            #
            # , "zabcdef-zdegc-zgfab", "zabcdef-zdfga-zgebc", "zabcd-zaebf-zfdec", "zabcd-zaecf-zbfde", "zabcd-zaecf-zfbed"

        ]
        # Exclude the specified test cases
        df_numpy = df_numpy[~df_numpy['einsum_string'].isin(exclude_cases)]
        df_avx2 = df_avx2[~df_avx2['einsum_string'].isin(exclude_cases)]
        df_blas = df_blas[~df_blas['einsum_string'].isin(exclude_cases)]

    # Plot
    plt.figure(figsize=(24, 10))
    sns.set_palette("colorblind")

    # Plot numpy_einsum as a transparent bar
    barplot_numpy = sns.barplot(x='einsum_string', y='time', data=df_numpy, color='blue', alpha=0.3, label='numpy')

    # Plot einsum_bmm_avx2 as a non-opaque bar in front of numpy_einsum
    barplot_avx2 = sns.barplot(x='einsum_string', y='time', data=df_avx2, color='red', alpha=0.7, label='bmm optimized')

    # Plot einsum_bmm_blas as a scatter plot with thick horizontal lines
    scatterplot_blas = sns.scatterplot(x='einsum_string', y='time', data=df_blas, color='black', marker='_', s=s_, linewidth=5, label='blas')

    plt.xlabel('Einsum String', fontsize=30)
    plt.ylabel('Time (s)', fontsize=30)
    plt.title('Einsum Benchmark', fontsize=35)
    plt.xticks(rotation=45, fontsize=20, ha='right')
    plt.yticks(fontsize=20)
    plt.yscale('log')
    plt.grid(True, axis='y')
    plt.tight_layout()

    # Customize the legend
    plt.legend(title='Einsum Function', fontsize=20, title_fontsize=25)

    # Save the plot
    plot_dir = 'plot'
    os.makedirs(plot_dir, exist_ok=True)
    filename = 'einsum_custom_plot'
    if exclude:
        filename += '_exclude_cases'
    plt.savefig(os.path.join(plot_dir, '{}.pdf'.format(filename)))
    plt.savefig(os.path.join(plot_dir, '{}.png'.format(filename)))
    plt.close()

# Example usage
# plot_einsum_benchmarks_custom(exclude=True)
# plot_einsum_benchmarks_custom(exclude=False)

# plot over shapes
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_einsum_benchmarks_by_shape():
    # Read the CSV file
    df = pd.read_csv('results/einsum_benchmark_comparison_shapes.csv', header=0,
                     names=['function', 'einsum_string', 'shape1', 'shape2', 'time', 'shapeA', 'shapeB'])

    # Strip any leading or trailing spaces from column names
    df.columns = df.columns.str.strip()

    # Combine shapeA and shapeB into a single column for the x-axis
    df['shapes'] = df['shapeA'] + ' x ' + df['shapeB']

    # Combine shapes and functions into a single column for the x-axis
    df['shapes_and_func'] = df['shapes'] + ' (' + df['function'] + ')'

    # Drop duplicate rows based on the 'shapes and function' column
    df = df.drop_duplicates(subset='shapes_and_func')

    # Filter the DataFrame to include only the specified functions
    df_numpy = df[df['function'] == 'numpy_einsum']
    df_avx2 = df[df['function'] == 'einsum_bmm_avx2']
    df_blas = df[df['function'] == 'einsum_bmm_blas']

    s_ = 2000  # blas linewidth

    # Plot
    plt.figure(figsize=(16, 8))
    sns.set_palette("colorblind")

    # Plot numpy_einsum as a transparent bar
    barplot_numpy = sns.barplot(x='shapes', y='time', data=df_numpy, color='blue', alpha=0.3, label='numpy')

    # Plot einsum_bmm_avx2 as a non-opaque bar in front of numpy_einsum
    barplot_avx2 = sns.barplot(x='shapes', y='time', data=df_avx2, color='red', alpha=0.7, label='bmm optimized')

    # Plot einsum_bmm_blas as a scatter plot with thick horizontal lines
    scatterplot_blas = sns.scatterplot(x='shapes', y='time', data=df_blas, color='black', marker='_', s=s_, linewidth=5, label='blas')

    plt.xlabel('Shapes', fontsize=30)
    plt.ylabel('Time (s)', fontsize=30)
    plt.title('Einsum Benchmark by Shapes', fontsize=35)
    plt.xticks(rotation=45, fontsize=20, ha='right')
    plt.yticks(fontsize=20)
    plt.yscale('log')
    plt.grid(True, axis='y')
    plt.tight_layout()

    # Customize the legend
    plt.legend(title='Einsum Function', fontsize=20, title_fontsize=25)

    # Save the plot
    plot_dir = 'plot'
    os.makedirs(plot_dir, exist_ok=True)
    filename = 'einsum_custom_plot_by_shapes'
    plt.savefig(os.path.join(plot_dir, '{}.pdf'.format(filename)))
    plt.savefig(os.path.join(plot_dir, '{}.png'.format(filename)))
    plt.close()

# Example usage
plot_einsum_benchmarks_by_shape()



# plot over shapes
def plot_einsum_benchmarks_by_shapes(func_3=True):
    # Read the CSV file
    df = pd.read_csv('results/einsum_benchmark_comparison_shapes.csv', header=0,
                     names=['function', 'einsum_string', 'shape1', 'shape2', 'time', 'shapeA', 'shapeB'])

    # Strip any leading or trailing spaces from column names
    df.columns = df.columns.str.strip()

    # Combine shapeA and shapeB into a single column for the x-axis
    df['shapes'] = df['shapeA'] + ' x ' + df['shapeB']

    # Combine shapes and functions into a single column for the x-axis
    df['shapes_and_func'] = df['shapes'] + ' (' + df['function'] + ')'

    # Drop duplicate rows based on the 'shapes and function' column
    df = df.drop_duplicates(subset='shapes_and_func')

    if func_3:
        # Filter the DataFrame to include only the specified functions
        df = df[df['function'].isin(['einsum_bmm_avx2', 'einsum_bmm_blas', 'numpy_einsum'])]
        # Define the order of the functions
        function_order = ['numpy_einsum', 'einsum_bmm_avx2', 'einsum_bmm_blas']
        # Define the legend labels
        legend_labels = ['numpy', 'bmm optimized', 'blas']
    else:
        # Define the order of all functions
        function_order = ['numpy_einsum', 'einsum_bmm_avx2', 'einsum_bmm_blas', 'einsum_bmm_omp', 'einsum_bmm_naive']
        # Define the legend labels
        legend_labels = ['numpy', 'bmm optimized', 'blas', 'omp', 'naive']

    # Plot
    plt.figure(figsize=(24, 10))
    sns.set_palette("colorblind")
    barplot = sns.barplot(x='shapes', y='time', hue='function', data=df, hue_order=function_order)
    plt.xlabel('Shapes', fontsize=30)
    plt.ylabel('Time (s)', fontsize=30)
    plt.title('Einsum Benchmark by Shapes', fontsize=35)
    plt.xticks(rotation=45, fontsize=20, ha='right')
    plt.yticks(fontsize=20)
    plt.yscale('log')
    plt.grid(True, axis='y')
    plt.tight_layout()

    # Customize the legend
    handles, labels = barplot.get_legend_handles_labels()
    barplot.legend(handles, legend_labels, title='Einsum Function', fontsize=20, title_fontsize=25)

    # Save the plot
    plot_dir = 'plot'
    os.makedirs(plot_dir, exist_ok=True)
    filename = 'einsum_benchmark_comparison_by_shapes'

    if func_3:
        filename += '_only_np_avx2_blas'
    plt.savefig(os.path.join(plot_dir, '{}.pdf'.format(filename)))
    plt.savefig(os.path.join(plot_dir, '{}.png'.format(filename)))
    plt.close()

# Example usage
plot_einsum_benchmarks(func_3=True)
plot_einsum_benchmarks(func_3=False)