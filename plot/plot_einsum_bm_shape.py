import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_einsum_benchmarks_by_shape():
    # Read the CSV file
    df = pd.read_csv('updated_einsum_benchmark_comparison.csv', header=0,
                     names=['function', 'einsum_string', 'shape1', 'shape2', 'time', 'shapeA', 'shapeB'])

    # Strip any leading or trailing spaces from column names
    df.columns = df.columns.str.strip()

    # Combine the second and third elements of shapeA and shapeB into a single column for the x-axis
    df['shapes'] = df['shapeA'].apply(lambda x: f"({eval(x)[1]}, {eval(x)[2]})") + ' x ' + df['shapeB'].apply(lambda x: f"({eval(x)[1]}, {eval(x)[2]})")

    # Combine shapes and functions into a single column for the x-axis
    df['shapes_and_func'] = df['shapes'] + ' (' + df['function'] + ')'

    # Drop duplicate rows based on the 'shapes_and_func' column
    df = df.drop_duplicates(subset='shapes_and_func')

    # Filter the DataFrame to include only the specified functions
    df_numpy = df[df['function'] == 'numpy_einsum']
    df_avx2 = df[df['function'] == 'einsum_bmm_avx2']
    df_blas = df[df['function'] == 'einsum_bmm_blas']

    s_ = 1900  # blas linewidth

    # Plot
    plt.figure(figsize=(16, 10))
    sns.set_palette("colorblind")

    # Plot numpy_einsum as a transparent bar
    barplot_numpy = sns.barplot(x='shapes', y='time', data=df_numpy, color='blue', alpha=0.3, label='numpy')

    # Plot einsum_bmm_avx2 as a non-opaque bar in front of numpy_einsum
    barplot_avx2 = sns.barplot(x='shapes', y='time', data=df_avx2, color='red', label='bmm blocking and\nkernel (AVX2)')

    # Plot einsum_bmm_blas as a scatter plot with thick horizontal lines
    scatterplot_blas = sns.scatterplot(x='shapes', y='time', data=df_blas, color='black', marker='_', s=s_, linewidth=5, label='bmm blas')

    plt.xlabel('Shapes', fontsize=30)
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
    filename = 'einsum_custom_plot_by_shapes'
    plt.savefig(os.path.join(plot_dir, '{}.pdf'.format(filename)))
    plt.savefig(os.path.join(plot_dir, '{}.png'.format(filename)))
    plt.close()

# Example usage
plot_einsum_benchmarks_by_shape()