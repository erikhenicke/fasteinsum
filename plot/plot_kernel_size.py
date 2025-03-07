import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Read the CSV file, skipping the first row
df = pd.read_csv('kernel_size.csv', header=0,
                 names=['Kernel', 'Time', 'BD', 'ARows', 'ACols', 'BCols', 'B1', 'B2', 'B3', 'Correctness'])

# Strip any leading or trailing spaces from column names
df.columns = df.columns.str.strip()

# Convert 'ARows' column to string, strip spaces, and convert to int
df['ARows'] = df['ARows'].astype(str).str.strip().astype(int)

# Check unique values in the ARows column
print(df['ARows'].unique())

# Function to plot data
def plot_data(df, arows_value, title, filename):
    # Filter data for the given ARows value
    df_filtered = df[df['ARows'] == arows_value]

    # Group by Kernel and calculate mean, min, and max times
    df_grouped = df_filtered.groupby('Kernel')['Time'].agg(['mean', 'min', 'max']).reset_index()

    # Extract kernel sizes from the 'Kernel' column
    kernel_sizes = df_grouped['Kernel'].str.extract(r'(\d+x\d+)')[0]

    # Plot
    plt.figure(figsize=(9, 7))
    sns.set_palette("colorblind")
    barplot = sns.barplot(x='Kernel', y='mean', data=df_grouped, errorbar=None)
    bar_width = barplot.patches[0].get_width()
    plt.errorbar(x=df_grouped['Kernel'], y=df_grouped['mean'],
                 yerr=[df_grouped['mean'] - df_grouped['min'], df_grouped['max'] - df_grouped['mean']], fmt='none',
                 c='black', capsize=bar_width * 32, elinewidth=4, capthick=4)
    plt.xlabel('Kernelsize', fontsize=30)
    plt.ylabel('Time (s)', fontsize=30)
    plt.title(title, fontsize=35)
    plt.xticks(ticks=range(len(kernel_sizes)), labels=kernel_sizes, rotation=45, fontsize=20, ha='right')
    plt.yticks(fontsize=20)
    plt.tight_layout()

    # Save the plot
    plot_dir = 'plot'
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f'{filename}.pdf'))
    plt.savefig(os.path.join(plot_dir, f'{filename}.png'))
    plt.close()

# Plot for small matrices (ARows = 1000)
# plot_data(df, 1000, 'Kernel Times (Matrix Size = 1000)', 'kernel_times_size_1000')

# Plot for big matrices (ARows = 2000)
plot_data(df, 2000, r'Kernel Comparison ($n = 2000$)', 'kernel_times_size_2000_final')