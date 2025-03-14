import pandas as pd

data = pd.read_csv('/home/erik/CLionProjects/ae_ws2024_bmm/data/einsum_benchmark_comparison_shapes.csv')
data['Tensor contraction'] = pd.Categorical(data['einsum_str'], categories=data['einsum_str'].unique(), ordered=True)
data = data.pivot(index=['Tensor contraction', 'shape1', 'shape2', 'shapeA', 'shapeB'], columns='function', values='time')
data.reset_index(inplace=True)
data.rename(columns={'einsum_bmm_naive': 'Naive', 'einsum_bmm_avx2': 'AVX2', 'einsum_bmm_omp': 'OMP', 'einsum_bmm_blas': 'BLAS', 'numpy_einsum': 'NumPy'}, inplace=True)

data['Tensor shapes'] = data['shape1'] + ';' + data['shape2']

data['shapeA'] = data['shapeA'].str.replace(r'\((\d+), (\d+), (\d+)\)', r'\1-\2-\3', regex=True)
data['shapeB'] = data['shapeB'].str.replace(r'\((\d+), (\d+), (\d+)\)', r'\1-\2-\3', regex=True)
data['Matrix shapes'] = data['shapeA'] + ';' + data['shapeB']

data = data.round(3)
data.to_csv('/home/erik/CLionProjects/ae_ws2024_bmm/data/einsum_benchmark_comparison_final.csv', index=False, columns=['Tensor contraction', 'Tensor shapes', 'Matrix shapes', 'Naive', 'AVX2', 'OMP', 'BLAS', 'NumPy'])
