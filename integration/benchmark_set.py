import numpy as np

list_format_strings = [

    "abc-bda-dc",
    "abc-dca-bd",
    "abcd-dbea-ec",
    "abcd-deca-be",
    "abcd-ebad-ce",

    "abcde-efbad-cf",
    "abcde-ecbfa-fd",
    "abcde-efcad-bf",
    "abcd-ea-ebcd",
    "abcd-eb-aecd",

    "abcd-ec-abed",
    "ab-ac-cb",
    "ab-acd-dbc",
    "ab-cad-dcb",
    "abc-acd-db",

    "abc-ad-bdc",
    "abc-adc-bd",
    "abc-adc-db",
    "abc-adec-ebd",
    "abcd-aebf-dfce",

    "abcd-aebf-fdec",
    "abcd-aecf-bfde",
    "abcd-aecf-fbed",
    "abcd-aedf-bfce",
    "abcd-aedf-fbec",

    "abcd-aefb-fdce",
    "abcd-aefc-fbed",
    "abcd-eafb-fdec",
    "abcd-eafc-bfde",
    "abcd-eafd-fbec",

    "abcdef-dega-gfbc",
    "abcdef-degb-gfac",
    "abcdef-degc-gfab",
    "abcdef-dfga-gebc",
    "abcdef-dfgb-geac",

    "abcdef-dfgc-geab",
    "abcdef-efga-gdbc",
    "abcdef-efgb-gdac",
    "abcdef-efgc-gdab",
    "abcdef-gdab-efgc",

    "abcdef-gdac-efgb",
    "abcdef-gdbc-efga",
    "abcdef-geab-dfgc",
    "abcdef-geac-dfgb",
    "abcdef-gebc-dfga",

    "abcdef-gfab-degc",
    "abcdef-gfac-degb",
    "abcdef-gfbc-dega"
]

list_sizes = [
    ### double precision
    "a:312;c:24;b:312;d:312;",
    "a:312;c:296;b:24;d:312;",
    "a:72;c:24;b:72;e:72;d:72;",
    "a:72;c:72;b:24;e:72;d:72;",
    "a:72;c:24;b:72;e:72;d:72;",
    "a:48;c:24;b:32;e:48;d:32;f:32;",
    "a:48;c:32;b:32;e:48;d:24;f:48;",
    "a:48;c:32;b:24;e:48;d:32;f:32;",
    "a:72;c:72;b:72;e:72;d:72;",
    "a:72;c:72;b:72;e:72;d:72;",
    "a:72;c:72;b:72;e:72;d:72;",
    "a:5136;c:5136;b:5120;",
    "a:312;c:296;b:296;d:312;",
    "a:312;c:312;b:296;d:312;",
    "a:312;c:296;b:296;d:312;",
    "a:312;c:296;b:312;d:296;",
    "a:312;c:296;b:312;d:296;",
    "a:312;c:296;b:296;d:312;",
    "a:72;c:72;b:72;e:72;d:72;",
    "a:72;c:72;b:72;e:72;d:72;f:72;",
    "a:72;c:72;b:72;e:72;d:72;f:72;",
    "a:72;c:72;b:72;e:72;d:72;f:72;",
    "a:72;c:72;b:72;e:72;d:72;f:72;",
    "a:72;c:72;b:72;e:72;d:72;f:72;",
    "a:72;c:72;b:72;e:72;d:72;f:72;",
    "a:72;c:72;b:72;e:72;d:72;f:72;",
    "a:72;c:72;b:72;e:72;d:72;f:72;",
    "a:72;c:72;b:72;e:72;d:72;f:72;",
    "a:72;c:72;b:72;e:72;d:72;f:72;",
    "a:72;c:72;b:72;e:72;d:72;f:72;",
    "a:24;c:16;b:16;e:16;d:24;g:24;f:16;",
    "a:24;c:16;b:16;e:16;d:24;g:24;f:16;",
    "a:24;c:16;b:16;e:16;d:24;g:24;f:16;",
    "a:24;c:16;b:16;e:16;d:24;g:24;f:16;",
    "a:24;c:16;b:16;e:16;d:24;g:24;f:16;",
    "a:24;c:16;b:16;e:16;d:24;g:24;f:16;",
    "a:24;c:16;b:16;e:24;d:16;g:24;f:16;",
    "a:24;c:16;b:16;e:24;d:16;g:24;f:16;",
    "a:24;c:16;b:16;e:24;d:16;g:24;f:16;",
    "a:24;c:16;b:16;e:24;d:16;g:24;f:16;",
    "a:24;c:16;b:16;e:24;d:16;g:24;f:16;",
    "a:24;c:16;b:16;e:24;d:16;g:24;f:16;",
    "a:24;c:16;b:16;e:16;d:24;g:24;f:16;",
    "a:24;c:16;b:16;e:16;d:24;g:24;f:16;",
    "a:24;c:16;b:16;e:16;d:24;g:24;f:16;",
    "a:24;c:16;b:16;e:16;d:24;g:24;f:16;",
    "a:24;c:16;b:16;e:16;d:24;g:24;f:16;",
    "a:24;c:16;b:16;e:16;d:24;g:24;f:16;"

    ### single precision
    # "a:384;c:24;b:384;d:384;",
    # "a:384;c:376;b:24;d:384;",
    # "a:96;c:24;b:84;e:96;d:96;",
    # "a:96;c:84;b:24;e:84;d:96;",
    # "a:96;c:24;b:84;e:96;d:84;",
    # "a:48;c:24;b:36;e:48;d:36;f:36;",
    # "a:48;c:36;b:36;e:48;d:24;f:48;",
    # "a:48;c:36;b:24;e:48;d:36;f:36;",
    # "a:96;c:84;b:84;e:96;d:84;",
    # "a:96;c:84;b:84;e:96;d:84;",
    # "a:96;c:84;b:84;e:96;d:84;",
    # "a:7248;c:7248;b:7240;",
    # "a:384;c:376;b:376;d:384;",
    # "a:384;c:384;b:376;d:384;",
    # "a:384;c:376;b:376;d:384;",
    # "a:384;c:376;b:384;d:376;",
    # "a:384;c:376;b:384;d:376;",
    # "a:384;c:376;b:376;d:384;",
    # "a:96;c:84;b:84;e:96;d:84;",
    # "a:96;c:84;b:84;e:84;d:96;f:84;",
    # "a:96;c:84;b:84;e:84;d:84;f:96;",
    # "a:96;c:84;b:96;e:84;d:84;f:84;",
    # "a:96;c:84;b:84;e:84;d:84;f:96;",
    # "a:96;c:84;b:96;e:84;d:84;f:84;",
    # "a:96;c:84;b:84;e:84;d:84;f:96;",
    # "a:96;c:84;b:84;e:84;d:84;f:96;",
    # "a:96;c:84;b:84;e:84;d:84;f:96;",
    # "a:96;c:84;b:84;e:96;d:84;f:96;",
    # "a:96;c:84;b:96;e:96;d:84;f:84;",
    # "a:96;c:84;b:84;e:96;d:84;f:96;",
    # "a:24;c:20;b:20;e:20;d:24;g:24;f:20;",
    # "a:24;c:20;b:20;e:20;d:24;g:24;f:20;",
    # "a:24;c:20;b:20;e:20;d:24;g:24;f:20;",
    # "a:24;c:20;b:20;e:20;d:24;g:24;f:20;",
    # "a:24;c:20;b:20;e:20;d:24;g:24;f:20;",
    # "a:24;c:20;b:20;e:20;d:24;g:24;f:20;",
    # "a:24;c:20;b:20;e:24;d:20;g:24;f:20;",
    # "a:24;c:20;b:20;e:24;d:20;g:24;f:20;",
    # "a:24;c:20;b:20;e:24;d:20;g:24;f:20;",
    # "a:24;c:20;b:20;e:24;d:20;g:24;f:20;",
    # "a:24;c:20;b:20;e:24;d:20;g:24;f:20;",
    # "a:24;c:20;b:20;e:24;d:20;g:24;f:20;",
    # "a:24;c:20;b:20;e:20;d:24;g:24;f:20;",
    # "a:24;c:20;b:20;e:20;d:24;g:24;f:20;",
    # "a:24;c:20;b:20;e:20;d:24;g:24;f:20;",
    # "a:24;c:20;b:20;e:20;d:24;g:24;f:20;",
    # "a:24;c:20;b:20;e:20;d:24;g:24;f:20;",
    # "a:24;c:20;b:20;e:20;d:24;g:24;f:20;"
]

num_repeats = [
    20,
    20,
    20,
    20,
    20,
    1,
    1,
    2,
    20,
    20,

    20,
    1,
    10,
    10,
    10,
    10,
    10,
    10,
    20,
    1,

    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,

    20,
    20,
    20,
    20,
    20,
    20,
    20,
    20,
    20,
    20,

    20,
    20,
    20,
    20,
    20,
    20,
    20,
    20
]

# helper functions
def format_str2short_str(einsum_str):
    idx1, rest = einsum_str.split(',')
    idx2, idx3 = rest.split('->')
    return f"{idx3}-{idx1}-{idx2}"

def short_str2format_str(short_str):
    idx3, idx1, idx2 = short_str.split('-')
    return f"{idx1},{idx2}->{idx3}"

def shape2shape_str(shape):
    return '-'.join(map(str, shape))

def shape_str2shape(shape_str):
    return tuple(map(int, shape_str.split('-')))

def parse_sizes(size_str):
    sizes = {}
    for item in size_str.split(';'):
        if item:
            key, value = item.split(':')
            sizes[key] = int(value)
    return sizes

def get_testcases():
    test_cases = []
    for expr, size_str, num_repeat in zip(list_format_strings, list_sizes, num_repeats):
        indices = expr.split('-')
        einsum_str = f"{indices[1]},{indices[2]}->{indices[0]}"
        sizes = parse_sizes(size_str)
        shapes = [tuple(sizes[dim] for dim in indices[1]), tuple(sizes[dim] for dim in indices[2])]
        test_cases.append((einsum_str, *shapes, num_repeat))
    return test_cases

# test_cases = make_testcases()
# print("No test cases: ", len(test_cases))
# for test_case in test_cases:
#     print(test_case)

def get_testcases_with_bd(bd):
    test_cases = []
    for expr, size_str, num_repeat in zip(list_format_strings, list_sizes, num_repeats):
        indices = expr.split('-')
        einsum_str = f"z{indices[1]},z{indices[2]}->z{indices[0]}"
        sizes = parse_sizes(size_str)
        shapes = [(bd,) + tuple(sizes[dim] for dim in indices[1]), (bd,) + tuple(sizes[dim] for dim in indices[2])]
        test_cases.append((einsum_str, *shapes, num_repeat))
    return test_cases

# Example usage
# bd = 10
# test_cases_bd = make_testcases_add_bd(bd)
# print("No test cases bd: ", len(test_cases_bd))
# for test_case in test_cases_bd:
#     print(test_case)


def make_bd_only_testcases(sizes):
    # tupel sizes: (bd, a_rows, a_cols, b_cols)
    test_cases = []
    for size in sizes:
        bd, a_rows, a_cols, b_cols = size
        einsum_str = f"zab,zbc->zac"
        shapes = [(bd, a_rows, a_cols), (bd, a_cols, b_cols)]
        test_cases.append((einsum_str, *shapes))
    return test_cases

# sizes = [(10, 100, 100, 100), (10, 200, 200, 200), (10, 100, 200, 300), (10, 300, 200, 100)]
# test_cases_only_bd = make_only_bd_testcases(sizes)
# print("No test cases only bd: ", len(test_cases_only_bd))
# for test_case in test_cases_only_bd:
#     print(test_case)


MAX_TENSOR_SIZE = 100000000  # Adjust this value as needed
BIG_TENSOR_SIZE = 700000000000000

def generate_einsum_input(test_case):
    einsum_str, shape1, shape2, num_repeat = test_case

    # size1 = np.prod(shape1)
    # size2 = np.prod(shape2)
    # if size1 > MAX_TENSOR_SIZE or size2 > MAX_TENSOR_SIZE:
    #     print(f"Warning: Skipping tensor generation for test case {test_case} due to large tensor size: {size1}, {size2}, {size1 * size2}")

    # Delete previous tensors if they exist
    try:
        del tensor1
        del tensor2
    except NameError:
        pass

    tensor1 = np.random.rand(*shape1)
    tensor2 = np.random.rand(*shape2)
    return einsum_str, tensor1, tensor2, num_repeat

# # Calculate einsum result for a test case
# test_case = test_cases[0]
# einsum_input = generate_einsum_input(test_case)
# result = np.einsum(*einsum_input)
#
# # Calculate einsum result for a test case with bd
# test_case_bd = test_cases_bd[0]
# einsum_input_bd = generate_einsum_input(test_case_bd)
# result_bd = np.einsum(*einsum_input_bd)
# #
# # Calculate einsum result for a test case with only bd
# test_case_only_bd = test_cases_only_bd[0]
# einsum_input_only_bd = generate_einsum_input(test_case_only_bd)
# result_only_bd = np.einsum(*einsum_input_only_bd)
#
# print("Result: ", result)
# print("Result bd: ", result_bd)
# print("Result only bd: ", result_only_bd)
