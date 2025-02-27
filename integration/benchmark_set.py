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

    "a:384;c:24;b:384;d:384;",
    "a:384;c:376;b:24;d:384;",
    "a:96;c:24;b:84;e:96;d:96;",
    "a:96;c:84;b:24;e:84;d:96;",
    "a:96;c:24;b:84;e:96;d:84;",
    "a:48;c:24;b:36;e:48;d:36;f:36;",
    "a:48;c:36;b:36;e:48;d:24;f:48;",
    "a:48;c:36;b:24;e:48;d:36;f:36;",
    "a:96;c:84;b:84;e:96;d:84;",
    "a:96;c:84;b:84;e:96;d:84;",
    "a:96;c:84;b:84;e:96;d:84;",
    "a:7248;c:7248;b:7240;",
    "a:384;c:376;b:376;d:384;",
    "a:384;c:384;b:376;d:384;",
    "a:384;c:376;b:376;d:384;",
    "a:384;c:376;b:384;d:376;",
    "a:384;c:376;b:384;d:376;",
    "a:384;c:376;b:376;d:384;",
    "a:96;c:84;b:84;e:96;d:84;",
    "a:96;c:84;b:84;e:84;d:96;f:84;",
    "a:96;c:84;b:84;e:84;d:84;f:96;",
    "a:96;c:84;b:96;e:84;d:84;f:84;",
    "a:96;c:84;b:84;e:84;d:84;f:96;",
    "a:96;c:84;b:96;e:84;d:84;f:84;",
    "a:96;c:84;b:84;e:84;d:84;f:96;",
    "a:96;c:84;b:84;e:84;d:84;f:96;",
    "a:96;c:84;b:84;e:84;d:84;f:96;",
    "a:96;c:84;b:84;e:96;d:84;f:96;",
    "a:96;c:84;b:96;e:96;d:84;f:84;",
    "a:96;c:84;b:84;e:96;d:84;f:96;",
    "a:24;c:20;b:20;e:20;d:24;g:24;f:20;",
    "a:24;c:20;b:20;e:20;d:24;g:24;f:20;",
    "a:24;c:20;b:20;e:20;d:24;g:24;f:20;",
    "a:24;c:20;b:20;e:20;d:24;g:24;f:20;",
    "a:24;c:20;b:20;e:20;d:24;g:24;f:20;",
    "a:24;c:20;b:20;e:20;d:24;g:24;f:20;",
    "a:24;c:20;b:20;e:24;d:20;g:24;f:20;",
    "a:24;c:20;b:20;e:24;d:20;g:24;f:20;",
    "a:24;c:20;b:20;e:24;d:20;g:24;f:20;",
    "a:24;c:20;b:20;e:24;d:20;g:24;f:20;",
    "a:24;c:20;b:20;e:24;d:20;g:24;f:20;",
    "a:24;c:20;b:20;e:24;d:20;g:24;f:20;",
    "a:24;c:20;b:20;e:20;d:24;g:24;f:20;",
    "a:24;c:20;b:20;e:20;d:24;g:24;f:20;",
    "a:24;c:20;b:20;e:20;d:24;g:24;f:20;",
    "a:24;c:20;b:20;e:20;d:24;g:24;f:20;",
    "a:24;c:20;b:20;e:20;d:24;g:24;f:20;",
    "a:24;c:20;b:20;e:20;d:24;g:24;f:20;"
]

def parse_sizes(size_str):
    sizes = {}
    for item in size_str.split(';'):
        if item:
            key, value = item.split(':')
            sizes[key] = int(value)
    return sizes

def make_testcases():
    test_cases = []
    for expr, size_str in zip(list_format_strings, list_sizes):
        indices = expr.split('-')
        einsum_str = f"{indices[1]},{indices[2]}->{indices[0]}"
        sizes = parse_sizes(size_str)
        shapes = [tuple(sizes[dim] for dim in indices[1]), tuple(sizes[dim] for dim in indices[2])]
        test_cases.append((einsum_str, *shapes))
    return test_cases

# test_cases = make_testcases()
# print("No test cases: ", len(test_cases))
# for test_case in test_cases:
#     print(test_case)

def make_testcases_add_bd(bd):
    test_cases = []
    for expr, size_str in zip(list_format_strings, list_sizes):
        indices = expr.split('-')
        einsum_str = f"z{indices[1]},z{indices[2]}->z{indices[0]}"
        sizes = parse_sizes(size_str)
        shapes = [(bd,) + tuple(sizes[dim] for dim in indices[1]), (bd,) + tuple(sizes[dim] for dim in indices[2])]
        test_cases.append((einsum_str, *shapes))
    return test_cases

# Example usage
# bd = 10
# test_cases_bd = make_testcases_add_bd(bd)
# print("No test cases bd: ", len(test_cases_bd))
# for test_case in test_cases_bd:
#     print(test_case)


def make_only_bd_testcases(sizes):
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


def generate_einsum_input(test_case):
    einsum_str, shape1, shape2 = test_case
    tensor1 = np.random.rand(*shape1)
    tensor2 = np.random.rand(*shape2)
    return einsum_str, tensor1, tensor2

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
