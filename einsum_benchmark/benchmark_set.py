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
    for expr, size_str in zip(list_format_strings, list_sizes):
        indices = expr.split('-')
        einsum_str = f"{indices[1]},{indices[2]}->{indices[0]}"
        sizes = parse_sizes(size_str)
        shapes = [tuple(sizes[dim] for dim in indices[1]), tuple(sizes[dim] for dim in indices[2])]
        test_cases.append((einsum_str, *shapes))
    return test_cases

def get_testcases_with_bd(bd):
    test_cases = []
    for expr, size_str in zip(list_format_strings, list_sizes):
        indices = expr.split('-')
        einsum_str = f"z{indices[1]},z{indices[2]}->z{indices[0]}"
        sizes = parse_sizes(size_str)
        shapes = [(bd,) + tuple(sizes[dim] for dim in indices[1]), (bd,) + tuple(sizes[dim] for dim in indices[2])]
        test_cases.append((einsum_str, *shapes))
    return test_cases

def make_bd_only_testcases(sizes):
    # tupel sizes: (bd, a_rows, a_cols, b_cols)
    test_cases = []
    for size in sizes:
        bd, a_rows, a_cols, b_cols = size
        einsum_str = f"zab,zbc->zac"
        shapes = [(bd, a_rows, a_cols), (bd, a_cols, b_cols)]
        test_cases.append((einsum_str, *shapes))
    return test_cases

def generate_einsum_input(test_case):
    einsum_str, shape1, shape2 = test_case

    tensor1 = np.random.rand(*shape1)
    tensor2 = np.random.rand(*shape2)
    return einsum_str, tensor1, tensor2
