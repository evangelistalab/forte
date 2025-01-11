#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import forte

import math


def calculate_power_of_10_and_rank_histogram(operators, max_rank):
    # Initialize the histogram as a dictionary of dictionaries
    histogram = {}
    for sqop, c in operators:
        rank = sqop.count()
        power = math.floor(math.log10(abs(c))) if c != 0 else None

        if power not in histogram:
            histogram[power] = [0] * (max_rank + 2)  # +2 for rank 0 and ranks > max_rank

        if rank > max_rank:
            histogram[power][-1] += 1  # Increment the count for ranks > max_rank
        else:
            histogram[power][rank] += 1  # Increment the count for the specific rank

    return histogram


# def print_histogram2(histogram, max_rank):
#     # Print header
#     header = ["Power/Rank"] + [str(rank) for rank in range(max_rank + 1)] + [">" + str(max_rank)]
#     print("{:<12}".format(header[0]), end="")
#     for head in header[1:]:
#         print("{:>10}".format(head), end="")
#     print()


#     for power in sorted(histogram):
#         print("{:<12}".format(f"10^{power}"), end="")
#         for count in histogram[power]:
#             print("{:>10}".format(count), end="")
#         print()
def print_histogram2(histogram, max_rank):
    # Determine which columns (ranks) to print
    columns_to_print = [False] * (max_rank + 2)  # +2 for rank 0 and ranks > max_rank
    for power, counts in histogram.items():
        for rank, count in enumerate(counts):
            if count > 0:
                columns_to_print[rank] = True  # Mark column to be printed

    columns_to_print[-1] = True  # Always print the "Total" column

    # Print header with rank titles, skipping those with all zeros, and add "Total" column
    header_titles = [str(rank) for rank, to_print in enumerate(columns_to_print[:-1]) if to_print] + [
        f">{max_rank}",
        "Total",
    ]
    header = f"{'Power/Rank':<12}" + "".join([f"{title:>8}" for title in header_titles])
    print(header)

    # Print each power of 10, skipping columns with all zeros, and add total column
    for power in sorted(histogram):
        row_total = sum(histogram[power])  # Calculate the total for the row
        row_data = [f"{count:>8}" for rank, count in enumerate(histogram[power]) if columns_to_print[rank]]
        row = f"{'10^' + str(power):<12}" + "".join(row_data) + f"{row_total:>8}"
        print(row)


def calculate_power_of_10_histogram(floats):
    # Calculate powers of 10
    power_counts = {}
    for number in floats:
        if number == 0:
            continue  # Skip zeros as log10(0) is undefined
        power = math.floor(math.log10(abs(number)))
        if power in power_counts:
            power_counts[power] += 1
        else:
            power_counts[power] = 1

    return power_counts


def print_histogram(power_counts):
    for power in sorted(power_counts):
        print(f"10^{power}: {power_counts[power]}")


def make_operator_histogram(A, max_rank):
    histogram = calculate_power_of_10_and_rank_histogram(A, max_rank)
    print_histogram2(histogram, max_rank)


# def make_operator_histogram(A):
#     cvals = []
#     for i in range(A.size()):
#         sqop, c = A.term(i)
#         rank = sqop.count()
#         cvals.append(c)

#     power_counts = calculate_power_of_10_histogram(cvals)
#     print_histogram(power_counts)


def compute_st_taylor(O, A):
    stO = forte.SparseOperator()
    stO += O
    C = forte.SparseOperator()
    C += O
    for i in range(1, 30):
        C = (1 / i) * C.commutator(A)
        stO += C
        if C.norm() < 1e-16:
            print(f"  A norm: {C.norm()} after {i} iterations")
            break
    return stO


def test_sparse_operator_transform_cc_1():
    n = 10
    o = 5

    H = forte.SparseOperator()
    # one-body operator
    for i in range(n):
        for j in range(n):
            H += forte.sparse_operator(f"[{i}a+ {j}a-]", -0.5 / (abs(i - j) ** 4 + 0.5))

    # two-body operator
    for i in range(n):
        for j in range(i + 1, n):
            for a in range(n):
                for b in range(a + 1, n):
                    H += forte.sparse_operator(f"[{a}a+ {b}a+ {j}a- {i}a-]", 0.1 / (abs(i + j - a - b) ** 2 + 0.5))

    Hbar = forte.SparseOperator()
    Hbar += H

    T = forte.SparseOperatorList()
    for i in range(o):
        for a in range(o, n):
            T.add(f"[{a}a+ {i}a-]", 0.1 / (abs(a - i) ** 2 + 0.5))

    # double excitation
    for i in range(o):
        for j in range(i + 1, o):
            for a in range(o, n):
                for b in range(a + 1, n):
                    T.add(f"[{a}a+ {b}a+ {j}a- {i}a-]", 1 / (abs(a + b - i - j) ** 2 + 0.5))

    Top = T.to_operator()

    ref = forte.SparseState({forte.det("2" * o): 1.0})
    print(f"{ref = }")

    print(f"{len(H) = }")
    print(f"{len(T) = }")
    forte.fact_trans_lin(Hbar, T, screen_thresh=1.0e-10)
    Hbar_ref = forte.apply_op(Hbar, ref)
    E1 = forte.overlap(ref, Hbar_ref)
    print(f"{E1 = } (transform)")
    print(f"{len(Hbar) = }")

    make_operator_histogram(Hbar, 10)

    exp_op = forte.SparseExp()
    expT_ref = exp_op.apply_op(Top, ref)
    H_expT_ref = forte.apply_op(H, expT_ref)
    E2 = forte.overlap(ref, H_expT_ref)
    print(f"{E2 = } (exponential)")
    print(f"{len(expT_ref) = }")

    # assert the two energies are close
    assert abs(E1 - E2) < 1e-12


def test_sparse_operator_transform_ucc_1():

    print("\nTesting UCC transformation")
    n = 8
    o = 4

    H = forte.SparseOperator()
    # one-body operator
    for i in range(n):
        for j in range(n):
            H += forte.sparse_operator(f"[{i}a+ {j}a-]", -0.5 / (abs(i - j) ** 4 + 0.5))

    # two-body operator
    for i in range(n):
        for j in range(i + 1, n):
            for a in range(n):
                for b in range(a + 1, n):
                    H += forte.sparse_operator(f"[{a}a+ {b}a+ {j}a- {i}a-]", 0.1 / (abs(i + j - a - b) ** 2 + 0.5))

    Hbar = forte.SparseOperator()
    Hbar += H

    T = forte.SparseOperatorList()
    for i in range(o):
        for a in range(o, n):
            T.add(f"[{a}a+ {i}a-]", 0.1 / (abs(a - i) ** 2 + 0.5))

    # double excitation
    for i in range(o):
        for j in range(i + 1, o):
            for a in range(o, n):
                for b in range(a + 1, n):
                    T.add(f"[{a}a+ {b}a+ {j}a- {i}a-]", 1 / (abs(a + b - i - j) ** 2 + 0.5))

    # Top = T.to_operator()

    ref = forte.SparseState({forte.det("+" * o): 1.0})
    print(f"{ref = }")

    print(f"{len(H) = }")
    print(f"{len(T) = }")
    forte.fact_unitary_trans_antiherm(Hbar, T, reverse=True, screen_thresh=1.0e-10)

    Hbar_ref = forte.apply_op(Hbar, ref)
    E = forte.overlap(ref, Hbar_ref)
    print(f"{E = } (exact transform)")
    Etr = E

    exp_op = forte.SparseFactExp()

    expT_ref = exp_op.apply_antiherm(T, ref)
    H_expT_ref = forte.apply_op(H, expT_ref)
    E = forte.overlap(expT_ref, H_expT_ref)

    print(f"{E = } (wave function)")
    print(f"{E - Etr = }")
    print(f"{len(Hbar) = }")
    print(f"{len(expT_ref) = }")
    print(f"{len(H_expT_ref) = }")

    # make_operator_histogram(Hbar, 10)

    # assert the two energies are close
    assert abs(E - Etr) < 1e-9


if __name__ == "__main__":
    test_sparse_operator_transform_cc_1()
    test_sparse_operator_transform_ucc_1()
