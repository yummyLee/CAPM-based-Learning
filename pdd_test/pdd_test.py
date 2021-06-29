import sys
import numpy as np


def p_matrix(n):
    matrix = [-1] * n
    for i in range(n):
        matrix[i] = [-1] * n
    xy_mid = int(n / 2)

    for i in range(n):
        for j in range(n):
            if j > i and j >= xy_mid and i + j < n:
                matrix[i][j] = 1
            if j > i and j < xy_mid and i < xy_mid:
                matrix[i][j] = 2
            if j < i and i < xy_mid and i < xy_mid:
                matrix[i][j] = 3
            if j < i and j < xy_mid and i >= xy_mid:
                matrix[i][j] = 4
            if j < i and j < xy_mid and i + j >= n:
                matrix[i][j] = 5
            if j < i and j >= xy_mid and i + j >= n:
                matrix[i][j] = 6
            if j > i and i >= xy_mid and i + j >= n:
                matrix[i][j] = 7
            if j > i and j > xy_mid and i < xy_mid and i + j >= n:
                matrix[i][j] = 8

    for i in range(n):
        if n % 2 == 1:
            matrix[i][xy_mid] = 0
            matrix[xy_mid][i] = 0
        matrix[i][i] = 0
        matrix[i][n - i - 1] = 0

    for i in range(n):
        for j in range(n):
            print(matrix[i][j], end='')
            if j < n - 1:
                print(' ', end='')
            elif j == n - 1 and i < n - 1:
                print()


if __name__ == "__main__":
    n = sys.stdin.readline().strip()
    n = int(n)
    # n = 11
    p_matrix(n)
