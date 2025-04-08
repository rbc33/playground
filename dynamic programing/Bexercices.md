I'll add 14 more dynamic programming problems to your file, maintaining the proper markdown structure throughout the entire file.

````markdown:/Users/ric/code/py/playgroung/dynamic programing/exercices.md
# Dynamic Programming Problems Collection

## 1. Fibonacci Sequence

- **Problem**: Calculate the nth Fibonacci number.
- **Description**: The Fibonacci sequence is defined as F(n) = F(n-1) + F(n-2) with base cases F(0) = 0 and F(1) = 1.
- **Example**:
  - Input: n = 6
  - Output: 8 (The sequence is 0, 1, 1, 2, 3, 5, 8, ...)
- **Approach**: Use memoization to avoid recalculating values.
- **Test Cases**:

```python
test_cases = [
    0,    # 0
    1,    # 1
    2,    # 1
    5,    # 5
    10,   # 55
    20,   # 6765
    40    # 102334155
]
````

## 2. Climbing Stairs

- **Problem**: Count the number of ways to climb n stairs, taking 1 or 2 steps at a time.
- **Description**: You are climbing a staircase with n steps. Each time you can climb 1 or 2 steps. How many distinct ways can you reach the top?
- **Example**:
  - Input: n = 3
  - Output: 3 (1+1+1, 1+2, 2+1)
- **Approach**: Similar to Fibonacci, where ways(n) = ways(n-1) + ways(n-2).
- **Test Cases**:

```python
test_cases = [
    1,    # 1
    2,    # 2
    3,    # 3
    4,    # 5
    5,    # 8
    10,   # 89
    20    # 10946
]
```

## 3. Coin Change

- **Problem**: Find the minimum number of coins needed to make a given amount.
- **Description**: Given a list of coin denominations and a target amount, find the minimum number of coins needed to make up that amount.
- **Example**:
  - Input: coins = [1, 2, 5], amount = 11
  - Output: 3 (5 + 5 + 1)
- **Approach**: Build a DP table where dp[i] represents the minimum coins needed for amount i.
- **Test Cases**:

```python
test_cases = [
    ([1, 2, 5], 11),         # 3
    ([2], 3),                # -1 (impossible)
    ([1], 0),                # 0
    ([1, 3, 4, 5], 7),       # 2 (3 + 4)
    ([1, 2, 5, 10, 20, 50, 100, 200], 123)  # 5 (100 + 20 + 2 + 1)
]
```

## 4. Longest Increasing Subsequence

- **Problem**: Find the length of the longest strictly increasing subsequence.
- **Description**: Given an array of integers, find the length of the longest subsequence where all elements are in strictly increasing order.
- **Example**:
  - Input: [10, 9, 2, 5, 3, 7, 101, 18]
  - Output: 4 ([2, 3, 7, 101])
- **Approach**: For each position, consider all previous positions with smaller values.
- **Test Cases**:

```python
test_cases = [
    [10, 9, 2, 5, 3, 7, 101, 18],  # 4
    [0, 1, 0, 3, 2, 3],            # 4
    [7, 7, 7, 7, 7, 7, 7],         # 1
    [],                            # 0
    [1, 3, 6, 7, 9, 4, 10, 5, 6]   # 6
]
```

## 5. 0/1 Knapsack

- **Problem**: Maximize value of items in a knapsack with weight constraint.
- **Description**: Given weights and values of n items, put these items in a knapsack of capacity W to get the maximum total value.
- **Example**:
  - Input: values = [60, 100, 120], weights = [10, 20, 30], capacity = 50
  - Output: 220 (items with values 100 and 120)
- **Approach**: Build a 2D DP table where dp[i][w] represents max value with first i items and weight w.
- **Test Cases**:

```python
test_cases = [
    ([60, 100, 120], [10, 20, 30], 50),  # 220
    ([1, 4, 5, 7], [1, 3, 4, 5], 7),     # 9
    ([10, 40, 30, 50], [5, 4, 6, 3], 10), # 90
    ([3, 2, 4, 1], [4, 3, 2, 1], 5)      # 7
]
```

## 6. Edit Distance

- **Problem**: Find minimum operations to convert one string to another.
- **Description**: Given two strings, find the minimum number of operations (insert, delete, replace) required to convert string1 to string2.
- **Example**:
  - Input: word1 = "horse", word2 = "ros"
  - Output: 3 (delete 'h', replace 'r' with 'o', delete 'e')
- **Approach**: Use a 2D DP table where dp[i][j] is the edit distance between first i chars of word1 and first j chars of word2.
- **Test Cases**:

```python
test_cases = [
    ("horse", "ros"),       # 3
    ("intention", "execution"), # 5
    ("", "a"),              # 1
    ("abc", "abc"),         # 0
    ("pneumonoultramicroscopicsilicovolcanoconiosis", "ultramicroscopically") # 30
]
```

## 7. Longest Common Subsequence

- **Problem**: Find the length of the longest subsequence common to two sequences.
- **Description**: Given two strings, find the length of their longest common subsequence (not necessarily contiguous).
- **Example**:
  - Input: text1 = "abcde", text2 = "ace"
  - Output: 3 ("ace")
- **Approach**: Use a 2D DP table where dp[i][j] is the LCS of first i chars of text1 and first j chars of text2.
- **Test Cases**:

```python
test_cases = [
    ("abcde", "ace"),       # 3
    ("abc", "abc"),         # 3
    ("abc", "def"),         # 0
    ("", "abc"),            # 0
    ("bsbininm", "jmjkbkjkv") # 1
]
```

## 8. Maximum Subarray

- **Problem**: Find the contiguous subarray with the largest sum.
- **Description**: Given an integer array, find the contiguous subarray with the largest sum.
- **Example**:
  - Input: [-2,1,-3,4,-1,2,1,-5,4]
  - Output: 6 ([4,-1,2,1])
- **Approach**: Use Kadane's algorithm, tracking current sum and maximum sum.
- **Test Cases**:

```python
test_cases = [
    [-2,1,-3,4,-1,2,1,-5,4],  # 6
    [1],                      # 1
    [5,4,-1,7,8],             # 23
    [-1,-2,-3,-4],            # -1
    [-2,-3,4,-1,-2,1,5,-3]    # 7
]
```

## 9. House Robber

- **Problem**: Maximize the amount of money you can rob without alerting the police.
- **Description**: Given an array of house values, you cannot rob adjacent houses. Find the maximum amount you can rob.
- **Example**:
  - Input: [1,2,3,1]
  - Output: 4 (rob house 1 and 3)
- **Approach**: Use DP where dp[i] is the maximum amount that can be robbed up to house i.
- **Test Cases**:

```python
test_cases = [
    [1,2,3,1],        # 4
    [2,7,9,3,1],      # 12
    [2,1,1,2],        # 4
    [1,3,1,3,100],    # 103
    [1,2,3,4,5,6,7]   # 16
]
```

## 10. Unique Paths

- **Problem**: Count all possible paths from top-left to bottom-right of a grid.
- **Description**: A robot is located at the top-left corner of a m x n grid. It can only move right or down. How many possible unique paths are there to the bottom-right corner?
- **Example**:
  - Input: m = 3, n = 7
  - Output: 28
- **Approach**: Use a 2D DP table where dp[i][j] is the number of unique paths to cell (i,j).
- **Test Cases**:

```python
test_cases = [
    (3, 7),  # 28
    (3, 2),  # 3
    (7, 3),  # 28
    (1, 1),  # 1
    (10, 10) # 48620
]
```

## 11. Minimum Path Sum

- **Problem**: Find path with minimum sum from top-left to bottom-right of grid.
- **Description**: Given a grid filled with non-negative numbers, find a path from top left to bottom right, which minimizes the sum of all numbers along its path.
- **Example**:
  - Input: [[1,3,1],[1,5,1],[4,2,1]]
  - Output: 7 (path: 1→3→1→1→1)
- **Approach**: Use a 2D DP table where dp[i][j] is the minimum path sum to cell (i,j).
- **Test Cases**:

```python
test_cases = [
    [[1,3,1],[1,5,1],[4,2,1]],  # 7
    [[1,2,3],[4,5,6]],          # 12
    [[1]],                      # 1
    [[9,9,0],[9,0,0],[0,9,9]],  # 9
    [[1,2],[5,6],[1,1]]         # 8
]
```

## 12. Partition Equal Subset Sum

- **Problem**: Determine if array can be partitioned into two equal sum subsets.
- **Description**: Given a non-empty array of positive integers, determine if it can be partitioned into two subsets such that the sum of elements in both subsets is equal.
- **Example**:
  - Input: [1,5,11,5]
  - Output: true ([1,5,5] and [11])
- **Approach**: Use DP to find if a subset with sum = totalSum/2 exists.
- **Test Cases**:

```python
test_cases = [
    [1,5,11,5],       # true
    [1,2,3,5],        # false
    [1,1,1,1],        # true
    [1,2,5],          # false
    [3,3,3,4,5]       # true
]
```

## 13. Longest Palindromic Subsequence

- **Problem**: Find the length of the longest palindromic subsequence.
- **Description**: Given a string, find the length of its longest palindromic subsequence (not necessarily contiguous).
- **Example**:
  - Input: "bbbab"
  - Output: 4 ("bbbb")
- **Approach**: Use a 2D DP table where dp[i][j] is the LPS from index i to j.
- **Test Cases**:

```python
test_cases = [
    "bbbab",          # 4
    "cbbd",           # 2
    "a",              # 1
    "aabaa",          # 5
    "character"       # 3
]
```

## 14. Word Break

- **Problem**: Determine if a string can be segmented into dictionary words.
- **Description**: Given a string and a dictionary of words, determine if the string can be segmented into a space-separated sequence of dictionary words.
- **Example**:
  - Input: s = "leetcode", wordDict = ["leet", "code"]
  - Output: true ("leet code")
- **Approach**: Use DP where dp[i] indicates if the first i characters can be segmented.
- **Test Cases**:

```python
test_cases = [
    ("leetcode", ["leet", "code"]),                # true
    ("applepenapple", ["apple", "pen"]),           # true
    ("catsandog", ["cats", "dog", "sand", "and", "cat"]), # false
    ("aaaaaaa", ["aaaa", "aaa"]),                  # true
    ("goalspecial", ["go", "goal", "goals", "special"]) # true
]
```

## 15. Rod Cutting

- **Problem**: Maximize profit by cutting a rod into pieces.
- **Description**: Given a rod of length n and prices for different lengths, find the maximum value obtainable by cutting up the rod and selling the pieces.
- **Example**:
  - Input: length = 8, prices = [1,5,8,9,10,17,17,20]
  - Output: 22 (cut into lengths of 2 and 6)
- **Approach**: Use DP where dp[i] is the maximum value obtainable from a rod of length i.
- **Test Cases**:

```python
test_cases = [
    (8, [1,5,8,9,10,17,17,20]),  # 22
    (4, [1,5,8,9,10]),           # 10
    (1, [1]),                    # 1
    (10, [1,5,8,9,10,17,17,20,24,30]), # 30
    (5, [2,5,7,8,10])            # 12
]
```

```

```
