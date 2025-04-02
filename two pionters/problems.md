# Two Pointer Problems Collection

## 11. Valid Palindrome II

- **Problem**: Check if string can become palindrome by removing at most one character
- **Example**:
  - Input: "abca"
  - Output: true (remove 'c')
- **Key Idea**: Compare chars from both ends, allow one mismatch

```python
test_cases = [
    "abca",           # true (remove 'c')
    "aba",            # true (already palindrome)
    "abcd",           # false (need more than one removal)
    "a",              # true (single char is palindrome)
    "",               # true (empty string is palindrome)
    "aa",             # true (already palindrome)
    "aaa",            # true (already palindrome)
    "deeee",          # true (remove 'd')
    "cbbcc",          # true (remove last 'c')
    "abcddcbea"       # true (remove 'e')
]
```

## 12. Three Sum

- **Problem**: Find all unique triplets that sum to zero
- **Example**:
  - Input: [-1,0,1,2,-1,-4]
  - Output: [[-1,-1,2],[-1,0,1]]
- **Key Idea**: Sort array, fix one number and use two pointers for remaining sum
- **Test Cases**:

```python
test_cases = [
    [-1,0,1,2,-1,-4],        # [[-1,-1,2],[-1,0,1]]
    [],                      # []
    [0],                     # []
    [0,0,0],                # [[0,0,0]]
    [-2,0,1,1,2],           # [[-2,0,2],[-2,1,1]]
    [-4,-2,-2,-2,0,1,2,2,2,3,3,4,4,6,6],  # Multiple duplicates
    [-1,-1,-1,0,0,0,1,1,1],  # Test with repeated numbers
    [1,2,3,4,5],            # No solution
    [-5,-4,-3,-2,-1],       # No solution (all negative)
    [1000000000,1000000000,1000000000],  # Large numbers
]
```

## 13. Three Sum Closest

- **Problem**: Find sum of three integers closest to target
- **Example**:
  - Input: nums=[-1,2,1,-4], target=1
  - Output: 2 (-1+2+1)
- **Key Idea**: Sort and minimize difference with target
- **Test Cases**:

```python
test_cases = [
    ([-1,2,1,-4], 1),              # 2 (-1+2+1)
    ([0,0,0], 1),                  # 0 (0+0+0)
    ([1,1,1,1], 0),                # 3 (1+1+1)
    ([-100,-98,2,89], -101),       # -196 (-100+-98+2)
    ([4,0,5,-5,3,3,0,0], -2),      # -2 (-5+0+3)
    ([1,2,4,8,16,32,64,128], 82),  # 82 (2+16+64)
    ([1], 100),                    # None (array too small)
    ([], 5),                       # None (empty array)
    ([-5,-5,-4,0,0,3,3,4,5], 0),   # 0 (-4+0+4)
    ([1000000000,1000000000,1000000000], 3000000000)  # Edge case with large numbers
]
```

## 14. Minimum Window Substring

- **Problem**: Find smallest substring containing all characters of target string
- **Example**:
  - Input: s="ADOBECODEBANC", t="ABC"
  - Output: "BANC"
- **Key Idea**: Sliding window with character frequency map

```python
test_cases = [
    ("ADOBECODEBANC", "ABC"),     # "BANC"
    ("a", "a"),                   # "a"
    ("a", "aa"),                  # ""
    ("", ""),                     # ""
    ("AAAAAAAAAA", "A"),         # "A"
    ("ABCDEFGHIJK", "ZX"),       # ""
    ("aaaaabaab", "abb"),        # "aab"
]
```

## 15. 4Sum

Medium
Topics
Companies
Given an array nums of n integers, return an array of all the unique quadruplets [nums[a], nums[b], nums[c], nums[d]] such that:

0 <= a, b, c, d < n
a, b, c, and d are distinct.
nums[a] + nums[b] + nums[c] + nums[d] == target
You may return the answer in any order.

Example 1:

Input: nums = [1,0,-1,0,-2,2], target = 0
Output: [[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]
Example 2:

Input: nums = [2,2,2,2,2], target = 8
Output: [[2,2,2,2]]

Constraints:

1 <= nums.length <= 200
-109 <= nums[i] <= 109
-109 <= target <= 109

## 16. Longest Substring Without Repeating Characters

- **Problem**: Find longest substring without duplicate characters
- **Example**:
  - Input: "abcabcbb"
  - Output: 3 ("abc")
- **Key Idea**: Sliding window with character set
  test_cases = [
  "abcabcbb", # 3 ("abc")
  "bbbbb", # 1 ("b")
  "pwwkew", # 3 ("wke")
  "", # 0 (empty string)
  "a", # 1 (single character)
  "aab", # 2 ("ab")
  "dvdf", # 3 ("vdf")
  "abba", # 2 ("ab" or "ba")
  "tmmzuxt", # 5 ("mzuxt")
  " ", # 1 (single space)
  "au", # 2 ("au")
  "abcdefghijklmnop", # 16 (all unique)
  "aaaabc", # 3 ("abc")
  ]

## 17. Merge Sorted Arrays

- **Problem**: Merge two sorted arrays into first array
- **Example**:
  - Input: [1,2,3,0,0,0], [2,5,6]
  - Output: [1,2,2,3,5,6]
- **Key Idea**: Fill from end to avoid overwriting
  test_cases = [
  ([1,2,3,0,0,0], [2,5,6], 3, 3), # [1,2,2,3,5,6]
  ([1], [], 1, 0), # [1]
  ([0], [1], 0, 1), # [1]
  ([4,5,6,0,0,0], [1,2,3], 3, 3), # [1,2,3,4,5,6]
  ([1,2,4,5,6,0], [3], 5, 1), # [1,2,3,4,5,6]
  ([0,0,0], [2,5,6], 0, 3), # [2,5,6]
  ([1,2,3,0,0,0,0], [2,5,6,7], 3, 4), # [1,2,2,3,5,6,7]
  ([2,0], [1], 1, 1), # [1,2]
  ([4,0,0,0], [1,2,3], 1, 3), # [1,2,3,4]
  ([0], [], 0, 0) # [0]
  ]

## 18. Remove Duplicates II

- **Problem**: Remove duplicates allowing at most two occurrences
- **Example**:
  - Input: [1,1,1,2,2,3]
  - Output: [1,1,2,2,3]
- **Key Idea**: Track count of current number

## 19. Partition Labels

- **Problem**: Partition string so each letter appears in at most one part
- **Example**:
  - Input: "ababcbacadefegdehijhklij"
  - Output: [9,7,8]
- **Key Idea**: Track last occurrence of each char

## 20. Push Dominoes

- **Problem**: Simulate falling dominoes
- **Example**:
  - Input: ".L.R....L"
  - Output: "LL.RRRLLL"
- **Key Idea**: Compare forces from both directions

## 21. Find K Closest Elements

- **Problem**: Find k closest numbers to target value x in sorted array
- **Example**:
  - Input: arr=[1,2,3,4,5], k=4, x=3
  - Output: [1,2,3,4]
- **Key Idea**: Binary search + two pointers expansion

## 22. Container With Most Water

- **Problem**: Find two lines that together with x-axis forms container holding most water
- **Example**:
  - Input: height=[1,8,6,2,5,4,8,3,7]
  - Output: 49
- **Key Idea**: Move pointer of smaller height inward

## 23. Next Permutation

- **Problem**: Find next lexicographically greater permutation
- **Example**:
  - Input: [1,2,3]
  - Output: [1,3,2]
- **Key Idea**: Find first decreasing element from right

## 24. Sort Colors

- **Problem**: Sort array of 0s, 1s, and 2s in-place
- **Example**:
  - Input: [2,0,2,1,1,0]
  - Output: [0,0,1,1,2,2]
- **Key Idea**: Three pointers (Dutch national flag)

## 25. Boats to Save People

- **Problem**: Find minimum number of boats to save people with weight limit
- **Example**:
  - Input: people=[1,2], limit=3
  - Output: 1
- **Key Idea**: Match heaviest with lightest possible

## 26. Longest Mountain

- **Problem**: Find length of longest mountain subarray
- **Example**:
  - Input: [2,1,4,7,3,2,5]
  - Output: 5 ([1,4,7,3,2])
- **Key Idea**: Track increasing and decreasing sequences

## 27. Subarray Product Less Than K

- **Problem**: Count subarrays with product less than k
- **Example**:
  - Input: nums=[10,5,2,6], k=100
  - Output: 8
- **Key Idea**: Sliding window with product

## 28. Find All Anagrams

- **Problem**: Find all anagrams of pattern in string
- **Example**:
  - Input: s="cbaebabacd", p="abc"
  - Output: [0,6]
- **Key Idea**: Sliding window with character count

## 29. Shortest Word Distance

- **Problem**: Find minimum distance between two words in array
- **Example**:
  - Input: words=["practice", "makes", "perfect", "coding"], word1="coding", word2="practice"
  - Output: 3
- **Key Idea**: Track last positions of both words

## 30. Intersection of Two Arrays

- **Problem**: Find intersection of two sorted arrays
- **Example**:
  - Input: nums1=[1,2,2,1], nums2=[2,2]
  - Output: [2]
- **Key Idea**: Two pointers on sorted arrays

## 31. String Compression

- **Problem**: Compress string using counts of repeated characters
- **Example**:
  - Input: ["a","a","b","b","c","c","c"]
  - Output: ["a","2","b","2","c","3"]
- **Key Idea**: Read and write pointers

## 32. Remove Element

- **Problem**: Remove all occurrences of value in-place
- **Example**:
  - Input: nums=[3,2,2,3], val=3
  - Output: 2, nums=[2,2,_,_]
- **Key Idea**: Two pointers for reading and writing

## 33. Move Zeroes

- **Problem**: Move all zeroes to end maintaining relative order
- **Example**:
  - Input: [0,1,0,3,12]
  - Output: [1,3,12,0,0]
- **Key Idea**: Two pointers for non-zero and position

## 34. Reverse Words in String

- **Problem**: Reverse words in string maintaining spaces
- **Example**:
  - Input: "the sky is blue"
  - Output: "blue is sky the"
- **Key Idea**: Reverse whole string then reverse each word

## 35. Find Pair with Given Difference

- **Problem**: Find if there exists a pair with given difference
- **Example**:
  - Input: arr=[5,20,3,2,50,80], diff=78
  - Output: true (80-2=78)
- **Key Idea**: Two pointers after sorting

Would you like me to add test cases for each problem as well?
