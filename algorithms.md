[Algorithms](#algorithms)

1. [General Tips](#general-tips)
1. [Binary Search](#binary-search)
1. [Topological Sort](#topological-sort)
1. [Sliding Window](#sliding-window)
1. [Tree Tricks](#tree-tricks)
1. [Binary Search Tree](#binary-search-tree)
1. [Anagrams](#anagrams)
1. [Dynamic Programming](#dynamic-programming)
1. [Cyclic Sort](#cyclic-sort)
1. [Quick Sort](#quick-sort)
1. [Merge Sort](#merge-sort)
1. [Merge K Sorted Arrays](#merge-arrays)
1. [Linked List](#linked-list)
1. [Convert Base](#convert-base)
1. [Parenthesis](#parenthesis)
1. [Max Profit Stock](#max-profit-stock)
1. [Shift Array Right](#shift-array-right)
1. [Continuous Subarrays with Sum k ](#continuous-subarrays-with-sum-k)
1. [Events](#events)
1. [Merge Meetings](#merge-meetings)
1. [Trie](#trie)
1. [Kadane's Algorithm - Max subarray sum](#kadane)
1. [Union Find/DSU](#union-find)
1. [Fast Power](#fast-power)
1. [Fibonacci Golden](#fibonacci-golden)
1. [Basic Calculator](#basic-calculator)
1. [Reverse Polish](#reverse-polish)
1. [Resevior Sampling](#resevior-sampling)
1. [Candy Crush](#candy-crush)


# Algorithms

## General Tips

- Get all info
- Debug example, is it a special case?
- Brute Force
  - Get to brute-force solution as soon as possible. State runtime and then optimize, don't code yet
- Optimize
  - Look for unused info
  - Solve it manually on example, then reverse engineer thought process
  - Space vs time, hashing
  - BUDS (Bottlenecks, Unnecessary work, Duplication)
- Walk through approach
- Code
- Test
  - Start small
  - Hit edge cases

## Binary Search

```python
def firstBadVersion(self, n):
    l, r = 0, n
    while l < r:
        m = l + (r-l) // 2
        if isBadVersion(m):
            r = m
        else:
            l = m + 1
    return l
```

```python
"""
12345678
FFTTTTTT
"""
def mySqrt(self, x: int) -> int:
  def condition(value, x) -> bool:
    return value * value > x

  if x == 1:
    return 1

  left, right = 1, x
  while left < right:
    mid = left + (right-left) // 2
    if condition(mid, x):
      right = mid
    else:
      left = mid + 1

  return left - 1
```

[binary search](https://leetcode.com/discuss/general-discussion/786126/python-powerful-ultimate-binary-search-template-solved-many-problems)

## Binary Search Tree

Use values to detect if number is missing

```python
def isCompleteTree(self, root: TreeNode) -> bool:
    self.total = 0
    self.mx = float('-inf')
    def dfs(node, cnt):
        if node:
            self.total += 1
            self.mx = max(self.mx, cnt)
            dfs(node.left, (cnt*2))
            dfs(node.right, (cnt*2)+1)
    dfs(root, 1)
    return self.total == self.mx
```

Get a range sum of values

```python
def rangeSumBST(self, root: TreeNode, L: int, R: int) -> int:
    self.total = 0
    def helper(node):
        if node is None:
            return 0
        if L <= node.val <= R:
            self.total += node.val
        if node.val > L:
            left = helper(node.left)
        if node.val < R:
            right = helper(node.right)
    helper(root)
    return self.total
```

Check if valid

```python
def isValidBST(self, root: TreeNode) -> bool:
    if not root:
        return True
    stk = [(root, float(-inf), float(inf))]
    while stk:
        node, floor, ceil = stk.pop()
        if node:
            if node.val >= ceil or node.val <= floor:
                return False
            stk.append((node.right, node.val, ceil))
            stk.append((node.left, floor, node.val))
    return True
```

## Topological Sort

[Kahn's algorithm](https://www.geeksforgeeks.org/all-topological-sorts-of-a-directed-acyclic-graph/), detects cycles through degrees and needs all the nodes represented to work

1. Initialize vertices as unvisited
1. Pick vertex with zero indegree, append to result, decrease indegree of neighbors
1. Now repeat for neighbors, resulting list is sorted by source -> dest

If cycle, then degree of nodes in cycle will not be 0 since there
is no origin

```python
def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
    # Kahns algorithm, topological sort
    adj = collections.defaultdict(list)
    degree = collections.Counter()

    for dest, orig in prerequisites:
        adj[orig].append(dest)
        degree[dest] += 1

    bfs = [c for c in range(numCourses) if degree[c] == 0]

    for o in bfs:
        for d in adj[o]:
            degree[d] -= 1
            if degree[d] == 0:
                bfs.append(d)

    return len(bfs) == numCourses
```

```python
def alienOrder(self, words: List[str]) -> str:
    nodes = set("".join(words))
    adj = collections.defaultdict(list)
    degree = collections.Counter(nodes)

    for w1, w2 in zip(words, words[1:]):
        for c1, c2 in zip(w1, w2):
            if c1 != c2:
                adj[c1].append(c2)
                degree[c2] += 1
                break
        else:
            if len(w1) > len(w2):
                return ""

    stk = list(filter(lambda x: degree[x]==1, degree.keys()))

    ans = []
    while stk:
        node = stk.pop()
        ans.append(node)
        for nei in adj[node]:
            degree[nei] -= 1
            if degree[nei] == 1:
                stk.append(nei)

    return "".join(ans) * (set(ans) == nodes)
```

## Sliding Window

1. Have a counter or hash-map to count specific array input and keep on increasing the window toward right using outer loop.
1. Have a while loop inside to reduce the window side by sliding toward right. Movement will be based on constraints of problem.
1. Store the current maximum window size or minimum window size or number of windows based on problem requirement.

### Typical Problem Clues:

1. Get min/max/number of satisfied sub arrays
1. Return length of the subarray with max sum/product
1. Return max/min length/number of subarrays whose sum/product equals K

Can require [2 or 3 pointers to solve](https://medium.com/algorithms-and-leetcode/magic-solution-to-leetcode-problems-sliding-window-algorithm-891e3d60bf89)

```python
    def slidingWindowTemplate(self, s: str):
        #init a collection or int value to save the result according the question.
        rtn = []

        # create a hashmap to save the Characters of the target substring.
        # (K, V) = (Character, Frequence of the Characters)
        hm = {}

        # maintain a counter to check whether match the target string as needed
        cnt = collections.Counter(s)

        # Two Pointers: begin - left pointer of the window; end - right pointer of the window if needed
        l = r = 0

        # loop at the begining of the source string
        for r, c in enumerate(s):

            if c in hm:
                l = max(hm[c]+1, l) # +/- 1 or set l to index, max = never move l left

            # update hm
            hm[c] = r

            # increase l pointer to make it invalid/valid again
            while cnt == 0: # counter condition
                cnt[c] += 1  # modify counter if needed

            # Save result / update min/max after loop is valid
            rtn = max(rtn, r-l+1)

        return rtn
```

```python
def fruits_into_baskets(fruits):
  maxCount, j = 0, 0
  ht = {}

  for i, c in enumerate(fruits):
    if c in ht:
      ht[c] += 1
    else:
      ht[c] = 1

    if len(ht) <= 2:
      maxCount = max(maxCount, i-j+1)
    else:
      jc = fruits[j]
      ht[jc] -= 1
      if ht[jc] <= 0:
        del ht[jc]
      j += 1

  return maxCount
```

## Greedy

Make the optimal [choice](https://brilliant.org/wiki/greedy-algorithm/) at each step.

[Increasing Triplet Subsequence](https://leetcode.com/problems/increasing-triplet-subsequence/), true if i < j < k

```python
def increasingTriplet(self, nums: List[int]) -> bool:
    l = m = float('inf')

    for n in nums:
        if n <= l:
            l = n
        elif n <= m:
            m = n
        else:
            return True

    return False
```

## Tree Tricks

Bottom up solution with arguments for min, max

```python
def maxAncestorDiff(self, root: TreeNode) -> int:
    if not root:
        return 0
    self.ans = 0
    def dfs(node, minval, maxval):
        if not node:
            self.ans = max(self.ans, abs(maxval - minval))
            return
        dfs(node.left, min(node.val, minval), max(node.val, maxval))
        dfs(node.right, min(node.val, minval), max(node.val, maxval))
    dfs(root, float('inf'), float('-inf'))
    return self.ans
```

Building a path through a tree

```python
def binaryTreePaths(self, root: TreeNode) -> List[str]:
    rtn = []
    if root is None: return []
    stk = [(root, str(root.val))]
    while stk:
        node, path = stk.pop()
        if node.left is None and node.right is None:
            rtn.append(path)
        if node.left:
            stk.append((node.left, path + "->" + str(node.left.val)))
        if node.right:
            stk.append((node.right, path + "->" + str(node.right.val)))
    return rtn
```

Using return value to sum

```python
def diameterOfBinaryTree(self, root: TreeNode) -> int:
    self.mx = 0
    def dfs(node):
        if node:
            l = dfs(node.left)
            r = dfs(node.right)
            total = l + r
            self.mx = max(self.mx, total)
            return max(l, r) + 1
        else:
            return 0
    dfs(root)
    return self.mx
```

Change Tree to Graph

```python
def distanceK(self, root: TreeNode, target: TreeNode, K: int) -> List[int]:
    adj = collections.defaultdict(list)

    def dfsa(node):
        if node.left:
            adj[node].append(node.left)
            adj[node.left].append(node)
            dfsa(node.left)
        if node.right:
            adj[node].append(node.right)
            adj[node.right].append(node)
            dfsa(node.right)

    dfsa(root)

    def dfs(node, prev, d):
        if node:
            if d == K:
                rtn.append(node.val)
            else:
                for nei in adj[node]:
                    if nei != prev:
                        dfs(nei, node, d+1)

    rtn = []
    dfs(target, None, 0)
    return rtn
```

## Anagrams

Subsection of sliding window, solve with Counter Dict

i.e.
abc = bca != eba
111 111 111

```python
def isAnagram(self, s: str, t: str) -> bool:
    sc = collections.Counter(s)
    st = collections.Counter(t)
    if sc != st:
        return False
    return True
```

Sliding Window version (substring)

```python
def findAnagrams(self, s: str, p: str) -> List[int]:
    cntP = collections.Counter(p)
    cntS = collections.Counter()
    P = len(p)
    S = len(s)
    if P > S:
        return []
    ans = []
    for i, c in enumerate(s):
        cntS[c] += 1
        if i >= P:
            if cntS[s[i-P]] > 1:
                cntS[s[i-P]] -= 1
            else:
                del cntS[s[i-P]]
        if cntS == cntP:
            ans.append(i-(P-1))
    return ans
```

## Dynamic Programming

1. [dynamic programming](https://leetcode.com/discuss/general-discussion/458695/Dynamic-Programming-Patterns)

```python
def coinChange(self, coins: List[int], amount: int) -> int:
  MAX = float('inf')
  dp =  [MAX] * (amount + 1)
  dp[0] = 0
  for c in coins:
    for a in range(c, amount+1):
      dp[a] =  min(dp[a], dp[a-c]+1)
  return dp[amount] if dp[amount] != MAX else -1
```

Classic DP grid, longest common subsequence

```python
def longestCommonSubsequence(self, text1: str, text2: str) -> int:
    Y = len(text2)+1
    X = len(text1)+1
    dp = [[0] * Y for _ in range(X)]
    # [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    for i, c in enumerate(text1):
        for j, d in enumerate(text2):
            if c == d:
                dp[i + 1][j + 1] = 1 + dp[i][j]
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    return dp[-1][-1]
# [[0,0,0,0],[0,1,1,1],[0,1,1,1],[0,1,2,2],[0,1,2,2],[0,1,2,3]]
# abcde
# "ace"
```

## Cyclic Sort

1. Useful algo when sorting in place

```python
# if my number is equal to my index, i+1
# if my number is equal to this other number, i+1 (dups)
# else swap
def cyclic_sort(nums):
  i = 0
  while i < len(nums):
    j = nums[i] - 1
    if nums[i] != nums[j]:
      nums[i], nums[j] = nums[j], nums[i]
    else:
      i += 1
  return nums
```

## Quick Sort

1. Can be modified for divide in conquer problems

```python
def quickSort(array):
	def sort(arr, l, r):
		if l < r:
			p = part(arr, l, r)
			sort(arr, l, p-1)
			sort(arr, p+1, r)

	def part(arr, l, r):
		pivot = arr[r]
		a = l
		for i in range(l,r):
			if arr[i] < pivot:
				arr[i], arr[a] = arr[a], arr[i]
				a += 1
		arr[r], arr[a] = arr[a], arr[r]
		return a

	sort(array, 0, len(array)-1)
	return array
```

## Merge Sort

```python
from collections import deque
def mergeSort(array):
    def sortArray(nums):
        if len(nums) > 1:
            mid = len(nums)//2
            l1 = sortArray(nums[:mid])
            l2 = sortArray(nums[mid:])
            nums = sort(l1,l2)
        return nums

    def sort(l1,l2):
        result = []
        l1 = deque(l1)
        l2 = deque(l2)
        while l1 and l2:
            if l1[0] <= l2[0]:
                result.append(l1.popleft())
            else:
                result.append(l2.popleft())
        result.extend(l1 or l2)
        return result
	return sortArray(array)
```

## Merge Arrays

Merge K sorted Arrays with a heap

```python
def mergeSortedArrays(self, arrays):
    return list(heapq.merge(*arrays))
```

Or manually with heappush/heappop.

```python
class Solution:
def mergeSortedArrays(self, arrays):
    pq = []
    for i, arr in enumerate(arrays):
        pq.append((arr[0], i, 0))
    heapify(pq)

    res = []
    while pq:
        num, i, j = heappop(pq)
        res.append(num)
        if j + 1 < len(arrays[i]):
            heappush(pq, (arrays[i][j + 1], i, j + 1))
    return res
```

Merging K Sorted Lists

```python
def mergeKLists(self, lists: List[ListNode]) -> ListNode:
    prehead = ListNode()
    heap = []
    for i in range(len(lists)):
        node = lists[i]
        while node:
            heapq.heappush(heap, node.val)
            node = node.next
    node = prehead
    while len(heap) > 0:
        val = heapq.heappop(heap)
        node.next = ListNode()
        node = node.next
        node.val = val
    return prehead.next
```

## Linked List

1. Solutions typically require 3 pointers: current, previous and next
1. Solutions are usually made simplier with a prehead or dummy head node you create and then add to. Then return dummy.next

Reverse:

```python
def reverseLinkedList(head):
    prev, node  = None, head
    while node:
        node.next, prev, node = prev, node, node.next
    return prev
```

Reversing is easier if you can modify the values of the list

```python
def reverse(head):
  node = head
  stk = []
  while node:
    if node.data % 2 == 0:
      stk.append(node)
    if node.data % 2 == 1 or node.next is None:
      while len(stk) > 1:
        stk[-1].data, stk[0].data = stk[0].data, stk[-1].data
        stk.pop(0)
        stk.pop(-1)
      stk.clear()
    node = node.next
  return head
```

Merge:

```python
def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
    dummy = ListNode(-1)

    prev = dummy

    while l1 and l2:
        if l1.val < l2.val:
            prev.next = l1
            l1 = l1.next
        else:
            prev.next = l2
            l2 = l2.next
        prev = prev.next

    prev.next = l1 if l1 is not None else l2

    return dummy.next
```

## Convert Base

1. Typically two steps. A digit modulo step and a integer division step by the next base then reverse the result or use a deque()

Base 10 to 16, or any base by changing '16' and index

```python
def toHex(self, num: int) -> str:
  rtn = []
  index = "0123456789abcdef"
  if num == 0: return '0'
  if num < 0: num += 2 ** 32
  while num > 0:
    digit = num % 16
    rtn.append(index[digit])
    num = num // 16
  return "".join(rtn[::-1])
```

## Parenthesis

1. Count can be used if simple case, otherwise stack. [Basic Calculator](#basic-calculator) is an extension of this algo

```python
def isValid(self, s) -> bool:
  cnt = 0
  for c in s:
    if c == '(':
      cnt += 1
    elif c == ')':
      cnt -= 1
      if cnt < 0:
        return False
  return cnt == 0
```

Stack can be used if more complex

```python
def isValid(self, s: str) -> bool:
  stk = []
  mp = {")":"(", "}":"{", "]":"["}
    for c in s:
      if c in mp.values():
        stk.append(c)
      elif c in mp.keys():
        test = stk.pop() if stk else '#'
        if mp[c] != test:
          return False
  return len(stk) == 0
```

Or must store parenthesis index for further modification

```python
def minRemoveToMakeValid(self, s: str) -> str:
  rtn = list(s)
  stk = []
  for i, c in enumerate(s):
    if c == '(':
      stk.append(i)
    elif c == ')':
      if len(stk) > 0:
        stk.pop()
      else:
        rtn[i] = ''
  while stk:
    rtn[stk.pop()] = ''
  return "".join(rtn)
```

## Max Profit Stock

Infinite Transactions, [base formula](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/discuss/75924/Most-consistent-ways-of-dealing-with-the-series-of-stock-problems)

```python
def maxProfit(self, prices: List[int]) -> int:
    t0, t1 = 0, float('-inf')
    for p in prices:
        t0old = t0
        t0 = max(t0, t1 + p)
        t1 = max(t1, t0old - p)
    return t0
```

Single Transaction, t0 (k-1) = 0

```python
def maxProfit(self, prices: List[int]) -> int:
    t0, t1 = 0, float('-inf')
    for p in prices:
        t0 = max(t0, t1 + p)
        t1 = max(t1, - p)
    return t0
```

K Transactions

```python
t0 = [0] * (k+1)
t1 = [float(-inf)] * (k+1)
for p in prices:
    for i in range(k, 0, -1):
        t0[i] = max(t0[i], t1[i] + p)
        t1[i] = max(t1[i], t0[i-1] - p)
return t0[k]
```

## Shift Array Right

Arrays can be shifted right by reversing the whole string, and then reversing 0,k-1 and k,len(str)

```python
def rotate(self, nums: List[int], k: int) -> None:
    def reverse(l, r, nums):
        while l < r:
            nums[l], nums[r] = nums[r], nums[l]
            l += 1
            r -= 1
    if len(nums) <= 1: return
    k = k % len(nums)
    reverse(0, len(nums)-1, nums)
    reverse(0, k-1, nums)
    reverse(k, len(nums)-1, nums)
```

## Continuous Subarrays with Sum k

The total number of continuous subarrays with sum k can be found by hashing the continuous sum per value and adding the count of continuous sum - k

```python
def subarraySum(self, nums: List[int], k: int) -> int:
    mp = {0: 1}
    rtn, total = 0, 0
    for n in nums:
        total += n
        rtn += mp.get(total - k, 0)
        mp[total] = mp.get(total, 0) + 1
    return rtn
```

## Events

Events pattern can be applied when to many interval problems such as 'Find employee free time between meetings' and 'find peak population' when individual start/ends
are irrelavent and sum start/end times are more important

```python
def employeeFreeTime(self, schedule: '[[Interval]]') -> '[Interval]':
    events = []
    for e in schedule:
        for m in e:
            events.append((m.start, 1))
            events.append((m.end, -1))
    events.sort()
    itv = []
    prev = None
    bal = 0
    for t, c in events:
        if bal == 0 and prev is not None and t != prev:
            itv.append(Interval(prev, t))
        bal += c
        prev = t
    return itv
```

## Merge Meetings

Merging a new meeting into a list

```python
def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
    bisect.insort(intervals, newInterval)
    merged = [intervals[0]]
    for i in intervals:
        ms, me = merged[-1]
        s, e = i
        if me >= s:
            merged[-1] = (ms, max(me, e))
        else:
            merged.append(i)
    return merged
```

## Trie

Good for autocomplete, spell checker, IP routing (match longest prefix), predictive text, solving word games

```python
class Trie:
    def __init__(self):
        self.root = {}

    def addWord(self, s: str):
        tmp = self.root
        for c in s:
            if c not in tmp:
                tmp[c] = {}
            tmp = tmp[c]
        tmp['#'] = s # Store full word at '#' to simplify

    def matchPrefix(self, s: str, tmp=None):
        if not tmp: tmp = self.root
        for c in s:
            if c not in tmp:
                return []
            tmp = tmp[c]

        rtn = []

        for k in tmp:
            if k == '#':
                rtn.append(tmp[k])
            else:
                rtn += self.matchPrefix('', tmp[k])
        return rtn

    def hasWord(self, s: str):
        tmp = self.root
        for c in s:
            if c in tmp:
                tmp = tmp[c]
            else:
                return False
        return True
```

Search example with . for wildcards

```python
def search(self, word: str) -> bool:
    def searchNode(word, node):
        for i,c in enumerate(word):
            if c in node:
                node = node[c]
            elif c == '.':
                return any(searchNode(word[i+1:], node[cn]) for cn in node if cn != '$' )
            else:
                return False
        return '$' in node
    return searchNode(word, self.trie)
```

## Kadane

local_maxiumum[i] = max(A[i], A[i] + local_maximum[i-1])
[Explanation](https://medium.com/@rsinghal757/kadanes-algorithm-dynamic-programming-how-and-why-does-it-work-3fd8849ed73d)
Determine max subarray sum

```python
# input: [-2,1,-3,4,-1,2,1,-5,4]
def maxSubArray(self, nums: List[int]) -> int:
    for i in range(1, len(nums)):
        if nums[i-1] > 0:
            nums[i] += nums[i-1]
    return max(nums) # max([-2,1,-2,4,3,5,6,1,5]) = 6
```

## Union Find

[Union Find](https://www.geeksforgeeks.org/union-find/) is a useful algorithm for graph

DSU for integers

```python
class DSU:
    def __init__(self, N):
        self.par = list(range(N))

    def find(self, x): # Find Parent
        if self.par[x] != x:
            self.par[x] = self.find(self.par[x])
        return self.par[x]

    def union(self, x, y):
        xr, yr = self.find(x), self.find(y)
        if xr == yr: # If parents are equal, return False
            return False
        self.par[yr] = xr # Give y node parent of x
        return True # return True if union occured
```

DSU for strings

```python
class DSU:
    def __init__(self):
        self.par = {}

    def find(self, x):
        if x != self.par.setdefault(x, x):
            self.par[x] = self.find(self.par[x])
        return self.par[x]

    def union(self, x, y):
        xr, yr = self.find(x), self.find(y)
        if xr == yr: return
        self.par[yr] = xr
```

DSU with union by rank

```python
class DSU:
    def __init__(self, N):
        self.par = list(range(N))
        self.sz = [1] * N

    def find(self, x):
        if self.par[x] != x:
            self.par[x] = self.find(self.par[x])
        return self.par[x]

    def union(self, x, y):
        xr, yr = self.find(x), self.find(y)
        if xr == yr:
            return False
        if self.sz[xr] < self.sz[yr]:
            xr, yr = yr, xr
        self.par[yr] = xr
        self.sz[xr] += self.sz[yr]
        return True
```

## Fast Power

Fast Power, or Exponential by [squaring](https://en.wikipedia.org/wiki/Exponentiation_by_squaring) allows calculating squares in logn time (x^n)*2 = x^(2*n)

```python
def myPow(self, x: float, n: int) -> float:
    if n < 0:
        n *= -1
        x = 1/x
    ans = 1
    while n > 0:
        if n % 2 == 1:
            ans = ans * x
        x *= x
        n = n // 2
    return ans
```

## Fibonacci Golden

Fibonacci can be calulated with [Golden Ratio](https://demonstrations.wolfram.com/GeneralizedFibonacciSequenceAndTheGoldenRatio/)

```python
def fib(self, N: int) -> int:
    golden_ratio = (1 + 5 ** 0.5) / 2
    return int((golden_ratio ** N + 1) / 5 ** 0.5)
```

## Basic Calculator

A calculator can be simulated with stack

```python
class Solution:
    def calculate(self, s: str) -> int:
        s += '$'
        def helper(stk, i):
            sign = '+'
            num = 0
            while i < len(s):
                c = s[i]
                if c == " ":
                    i += 1
                    continue
                elif c.isdigit():
                    num = num * 10 + int(c)
                    i += 1
                elif c == '(':
                    num, i = helper([], i+1)
                else:
                    if sign == '+':
                        stk.append(num)
                    if sign == '-':
                        stk.append(-num)
                    if sign == '*':
                        stk.append(stk.pop() * num)
                    if sign == '/':
                        stk.append(int(stk.pop() / num))
                    i += 1
                    num = 0
                    if c == ')':
                        return sum(stk), i
                    sign = c
            return sum(stk)
        return helper([],0)
```

## Reverse Polish

```python
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        stk = []
        while tokens:
            c = tokens.pop(0)
            if c not in '+-/*':
                stk.append(int(c))
            else:
                a = stk.pop()
                b = stk.pop()
                if c == '+':
                    stk.append(a + b)
                if c == '-':
                    stk.append(b-a)
                if c == '*':
                    stk.append(a * b)
                if c == '/':
                    stk.append(int(b / a))
        return stk[0]
```

## Resevior Sampling

Used to sample large unknown populations. Each new item added has a 1/count chance of being selected

```python
def __init__(self, nums):
    self.nums = nums
def pick(self, target):
    res = None
    count = 0
    for i, x in enumerate(self.nums):
        if x == target:
            count += 1
            chance = random.randint(1, count)
            if chance == 1:
                res = i
    return res
```

## String Subsequence

Can find the min number of subsequences of strings in some source through binary search and a dict of the indexes of the source array

```python
def shortestWay(self, source: str, target: str) -> int:
    ref = collections.defaultdict(list)
    for i,c in enumerate(source):
        ref[c].append(i)

    ans = 1
    i = -1
    for c in target:
        if c not in ref:
            return -1
        offset = ref[c]
        j = bisect.bisect_left(offset, i)
        if j == len(offset):
            ans += 1
            i = offset[0] + 1
        else:
            i = offset[j] + 1

    return ans
```

## Candy Crush

Removing adjacent duplicates is much more effective with a stack

```python
def removeDuplicates(self, s: str, k: int) -> str:
    stk = []
    for c in s:
        if stk and stk[-1][0] == c:
            stk[-1][1] += 1
            if stk[-1][1] >= k:
                stk.pop()
        else:
            stk.append([c, 1])
    ans = []
    for c in stk:
        ans.extend([c[0]] * c[1])
    return "".join(ans)
```

## Dutch Flag

[Dutch National Flag Problem](https://en.wikipedia.org/wiki/Dutch_national_flag_problem) proposed by [Edsger W. Dijkstra](https://en.wikipedia.org/wiki/Edsger_W._Dijkstra)

```python
def sortColors(self, nums: List[int]) -> None:
    """
    Do not return anything, modify nums in-place instead.
    """
    # for all idx < p0 : nums[idx < p0] = 0
    # curr is an index of element under consideration
    p0 = curr = 0
    # for all idx > p2 : nums[idx > p2] = 2
    p2 = len(nums) - 1

    while curr <= p2:
        if nums[curr] == 0:
            nums[p0], nums[curr] = nums[curr], nums[p0]
            p0 += 1
            curr += 1
        elif nums[curr] == 2:
            nums[curr], nums[p2] = nums[p2], nums[curr]
            p2 -= 1
        else:
            curr += 1
```
