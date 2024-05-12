Python3 reference for interview coding problems/light competitive programming. Contributions welcome!

## How

I built this cheatsheet while teaching myself Python3 for various interviews and leetcoding for fun after not using Python for about a decade. This cheetsheet only contains code that I didn't know but needed to use to solve a specific coding problem. I did this to try to get a smaller high frequency subset of Python vs a comprehensive list of all methods. Additionally, the act of recording the syntax and algorithms helped me store it in memory and as a result I almost never actually referenced this sheet. Hopefully it helps you in your efforts or inspires you to build your own and best of luck!

## Why

The [rule of least power](https://en.wikipedia.org/wiki/Rule_of_least_power)

I choose Python3 despite being more familiar with Javascript, Java, C++ and Golang for interviews as I felt Python had the combination of the most standard libraries available as well as syntax that resembles psuedo code, therefore being the most expressive. Python and Java both have the most examples but Python wins in this case due to being much more concise. I was able to get myself reasonably prepared with Python syntax in six weeks of practice. After picking up Python I have timed myself solving the same exercises in Golang and Python. Although I prefer Golang, I find that I can complete Python examples in half the time even accounting for +50% more bugs (approximately) that I tend to have in Python vs Go. This is optimizing for solved interview questons under pressure, when performance is considered then Go/C++ does consistently perform 1/10 the time of Python. In some rare cases, algorithms that time out in Python sometimes pass in C++/Go on Leetcode.

[Language Mechanics](#language-mechanics)

1. [Literals](#literals)
1. [Loops](#loops)
1. [Strings](#strings)
1. [Slicing](#slicing)
1. [Tuples](#tuple)
1. [Sort](#sort)
1. [Hash](#hash)
1. [Set](#set)
1. [List](#list)
1. [Dict](#dict)
1. [Binary Tree](#binarytree)
1. [heapq](#heapq)
1. [lambda](#lambda)
1. [zip](#zip)
1. [Random](#random)
1. [Constants](#constants)
1. [Ternary Condition](#ternary)
1. [Bitwise operators](#bitwise-operators)
1. [For Else](#for-else)
1. [Modulo](#modulo)
1. [any](#any)
1. [all](#all)
1. [bisect](#bisect)
1. [math](#math)
1. [iter](#iter)
1. [map](#map)
1. [filter](#filter)
1. [reduce](#reduce)
1. [itertools](#itertools)
1. [regular expression](#regular-expression)
1. [Types](#types)
1. [Grids](#grids)

[Collections](#collections)

1. [Deque](#deque)
1. [Counter](#counter)
1. [Default Dict](#default-dict)


# Language Mechanics

## Literals

```python
255, 0b11111111, 0o377, 0xff # Integers (decimal, binary, octal, hex)
123.0, 1.23                  # Float
7 + 5j, 7j                   # Complex
'a', '\141', '\x61'          # Character (literal, octal, hex)
'\n', '\\', '\'', '\"'       # Newline, backslash, single quote, double quote
"string\n"                   # String of characters ending with newline
"hello"+"world"              # Concatenated strings
True, False                  # bool constants, 1 == True, 0 == False
[1, 2, 3, 4, 5]              # List
['meh', 'foo', 5]            # List
(2, 4, 6, 8)                 # Tuple, immutable
{'name': 'a', 'age': 90}     # Dict
{'a', 'e', 'i', 'o', 'u'}    # Set
None                         # Null var
```

## Loops

Go through all elements

```python
i = 0
while i < len(str):
  i += 1
```

equivalent

```python
for i in range(len(message)):
  print(i)
```

Get largest number index from right

```python
while i > 0 and nums [i-1] >= nums[i]:
  i -= 1
```

Manually reversing

```python
l, r = i, len(nums) - 1
while l < r:
  nums[l], nums[r] = nums[r], nums[l]
  l += 1
  r -= 1
```

Go past the loop if we are clever with our boundry

```python
for i in range(len(message) + 1):
  if i == len(message) or message[i] == ' ':
```

Fun with Ranges - range(start, stop, step)

```python
for a in range(0,3): # 0,1,2
for a in reversed(range(0,3)) # 2,1,0
for i in range(3,-1,-1) # 3,2,1,0
for i in range(len(A)//2): # A = [0,1,2,3,4,5]
  print(i) # 0,1,2
  print(A[i]) # 0,1,2
  print(~i) # -1,-2,-3
  print(A[~i]) # 5,4,3
```

## Strings

```python
str1.find('x')          # find first location of char x and return index
str1.rfind('x')         # find first int location of char x from reverse
```

Parse a log on ":"

```python
l = "0:start:0"
tokens = l.split(":")
print(tokens) # ['0', 'start', '0']
```

Reverse works with built in split, [::-1] and " ".join()

```python
# s = "the sky  is blue"
def reverseWords(self, s: str) -> str:
  wordsWithoutWhitespace = s.split() # ['the', 'sky', 'is', 'blue']
  reversedWords = wordsWithoutWhitespace[::-1] # ['blue', 'is', 'sky', 'the']
  final = " ".join(reversedWords) # blue is sky the
```

Manual split based on isalpha()

```python
def splitWords(input_string) -> list:
  words = [] #
  start = length = 0
  for i, c in enumerate(input_string):
    if c.isalpha():
      if length == 0:
        start = i
        length += 1
      else:
        words.append(input_string[start:start+length])
        length = 0
  if length > 0:
    words.append(input_string[start:start+length])
  return words
```

Test type of char

```python
def rotationalCipher(input, rotation_factor):
  rtn = []
  for c in input:
    if c.isupper():
      ci = ord(c) - ord('A')
      ci = (ci + rotation_factor) % 26
      rtn.append(chr(ord('A') + ci))
    elif c.islower():
      ci = ord(c) - ord('a')
      ci = (ci + rotation_factor) % 26
      rtn.append(chr(ord('a') + ci))
    elif c.isnumeric():
      ci = ord(c) - ord('0')
      ci = (ci + rotation_factor) % 10
      rtn.append(chr(ord('0') + ci))
    else:
      rtn.append(c)
  return "".join(rtn)
```

AlphaNumberic

```python
isalnum()
```

Get charactor index

```python
print(ord('A')) # 65
print(ord('B')-ord('A')+1) # 2
print(chr(ord('a') + 2)) # c
```

Replace characters or strings

```python
def isValid(self, s: str) -> bool:
  while '[]' in s or '()' in s or '{}' in s:
    s = s.replace('[]','').replace('()','').replace('{}','')
  return len(s) == 0
```

Insert values in strings

```python
txt3 = "My name is {}, I'm {}".format("John",36) # My name is John, I'm 36
```

Multiply strings/lists with \*, even booleans which map to True(1) and False(0)

```python
'meh' * 2 # mehmeh
['meh'] * 2 # ['meh', 'meh']
['meh'] * True #['meh']
['meh'] * False #[]
```

Find substring in string

```python
txt = "Hello, welcome to my world."
x = txt.find("welcome")  # 7
```

startswith and endswith are very handy

```python
str = "this is string example....wow!!!"
str.endswith("!!") # True
str.startswith("this") # True
str.endswith("is", 2, 4) # True
```

Python3 format strings

```python
name = "Eric"
profession = "comedian"
affiliation = "Monty Python"
message = (
     f"Hi {name}. "
     f"You are a {profession}. "
     f"You were in {affiliation}."
)
message
'Hi Eric. You are a comedian. You were in Monty Python.'
```

Print string with all chars, useful for debugging

```python
print(repr("meh\n"))     # 'meh\n'
```

## Slicing

Slicing [intro](https://stackoverflow.com/questions/509211/understanding-slice-notation)

```python
                +---+---+---+---+---+---+
                | P | y | t | h | o | n |
                +---+---+---+---+---+---+
Slice position: 0   1   2   3   4   5   6
Index position:   0   1   2   3   4   5
p = ['P','y','t','h','o','n']
p[0] 'P' # indexing gives items, not lists
alpha[slice(2,4)] # equivalent to p[2:4]
p[0:1] # ['P'] Slicing gives lists
p[0:5] # ['P','y','t','h','o'] Start at beginning and count 5
p[2:4] = ['t','r'] # Slice assignment  ['P','y','t','r','o','n']
p[2:4] = ['s','p','a','m'] # Slice assignment can be any size['P','y','s','p','a','m','o','n']
p[4:4] = ['x','y'] # insert slice ['P','y','t','h','x','y','o','n']
p[0:5:2] # ['P', 't', 'o'] sliceable[start:stop:step]
p[5:0:-1] # ['n', 'o', 'h', 't', 'y']
```

Go through num and get combinations missing a member

```python
numList = [1,2,3,4]
for i in range(len(numList)):
    newList = numList[0:i] + numList[i+1:len(numList)]
    print(newList) # [2, 3, 4], [1, 3, 4], [1, 2, 4], [1, 2, 3]
```

## Tuple

Collection that is ordered and unchangable

```python
thistuple = ("apple", "banana", "cherry")
print(thistuple[1]) # banana
```

Can be used with Dicts

```python
def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
    d = defaultdict(list)
    for w in strs:
        key = tuple(sorted(w))
        d[key].append(w)
    return d.values()
```

## Sort

sorted(iterable, key=key, reverse=reverse)

Sort sorts alphabectically, from smallest to largest

```python
print(sorted(['Ford', 'BMW', 'Volvo'])) # ['BMW', 'Ford', 'Volvo']
nums = [-4,-1,0,3,10]
print(sorted(n*n for n in nums)) # [0,1,9,16,100]
```

```python
cars = ['Ford', 'BMW', 'Volvo']
cars.sort() # returns None type
cars.sort(key=lambda x: len(x) ) # ['BMW', 'Ford', 'Volvo']
print(sorted(cars, key=lambda x:len(x))) # ['BMW', 'Ford', 'Volvo']
```

Sort key by value, even when value is a list

```python
meh = {'a':3,'b':0,'c':2,'d':-1}
print(sorted(meh, key=lambda x:meh[x])) # ['d', 'b', 'c', 'a']
meh = {'a':[0,3,'a'],'b':[-2,-3,'b'],'c':[2,3,'c'],'d':[-2,-2,'d']}
print(sorted(meh, key=lambda x:meh[x])) # ['b', 'd', 'a', 'c']
```

```python
def merge_sorted_lists(arr1, arr2): # built in sorted does Timsort optimized for subsection sorted lists
    return sorted(arr1 + arr2)
```

Sort an array but keep the original indexes

```python
self.idx, self.vals = zip(*sorted([(i,v) for i,v in enumerate(nums)], key=lambda x:x[1]))
```

Sort by tuple, 2nd element then 1st ascending

```python
a = [(5,10), (2,20), (2,3), (0,100)]
test = sorted(a, key = lambda x: (x[1],x[0]))
print(test) # [(2, 3), (5, 10), (2, 20), (0, 100)]
test = sorted(a, key = lambda x: (-x[1],x[0]))
print(test) # [(0, 100), (2, 20), (5, 10), (2, 3)]
```

Sort and print dict values by key

```python
ans = {-1: [(10, 1), (3, 3)], 0: [(0, 0), (2, 2), (7, 4)], -3: [(8, 5)]}
for key, value in sorted(ans.items()): print(value)
# [(8, 5)]
# [(10, 1), (3, 3)]
# [(0, 0), (2, 2), (7, 4)]

# sorted transforms dicts to lists
sorted(ans) # [-3, -1, 0]
sorted(ans.values()) # [[(0, 0), (2, 2), (7, 4)], [(8, 5)], [(10, 1), (3, 3)]]
sorted(ans.items()) # [(-3, [(8, 5)]), (-1, [(10, 1), (3, 3)]), (0, [(0, 0), (2, 2), (7, 4)])]
# Or just sort the dict directly
[ans[i] for i in sorted(ans)]
# [[(8, 5)], [(10, 1), (3, 3)], [(0, 0), (2, 2), (7, 4)]]
```

## Hash

```python
for c in s1: # Adds counter for c
  ht[c] = ht.get(c, 0) + 1 # ht[a] = 1, ht[a]=2, etc
```

## Set

```python
a = 3
st = set()
st.add(a) # Add to st
st.remove(a) # Remove from st
st.discard(a) # Removes from set with no error
st.add(a) # Add to st
next(iter(s)) # return 3 without removal
st.pop() # returns 3
```

```python
s = set('abc') # {'c', 'a', 'b'}
s |= set('cdf') # {'f', 'a', 'b', 'd', 'c'} set s with elements from new set
s &= set('bd') # {'d', 'b'} only elements from new set
s -= set('b') # {'d'} remove elements from new set
s ^= set('abd') # {'a', 'b'} elements from s or new but not both
```

## List

Stacks are implemented with Lists. Stacks are good for parsing and graph traversal

```python
test = [0] * 100 # initialize list with 100 0's
```

2D

```python
rtn.append([])
rtn[0].append(1) # [[1]]
```

List Comprehension

```python
number_list = [ x for x in range(20) if x % 2 == 0]
print(number_list) # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
```

Reverse a list

```python
ss = [1,2,3]
ss.reverse()
print(ss) #3,2,1
```

Join list

```python
list1 = ["a", "b" , "c"]
list2 = [1, 2, 3]
list3 = list1 + list2 # ['a', 'b', 'c', 1, 2, 3]
```

## Dict

Hashtables are implemented with dictionaries

```python
d = {'key': 'value'}         # Declare dict{'key': 'value'}
d['key'] = 'value'           # Add Key and Value
{x:0 for x in {'a', 'b'}}    # {'a': 0, 'b': 0} declare through comprehension
d['key'])                    # Access value
d.items()                    # Items as tuple list dict_items([('key', 'value')])
if 'key' in d: print("meh")  # Check if value exists
par = {}
par.setdefault(1,1)          # returns 1, makes par = { 1 : 1 }
par = {0:True, 1:False}
par.pop(0)                   # Remove key 0, Returns True, par now {1: False}
for k in d: print(k)         # Iterate through keys
```

Create Dict of Lists that match length of list to count votes

```python
votes = ["ABC","CBD","BCA"]
rnk = {v:[0] * len(votes[0]) for v in votes[0]}
print(rnk) # {'A': [0, 0, 0], 'B': [0, 0, 0], 'C': [0, 0, 0]}
```

## Tree

1. A [tree](https://www.geeksforgeeks.org/some-theorems-on-trees/) is an undirected [graph](https://www.cs.sfu.ca/~ggbaker/zju/math/trees.html) in which any two vertices are
   connected by exactly one path.
1. Any connected graph who has n nodes with n-1 edges is a tree.
1. The degree of a vertex is the number of edges connected to the vertex.
1. A leaf is a vertex of degree 1. An internal vertex is a vertex of degree at least 2.
1. A [path graph](https://en.wikipedia.org/wiki/Path_graph) is a tree with two or more vertices with no branches, degree of 2 except for leaves which have degree of 1

1. Any two vertices in G can be connected by a unique simple path.
1. G is acyclic, and a simple cycle is formed if any edge is added to G.
1. G is connected and has no cycles.
1. G is connected but would become disconnected if any single edge is removed from G.

## BinaryTree

DFS Pre, In Order, and Post order Traversal

- Preorder
  - encounters roots before leaves
  - Create copy
- Inorder
  - flatten tree back to original sequence
  - Get values in non-decreasing order in BST
- Post order
  - encounter leaves before roots
  - Helpful for deleting

Recursive

```python
"""
     1
    / \
   2   3
  / \
 4   5
"""
# PostOrder 4 5 2 3 1  (Left-Right-Root)
def postOrder(node):
  if node is None:
    return
  postorder(node.left)
  postorder(node.right)
  print(node.value, end=' ')
```

Iterative PreOrder

```python
# PreOrder  1 2 4 5 3 (Root-Left-Right)
def preOrder(tree_root):
  stack = [(tree_root, False)]
  while stack:
    node, visited = stack.pop()
    if node:
      if visited:
        print(node.value, end=' ')
      else:
        stack.append((node.right, False))
        stack.append((node.left, False))
        stack.append((node, True))
```

Iterative InOrder

```python
# InOrder   4 2 5 1 3 (Left-Root-Right)
def inOrder(tree_root):
  stack = [(tree_root, False)]
  while stack:
    node, visited = stack.pop()
    if node:
      if visited:
        print(node.value, end=' ')
      else:
        stack.append((node.right, False))
        stack.append((node, True))
        stack.append((node.left, False))
```

Iterative PostOrder

```python
# PostOrder 4 5 2 3 1  (Left-Right-Root)
def postOrder(tree_root):
  stack = [(tree_root, False)]
  while stack:
    node, visited = stack.pop()
    if node:
      if visited:
        print(node.value, end=' ')
      else:
        stack.append((node, True))
        stack.append((node.right, False))
        stack.append((node.left, False))
```

Iterative BFS(LevelOrder)

```python
from collections import deque

#BFS levelOrder 1 2 3 4 5
def levelOrder(tree_root):
  queue = deque([tree_root])
  while queue:
    node = queue.popleft()
    if node:
        print(node.value, end=' ')
        queue.append(node.left)
        queue.append(node.right)

def levelOrderStack(tree_root):
    stk = [(tree_root, 0)]
    rtn = []
    while stk:
        node, depth = stk.pop()
        if node:
            if len(rtn) < depth + 1:
                rtn.append([])
            rtn[depth].append(node.value)
            stk.append((node.right, depth+1))
            stk.append((node.left, depth+1))
    print(rtn)
    return True

def levelOrderStackRec(tree_root):
    rtn = []

    def helper(node, depth):
        if len(rtn) == depth:
            rtn.append([])
        rtn[depth].append(node.value)
        if node.left:
            helper(node.left, depth + 1)
        if node.right:
            helper(node.right, depth + 1)

    helper(tree_root, 0)
    print(rtn)
    return rtn
```

Traversing data types as a graph, for example BFS

```python
def removeInvalidParentheses(self, s: str) -> List[str]:
    rtn = []
    v = set()
    v.add(s)
    if len(s) == 0: return [""]
    while True:
        for n in v:
            if self.isValid(n):
                rtn.append(n)
        if len(rtn) > 0: break
        level = set()
        for n in v:
            for i, c in enumerate(n):
                if c == '(' or c == ')':
                    sub = n[0:i] + n[i + 1:len(n)]
                    level.add(sub)
        v = level
    return rtn
```

Reconstructing binary trees

1. Binary tree could be constructed from preorder and inorder traversal
1. Inorder traversal of BST is an array sorted in the ascending order

Convert tree to array and then to balanced tree

```python
def balanceBST(self, root: TreeNode) -> TreeNode:
    self.inorder = []

    def getOrder(node):
        if node is None:
            return
        getOrder(node.left)
        self.inorder.append(node.val)
        getOrder(node.right)

    # Get inorder treenode ["1,2,3,4"]
    getOrder(root)

    # Convert to Tree
    #        2
    #       1 3
    #          4
    def bst(listTree):
        if not listTree:
            return None
        mid = len(listTree) // 2
        root = TreeNode(listTree[mid])
        root.left = bst(listTree[:mid])
        root.right = bst(listTree[mid+1:])
        return root

    return bst(self.inorder)
```

## Graph

Build an [adjecency graph](https://www.khanacademy.org/computing/computer-science/algorithms/graph-representation/a/representing-graphs) from edges list

```python
# N = 6, edges = [[0,1],[0,2],[2,3],[2,4],[2,5]]
graph = [[] for _ in range(N)]
for u,v in edges:
    graph[u].append(v)
    graph[v].append(u)
# [[1, 2], [0], [0, 3, 4, 5], [2], [2], [2]]
```

Build adjecency graph from traditional tree

```python
adj = collections.defaultdict(list)
def dfs(node):
    if node.left:
        adj[node].append(node.left)
        adj[node.left].append(node)
        dfs(node.left)
    if node.right:
        adj[node].append(node.right)
        adj[node.right].append(node)
        dfs(node.right)
dfs(root)
```

Traverse Tree in graph notation

```python
# [[1, 2], [0], [0, 3, 4, 5], [2], [2], [2]]
def dfs(node, par=-1):
    for nei in graph[node]:
        if nei != par:
            res = dfs(nei, node)
dfs(0) # 1->2->3->4->5
```

## Heapq

```
      1
     / \
    2   3
   / \ / \
  5  6 8  7
```

[Priority Queue](https://realpython.com/python-heapq-module/#data-structures-heaps-and-priority-queues)

1. Implemented as complete binary tree, which has all levels as full excepted deepest
1. In a heap tree the node is smaller than its children

```python
import heapq # (minHeap by Default)

nums = [5, 7, 9, 1, 3]

heapq.heapify(nums) # converts list into heap. Can be converted back to list by list(nums).
heapq.heappush(nums,element) # Push an element into the heap
heapq.heappop(nums) # Pop an element from the heap
#heappush(heap, ele) :- This function is used to insert the element mentioned in its arguments into heap. The order is adjusted, so as heap structure is maintained.
#heappop(heap) :- This function is used to remove and return the smallest element from heap. The order is adjusted, so as heap structure is maintained.

# Other Methods Available in the Library
# Used to return the k largest elements from the iterable specified 
# The key is a function with that accepts single element from iterable,
# and the returned value from that function is then used to rank that element in the heap
heapq.nlargest(k, iterable, key = fun)
heapq.nsmallest(k, iterable, key = fun)

books = [
    {"title": "Book A", "price": 30},
    {"title": "Book B", "price": 20},
]

# Function to extract the price from a book dictionary
def get_book_price(book):
    return book["price"]

# Find the top 3 most expensive books based on price
top_expensive_books = heapq.nlargest(3, books, key=get_book_price)

# Insert custom objects into the min-heap based on priority
#the tuple (priority,object) is used for custom data structures

# Define a custom class representing a task
class Task:
    def __init__(self, name, priority):
        self.name = name
        self.priority = priority

# List of Task objects
tasks = [
    Task("Task A", 3),
    Task("Task B", 1),
]

# Convert the list of Task objects into a list of tuples (priority, Task)
task_heap = [(task.priority, task) for task in tasks]

# Use heapq.heapify to rearrange the list into a min-heap based on priority
heapq.heapify(task_heap) 

```

```python
def maximumProduct(self, nums: List[int]) -> int:
  l = heapq.nlargest(3, nums)
  s = heapq.nsmallest(3, nums)
  return max(l[0]*l[1]*l[2],s[0]*s[1]*l[0])
```

Heap elements can be tuples, heappop() frees the smallest element (flip sign to pop largest)

```python
def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
    heap = []
    for p in points:
        distance = sqrt(p[0]* p[0] + p[1]*p[1])
        heapq.heappush(heap,(-distance, p))
        if len(heap) > K:
            heapq.heappop(heap)
    return ([h[1] for h in heap])
```

nsmallest can take a lambda argument

```python
def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
    return heapq.nsmallest(K, points, lambda x: x[0]*x[0] + x[1]*x[1])
```

The key can be a function as well in nsmallest/nlargest

```python
def topKFrequent(self, nums: List[int], k: int) -> List[int]:
    count = Counter(nums)
    return heapq.nlargest(k, count, count.get)
```

Tuple sort, 1st/2nd element. increasing frequency then decreasing order

```python
def topKFrequent(self, words: List[str], k: int) -> List[str]:
    freq = Counter(words)
    return heapq.nsmallest(k, freq.keys(), lambda x:(-freq[x], x))
```



## Lambda

Can be used with (list).sort(), sorted(), min(), max(), (heapq).nlargest,nsmallest(), map()

```python
# a=3,b=8,target=10
min((b,a), key=lambda x: abs(target - x)) # 8
```

```python
>>> ids = ['id1', 'id2', 'id30', 'id3', 'id22', 'id100']
>>> print(sorted(ids)) # Lexicographic sort
['id1', 'id100', 'id2', 'id22', 'id3', 'id30']
>>> sorted_ids = sorted(ids, key=lambda x: int(x[2:])) # Integer sort
>>> print(sorted_ids)
['id1', 'id2', 'id3', 'id22', 'id30', 'id100']
```

```python
trans = lambda x: list(al[i] for i in x) # apple, a->0..
print(trans(words[0])) # [0, 15, 15, 11, 4]
```

Lambda can sort by 1st, 2nd element in tuple

```python
sorted([('abc', 121),('bbb',23),('abc', 148),('bbb', 24)], key=lambda x: (x[0],x[1]))
# [('abc', 121), ('abc', 148), ('bbb', 23), ('bbb', 24)]
```

## Zip

Combine two dicts or lists

```python
s1 = {2, 3, 1}
s2 = {'b', 'a', 'c'}
list(zip(s1, s2)) # [(1, 'a'), (2, 'c'), (3, 'b')]
```

Traverse in Parallel

```python
letters = ['a', 'b', 'c']
numbers = [0, 1, 2]
for l, n in zip(letters, numbers):
  print(f'Letter: {l}') # a,b,c
  print(f'Number: {n}') # 0,1,2
```

Empty in one list is ignored

```python
letters = ['a', 'b', 'c']
numbers = []
for l, n in zip(letters, numbers):
  print(f'Letter: {l}') #
  print(f'Number: {n}') #
```

Compare characters of alternating words

```python
for a, b in zip(words, words[1:]):
    for c1, c2 in zip(a,b):
        print("c1 ", c1, end=" ")
        print("c2 ", c2, end=" ")
```

Passing in [\*](https://stackoverflow.com/questions/29139350/difference-between-ziplist-and-ziplist/29139418) unpacks a list or other iterable, making each of its elements a separate argument.

```python
a = [[1,2],[3,4]]
test = zip(*a)
print(test) # (1, 3) (2, 4)
matrix = [[1,2,3],[4,5,6],[7,8,9]]
test = zip(*matrix)
print(*test) # (1, 4, 7) (2, 5, 8) (3, 6, 9)
```

Useful when rotating a matrix

```python
# matrix = [[1,2,3],[4,5,6],[7,8,9]]
matrix[:] = zip(*matrix[::-1]) # [[7,4,1],[8,5,2],[9,6,3]]
```

Iterate through chars in a list of strs

```python
strs = ["cir","car","caa"]
for i, l in enumerate(zip(*strs)):
    print(l)
    # ('c', 'c', 'c')
    # ('i', 'a', 'a')
    # ('r', 'r', 'a')
```

Diagonals can be traversed with the help of a list

```python
"""
[[1,2,3],
 [4,5,6],
 [7,8,9],
 [10,11,12]]
"""
def printDiagonalMatrix(self, matrix: List[List[int]]) -> bool:
    R = len(matrix)
    C = len(matrix[0])

    tmp = [[] for _ in range(R+C-1)]

    for r in range(R):
        for c in range(C):
            tmp[r+c].append(matrix[r][c])

    for t in tmp:
        for n in t:
            print(n, end=' ')
        print("")
"""
 1,
 2,4
 3,5,7
 6,8,10
 9,11
 12
"""
```

## Random

```Python
for i, l in enumerate(shuffle):
  r = random.randrange(0+i, len(shuffle))
  shuffle[i], shuffle[r] = shuffle[r], shuffle[i]
return shuffle
```

Other random generators

```Python
import random
ints = [0,1,2]
random.choice(ints) # 0,1,2
random.choices([1,2,3],[1,1,10]) # 3, heavily weighted
random.randint(0,2) # 0,1, 2
random.randint(0,0) # 0
random.randrange(0,0) # error
random.randrange(0,2) # 0,1
```

## Constants

```Python
max = float('-inf')
min = float('inf')
```

## Ternary

a if condition else b

```Python
test = stk.pop() if stk else '#'
```

## Bitwise Operators

```python
'0b{:04b}'.format(0b1100 & 0b1010) # '0b1000' and
'0b{:04b}'.format(0b1100 | 0b1010) # '0b1110' or
'0b{:04b}'.format(0b1100 ^ 0b1010) # '0b0110' exclusive or
'0b{:04b}'.format(0b1100 >> 2)     # '0b0011' shift right
'0b{:04b}'.format(0b0011 << 2)     # '0b1100' shift left
```

## For Else

Else condition on for loops if break is not called

```python
for w1, w2 in zip(words, words[1:]): #abc, ab
    for c1, c2 in zip(w1, w2):
        if c1 != c2:
            adj[c1].append(c2)
            degrees[c2] += 1
            break
    else: # nobreak
        if len(w1) > len(w2):
            return ""   # Triggers since ab should be before abc, not after
```

## Modulo

```python
for n in range(-8,8):
    print n, n//4, n%4

 -8 -2 0
 -7 -2 1
 -6 -2 2
 -5 -2 3

 -4 -1 0
 -3 -1 1
 -2 -1 2
 -1 -1 3

  0  0 0
  1  0 1
  2  0 2
  3  0 3

  4  1 0
  5  1 1
  6  1 2
  7  1 3
```

## Any

if any element of the iterable is True

```python
def any(iterable):
    for element in iterable:
        if element:
            return True
    return False
```

## All

```python
def all(iterable):
    for element in iterable:
        if not element:
            return False
    return True
```

## Bisect

- bisect.bisect_left returns the leftmost place in the sorted list to insert the given element
- bisect.bisect_right returns the rightmost place in the sorted list to insert the given element

```python
import bisect
bisect.bisect_left([1,2,3,4,5], 2)  # 1
bisect.bisect_right([1,2,3,4,5], 2) # 2
bisect.bisect_left([1,2,3,4,5], 7)  # 5
bisect.bisect_right([1,2,3,4,5], 7) # 5
```

Insert x in a in sorted order. This is equivalent to a.insert(bisect.bisect_left(a, x, lo, hi), x) assuming that a is already sorted. Search is binary search O(logn) and insert is O(n)

```python
import bisect
l = [1, 3, 7, 5, 6, 4, 9, 8, 2]
result = []
for e in l:
    bisect.insort(result, e)
print(result) # [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

```python
li1 = [1, 3, 4, 4, 4, 6, 7] # [1, 3, 4, 4, 4, 5, 6, 7]
bisect.insort(li1, 5) #
```

Bisect can give two ends of a range, if the array is sorted of course

```python
s = bisect.bisect_left(nums, target)
e = bisect.bisect(nums, target) -1
if s <= e:
    return [s,e]
else:
    return [-1,-1]
```

## Math

Calulate power

```python
# (a ^ b) % p.
d = pow(a, b, p)
```

Division with remainder

```python
divmod(8, 3) # (2, 2)
divmod(3, 8) #  (0, 3)
```

## eval

Evaluates an expression

```python
x = 1
print(eval('x + 1'))
```

## Iter

Creates iterator from container object such as list, tuple, dictionary and set

```python
mytuple = ("apple", "banana", "cherry")
myit = iter(mytuple)
print(next(myit)) # apple
print(next(myit)) # banana
```

## Map

map(func, \*iterables)

```python
my_pets = ['alfred', 'tabitha', 'william', 'arla']
uppered_pets = list(map(str.upper, my_pets)) # ['ALFRED', 'TABITHA', 'WILLIAM', 'ARLA']
my_strings = ['a', 'b', 'c', 'd', 'e']
my_numbers = [1,2,3,4,5]
results = list(map(lambda x, y: (x, y), my_strings, my_numbers)) # [('a', 1), ('b', 2), ('c', 3), ('d', 4), ('e', 5)]
```

```python
A1 = [1, 4, 9]
''.join(map(str, A1))
```

## Filter

filter(func, iterable)

```python
scores = [66, 90, 68, 59, 76, 60, 88, 74, 81, 65]
over_75 = list(filter(lambda x: x>75, scores)) # [90, 76, 88, 81]
```

```python
scores = [66, 90, 68, 59, 76, 60, 88, 74, 81, 65]
def is_A_student(score):
    return score > 75
over_75 = list(filter(is_A_student, scores)) # [90, 76, 88, 81]
```

```python
dromes = ("demigod", "rewire", "madam", "freer", "anutforajaroftuna", "kiosk")
palindromes = list(filter(lambda word: word == word[::-1], dromes)) # ['madam', 'anutforajaroftuna']
```

Get degrees == 0 from list

```python
stk = list(filter(lambda x: degree[x]==0, degree.keys()))
```

## Reduce

reduce(func, iterable[, initial])
where initial is optional

```python
numbers = [3, 4, 6, 9, 34, 12]
result = reduce(lambda x, y: x+y, numbers) # 68
result = reduce(lambda x, y: x+y, numbers, 10) #78
```

## itertools

[itertools.accumulate(iterable[, func]) –> accumulate object](https://www.geeksforgeeks.org/python-itertools-accumulate/)

```python
import itertools
data = [3, 4, 6, 2, 1, 9, 0, 7, 5, 8]
list(itertools.accumulate(data)) # [3, 7, 13, 15, 16, 25, 25, 32, 37, 45]
list(accumulate(data, max))  # [3, 4, 6, 6, 6, 9, 9, 9, 9, 9]
cashflows = [1000, -90, -90, -90, -90]  # Amortize a 5% loan of 1000 with 4 annual payments of 90
list(itertools.accumulate(cashflows, lambda bal, pmt: bal*1.05 + pmt)) [1000, 960.0, 918.0, 873.9000000000001, 827.5950000000001]
for k,v in groupby("aabbbc")    # group by common letter
    print(k)                    # a,b,c
    print(list(v))              # [a,a],[b,b,b],[c,c]
```

## Regular Expression

RE module allows regular expressions in python

```python
def removeVowels(self, S: str) -> str:
    return re.sub('a|e|i|o|u', '', S)
```

## Types

from typing import List, Set, Dict, Tuple, Optional
[cheat sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)

## Grids

Useful helpful function

```python
R = len(grid)
C = len(grid[0])

def neighbors(r, c):
    for nr, nc in ((r,c-1), (r,c+1), (r-1, c), (r+1,c)):
        if 0<=nr<R and 0<=nc<C:
            yield nr, nc

def dfs(r,c, index):
    area = 0
    grid[r][c] = index
    for x,y in neighbors(r,c):
        if grid[x][y] == 1:
            area += dfs(x,y, index)
    return area + 1
```

# Collections

Stack with appendleft() and popleft()

## Deque

```python
from collections import deque
deq = deque([1, 2, 3])
deq.appendleft(5)
deq.append(6)
deq
deque([5, 1, 2, 3, 6])
deq.popleft()
5
deq.pop()
6
deq
deque([1, 2, 3])
deque[0] #gets left element
deque[-1] #gets right element
```

## Counter

```python
from collections import Counter
count = Counter("hello") # Counter({'h': 1, 'e': 1, 'l': 2, 'o': 1})
count['l'] # 2
count['l'] += 1
count['l'] # 3
```

Get counter k most common in list of tuples

```python
# [1,1,1,2,2,3]
# Counter  [(1, 3), (2, 2)]
def topKFrequent(self, nums: List[int], k: int) -> List[int]:
    if len(nums) == k:
        return nums
    return [n[0] for n in Counter(nums).most_common(k)] # [1,2]
```

elements() lets you walk through each number in the Counter

```python
def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
    c1 = collections.Counter(nums1) # [1,2,2,1]
    c2 = collections.Counter(nums2) # [2,2]
    dif = c1 & c2                   # {2:2}
    return list(dif.elements())     # [2,2]
```

operators work on Counter

```python
c = Counter(a=3, b=1)
d = Counter(a=1, b=2)
c + d # {'a': 4, 'b': 3}
c - d # {'a': 2}
c & d # {'a': 1, 'b': 1}
c | d # {'a': 3, 'b': 2}
c = Counter(a=2, b=-4)
+c # {'a': 2}
-c # {'b': 4}
```

## Default Dict

```python
d={}
print(d['Grapes'])# This gives Key Error
from collections import defaultdict
d = defaultdict(int) # set default
print(d['Grapes']) # 0, no key error
d = collections.defaultdict(lambda: 1)
print(d['Grapes']) # 1, no key error
```

```python
from collections import defaultdict
dd = defaultdict(list)
dd['key'].append(1) # defaultdict(<class 'list'>, {'key': [1]})
dd['key'].append(2) # defaultdict(<class 'list'>, {'key': [1, 2]})
```
