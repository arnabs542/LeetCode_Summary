# LeetCode_Summary

 A;
    Abc abc abc


B：
    Bcd acd def

Since two trees are not on the same root, connect the A to B
Then we would have to union them by using union-find. The resulting 
Set would be since abc-> {bcd, acd, def}


The size would get updated as 
  1     1      1
   Abc abc   abc
  1      1     1 
   Bcd acd  def


1            2       2         2
Abc     bcd    acd     def

And abc itself is the father node so the largest set is the set with all books.




431


{1,2,4#2,1,4#3,5#4,1,2#5,3}


1------2                
\        |
 \       |
  \ ----4


3
|
|
5
假设x是要找的节点label,  那么 先判断x是否在father map 里面，如果没有那么就当它是father节点。然后如果有，先求出它对应的father 是什么，假设这个节点是root，然后判断root有三种情况，
Root对应---x, 那么直接返回x，因为是father 节点
如果root 不是他自己的父亲，那么while 找到自己父亲
如果root 不是x对应的父亲的值，那么x 就不是父亲节点，
更新fathers map 让x的value变成root，然后更新x为自己的更新前的father 节点


**114 Unique Path**

M = 3 N = 3

6 unique paths

D: down
R: right

DDRR
DRDR
DRRD
RRDD
RDRD
RDDR

So the dp[i][j] represents how many paths robot is when at i j
Transit dp[i][j] = previous up and previous right = dp[i - 1][j] + dp[i][j - 1]

Init: dp[i][0] = 1 dp [0][j] = 1


**76 Longest Increasing Subsequence --- Very Classic and Extensive to other DP problem**

Example 1:
    Input:  [5,4,1,2,3]
    Output:  3
    
    Explanation:
    LIS is [1,2,3]
    
If we want to find 3 LIS
----
We need to find 2 LIS
----
We need to find 1 LIS
---
Discover the exit

State dp[i] the first i LIS in the problem, but this can not do the state transition
So reset it as dp[i] is the subsequence (not LIS) first i and ends at i 
So now the state transit can be determined 
---
State transit
Dp[i] = max(dp[i], dp[j] + 1) nums[j] < nums[i] and j < i

O(n ^ 2)

O(nlogn) use Binary Search



**N-Queens**

. Q . . .

… . Q

Q ….
.. Q ..


Solution
Brute force nested for loops 
Each loop enumerate the position of the Queen
DFS + HashSet, Each not allowed positions are 
Same col, (i + level), Same row (i - level), or Same 
Diagonal ()

Say I have three sets for storing each scenarios that can
‘T reached by the queen, sum, diff, and col, each one of them 
Would be gonna be, each time for the successful next move 
i t would be three set doesn’t have the coordinate.


**802 Sudoku Problem**

Solution: brute-force, nested for loops, each block would have 9 possible 
Numbers to pick, each cell has to check if previous has been selected

Solution 2 DFS

Rules to obey when come to the non-zero values in the matrix, we need to mark
The col and row indices, and the subgrid index as visited;

For zero value, I need to know it can’t break the rule, as the col, row, and subgrid would 
Not contain the number that I want to place

Make sure the placement would not break the board

Finally we fill the whole board

Recursion method 



**623 K Edit Distances**

[“abc”, “abd”, “abcd”, “abc”]

Edit distance with the target no greater than k

target=”ac”, k = “1”

Outputs [“abc”, “adc”] -- abc 

For each word, do the three operations on kth character so that the
Target can be reached or not

For search the word in the words, use Trie to realize since Trie is used to find the string and match the prefix

**406 Minimum size subarray sum**

[2,3,1,2,4,3], s = 7
Ans: 2 since [4, 3] has the sum equals to 7

Two pointers

`inc` 
2   3   1   2   4   3 
^
l,r

2   3   1   2   4   3 
^    ^
L,   r

2   3   1   2   4   3 
^         ^
L,        r

2   3   1   2   4   3 
^             ^
L,            r

2   3   1   2   4   3 
     ^        ^
     L,       r
Sum = 6

…..


2   3   1   2   4   3 
     ^              ^ 
     L,             r
Sum = 10
2   3   1   2   4   3 
           ^        ^
           L,        r
Sum = 9
2   3   1   2   4   3 
               ^    ^
               L,   r
Sum = 7

Algo
So enumerate the right until the sum is greater than target sum



S -- prefix sum
Si = a[0] + a[0] + … + a[i]

A[i] + a[i +1] + ...  + a[j] = S[j] - S[i - 1]

Check if (Sj - Si - 1) greater than S


**384 Longest Substring without Repeating Characters**

A   b   b   a   c
^   
L

A   b   b   a   c
     ^   
     L

A   b   b   a   c
           ^   
           L

For checking repeating characters, need an array 
To store the characters the main and accessory pointers are pointing at 
Each increase the with the corresponding character

Int[] cnt = new int[256]
For accessory pointer, 
Cnt[sc[r]] ++

For main pointer, it is actually deleting the character
Cnt[sc[l]]--


Red stands for bad

If encounter sudden change from good to bad
Or sudden change from bad to good

Wanna find the longest

Left won’t move left. When right moves to right 

For loops  + while loops
Main pointer (right pointer)   accessory pointer (left pointer)

Two Pointers Template
For (left pointer, right pointer; right pointer < length; right pointer++) {

    While (some condition) {
        l++;
}
    // Update the answer
    
}

Two Pointers Template 3
**32 Minimum window Substring**

source : abczdedf target: acdd

Output: abczded

Source:    a     b     c     z    d    e    d     f
                 ^
                  L 

…

Source:    a     b     c     z    d    e    d     f
                                                         ^
                                                          L 

Freq[a] = 1
Freq[c] = 1
Freq[d] = 2
  
Use an array freq, to record the character, ch

If ch show w, as long as the freq[ch] increase to w, 

Target:     a     c     d     d


快慢指针
Find Nth node from the end of List
(fast pointer is leading n nodes before the  slow pointer)

FInd the middle of the linked list (fast pointer moves two steps, slow pointer moves one step)

616 Course Schedule II 
Ask the ordering of all the courses you should take, 
N = 4 prerequisites = [[0, 1], [2,0],[3,1],[3,2]] “to take course 0 you have to take course 1“
Think of it as a graph problem, and these nodes depend each other, so in order to get the order of the courses, use topological sort would be a good way 

The topological sort
Build graph
Build indegree graph 
Initialize a queue and put all the nodes with zero indegree into the graph
BFS search for other nodes if their indegree decreases to 1, put it into the queue




Red arrow means the indegree has been decreased to 0, put node 2 to queue



Put node 3 into queue


Put node 1 into queue

At the end all courses are sorted according to their prerequisites



**613 High Five**
[[1,91],[1,92],[2,93],[2,99],[2,98],[2,97],[1,60],[1,58],[2,100],[1,61]]
Id: 1 Highest five scores: 92 91 61 60 58
Id: 2 Highest five scores: 100 99 98 93 92 

For Heap, if the heap.size() < 5 add the item, check if the top of the heap is smaller than the next score, if yes then poll that top of the heap and add the new score


**612 K closest points**
Given some points and origin, find k points which are nearest to the origin, return the points sorted by distance


points = [[4,6],[4,7],[4,4],[2,5],[1,1]], origin = [0, 0], k = 3
Use Heap, if the heap size is smaller than k, than add the item, if the item is shorter than the top of the heap value, then the poll the top of the heap and add the new node.
PriorityQueue<> Stores the points, compare the points from small to large using the comparator, say you have two points, first you want to compare them with the distances from the origin. Then you want to compare their x values if their x values are not the same, finally, compare their y values if their values are not the same
Everytime update the answer with the queue poll 
123 Word Search 
["ABCE","SFCS","ADEE"]，"ABCCED"

[    
     A B C E
     S F C S 
     A D E E
]

A(0,0) -> B(1,0) -> C(2,0) -> C(2, 1) -> E(2,2) -> D(1,2)

## DFS in the matrix
When you see the grid character is matched with the word’s first character
For each char in the String, perform the dfs in the grid, need to search the grid for the four directions i - 1, i + 1, j - 1, j + 1

Here needs to mark the board[i][j] = temp, and set the board[i][j] = ‘ ’
And set the board[i][j] = temp back for back tracking


-----------------------------------------------------------------------------



**388 Permutation Sequence**

Given n and k, find kth permutations of the dictionary order in full permutation of n

Ex. n = 3, k = 4, 

Solution:

Find the candidates for all the permutations from 1 to 3

Solution 1. Brute-Force
“123”
For each digit, start from 1, find the permutation of 2 and 3,
                Start from 2, find the permutation of 1 and 3
               Start from 3, find the permutation of 1 and 2
After find all the permutations, get the k - 1 element 

O(N^3)

Solution 2. Use Recursion to find the permutation
Given we know the total numbers that we want to permutate, which is n
So we could convert them into String by using StringBuffer/StringBuilder

Then we have another StringBuffer/StringBuilder, output, to handle the character of the String converted from the previous StringBuffer

Then we need a boolean array to record if the digit character has been visited or not. 

Then we could start recursion to find the next permutation


“123” → permutate(“1”, “23”)
      → permutate(“2”, “13”)
      → permutate(“3”, “12”)

Each gets to the next permutation


Public class Solution{
    // O(N). S()
Public String getPermutation(int n, int k) {
If (n == 0) {
    Return “”;
}

If (k == 1) {
    Return String.valueOf(n);
}

StringBuffer sb = new StringBuffer();
For (int i = 1; i <= n; i++) {
    sb.append(String.valueOf(i));
}

// for example, Sb → “123”
    
    StringBuffer output = new StringBuffer();
    Boolean[] used = new boolean[n.length];
    
    List<String> candidates = new ArrayList<>();
    // sb, start, end, output, candidates
Permutation(sb.toString(), 0, n, output, candidates);
    
// because k starts from 1, not 0
    Return candidates.get(k - 1);
}


Private void permutation(String str, int index, int end, StringBuffer output, List<String> candidates) {
    // base case
    If (index == end) {
    candidates.add(output.toString());
}

For (int i = 0; i < end; i++) {
    // remove duplicate
    If (i > 0 && str.charAt(i - 1) == str.charAt(i) && !used[i - 1]) {
    continue;
}

    If (!used[i]) {
        output.append(str.charAt(i));
        Used[i] = true;
        permutation(str, index + 1, end, output, candidates);
        // back tracking
        Used[i] = false;
        output.remove(output.size() - 1);
}
}
}
}            

-----------------------------------------------------------------------------
MicroSoft Interview Problems (LeetCode)

Two sum
Use HashMap to store the numbers and their indices. If found the corresponding value, return the corresponding index from the map and the index i


Valid Palindrome (Microsoft Interview Problems)

To solve this problem, we need two pointers, one starting from 0 and another one starting from nums.length - 1. So we could take it into a while loop
As for example 

S = "A man, a plan, a canal: Panama"
Every S.charAt(l) == S.charAt(r), so the matching would be return true;

But also need to check if the character is actually digit or letter, not symbol or white space. This needs Character.isLetterOrDigit() to check

Then we would convert all the letters or digits to the lower cases to check if every of them is matched. If not, return false;
Time Complexity: O(N)
Space Complexity: O(N)

String to Integer (atoi)
The method here is to use a long integer as result. The thing is that using int would exceed the limit, then the 
Example 
“      - 4 2”
 0 1 2 3 4 5


First, you need to skip all the white space, till reach to the sign
In this example, i = 3 since it reaches the sign character

Second, you need to check whether the sign character is ‘+’ or ‘-’
Sign = 1; // initialize 

Sign = str.charAt(i) == ‘+’ ? 1 : -1 
Remember increase the i to the actual digit I++;

Use while loop to check if i < n and the Character.isDigit(s.charAt(i))

Use the horner’s rule to convert the string to the integer

Horner’s rule walk through

S = “321”
Res = 0
Res = res * 10 + s.charAt(0) - ‘0’
Res = 3
Res = 3 * 10 + 2 = 32
Res = 32 * 10 + 1 = 321


Check if the number converted is whether bigger than the MAX_VALUE or smaller than the MIN_VALUE; 
Check the sign if the sign is 1 then the number is MAX_VALUE, if not, it is MIN_VALUE


After the loops, return the int form of the res


Reverse a String 

Example 
[‘h’, ‘e’, ‘l’, ‘l’, ‘o’]
  ^                   ^
  L                   r

Swap
[‘o’, ‘e’, ‘l’, ‘l’, ‘h’]
       ^         ^
       L         r

Swap
[‘o’, ‘l’, ‘l’, ‘e’, ‘h’]
            ^
            L,r
Pointers meet up, end the process


Reverse words in a String

Str = “Sky is Blue”

String[] list = [“Sky”, “is”, “Blue”], which is split by white spaces


Then do the swap element in the array using two pointers
[“Sky”, “is”, “Blue”]
   ^            ^
   L            r

Swap

[“Blue”, “is”, “Sky”]
          ^
          L,r

And then use a StringBuffer sb  Q: why StringBuffer? Because it is suitable for multi thread application, StringBuilder itself is more suitable for single thread, but for this case, either is fine

For String item : reversedList 
    sb.append(item + “ ”);

But it will have one extra white space at the end, so it would be nice if we can remove it
Sb.remove(sb.size() - 1)

Finally return the answer sb.toString();

But this method is somewhat faulty since it won’t clear out the white space, instead, it add some white spaces, for “a good example” as instance. The space is one extra

Time Complexity:

Another Solution is Starts from the end of the String array, for loop back to 0, while check to see if the list[i] is empty, only do the append action when its not empty

Another way to check is see if the StringBuffer is non-empty, remember to add spaces back to the StringBuffer

Don’t do if-else for the checking the StringBuffer emptyness. This would cause appending of next String only once!



Reverse words in String II

Input: ["t","h","e"," ","s","k","y"," ","i","s"," ","b","l","u","e"]

Output: ["b","l","u","e"," ","i","s"," ","s","k","y"," ","t","h","e"]

 Now we can not use the split function to split them into words

We still need the reverse function to reverse all the characters from 0 and s.length - 1 first
We will have this below
[‘e’, ‘l’, ‘u’, ‘b’, ‘s’, ‘i’, ‘y’, ‘k’, ‘s’, ‘e’, ‘h’, ‘t’]

Then we would start to reverse every word in the char array
[‘e’, ‘l’, ‘u’, ‘b’, ‘ ’, ‘s’, ‘i’, ‘ ’, ‘y’, ‘k’, ‘s’, ‘ ’, ‘e’, ‘h’, ‘t’]
  ^         ^
  Pre       i

If pointer i hits the spaces, need to perform the swap
Example [‘e’, ‘l’, ‘u’, ‘b’]
          ^              ^
          L              r
        [‘b’, ‘l’, ‘u’, ‘e’]
               ^    ^
               L,   r 
Move the pre pointer to the i + 1
[‘b’, ‘l’, ‘u’, ‘e’ , ‘ ’, ‘s’, ‘i’, ‘ ’, ‘y’, ‘k’, ‘s’, ‘ ’, ‘e’, ‘h’, ‘t’]
                            ^         ^
                            Pre       i           
After swap
[‘b’, ‘l’, ‘u’, ‘e’ , ‘ ’, ‘i’, ‘s’, ‘ ’, ‘y’, ‘k’, ‘s’, ‘ ’, ‘e’, ‘h’, ‘t’]
                                           ^              ^
                                      Pre             i           
After swap
[‘b’, ‘l’, ‘u’, ‘e’ , ‘ ’, ‘i’, ‘s’, ‘ ’, ‘s’, ‘k’, ‘y’, ‘ ’, ‘e’, ‘h’, ‘t’]
                                                               ^         ^
                                           Pre       i
After swap 
[‘b’, ‘l’, ‘u’, ‘e’ , ‘ ’, ‘i’, ‘s’, ‘ ’, ‘s’, ‘k’, ‘y’, ‘ ’, ‘t’, ‘h’, ‘e’]
                     ^         ^
                    Pre        i

           
Valid Parentheses
Could use a Stack, called stack, to store all the left parentheses, such as “(“, “{”, and “[”

So given an example
“()[]{}”

Stack<Character> 
|  (  |
|  [    |
|  {    |
_______

Then see if the poll element from the stack is matched with any right parentheses, like “)”, when the item is “)” if the pop element is not “(”, then it is false, same with “{” “}”, and “[” “]”

If at the end we see there is nothing in stack meaning all the parentheses are paired, return true; if not, return false;

Longest Palindromic Substring
Str = “Babad”
DP or Two pointer starting from center at the same time

‘B’ ‘a’ ‘b’ ‘a’ ‘d’
 ^               ^
 i=0            J = 4
State: dp[i][j] represents if from i to j, whether substring from i to j is a palindrome.
And now the state transfer function is depends on the whether the str.charAt(i) and str.charAt(j), but also depends on the previous dp state, 
like from the i + 1 to j - 1 position
But there are some special cases we need to handle, like the in the example if 
‘B’ ‘a’ ‘b’ ‘a’ ‘d’
         ^   ^
        i=2  J = 3
I + 1 = 3 j - 1 = 2, this can’t be allowed in the state transfer function
So to constraint this, we need to say the size between the j and i are bigger than or equal to 2
Dp[i][j] = (str.charAt(i) == str.charAt(j)) && ((j - i <= 2) || dp[i + 1][j - 1])
If dp[i][j] == true
    Use a max value to find the maximum substring length
    Then use a String to update the result.
     Since the result is guaranteed, so this is ok to return the res at the end

Time: O(n^2) Space: O(n^2)

Solution 2 two pointers starting from the middle

When go to the 
‘B’ ‘a’ ‘b’ ‘a’ ‘d’
     ^       ^
     L       r
L = 2, r = 2
L-- r++
L = 1, r = 3
L--. r++
L == 0, r++
So final substring should be the s.substring(l + 1, r)

Time: O(N^2) because we query the element twice in the array
Space: O(1) because we did not open any new arrays







Trapping Rain Water
[0,1,0,2,1,0,1,3,2,1,2,1]
 0 1 2 3 4 5 6 7 8 9 10 11

Say we have leak from 1, 3 and 4, so the water would fill the left first, and then will fill the 3 and 4, but how to find the indices?

So if we first find the left highest array, called leftHighest, e.g, leftHighest[1] = max(leftHighest[0],heights[0])
 So leftHighest[1] = 0 leftHighest[2] = 1, … 
Then we want to find the right highest, which is only a single number, 3 in this case. To find if the water can fall down to the empty space, we could check if the minimum between the leftHighest[i] and the rightHighest is bigger or smaller than the height itself. 
Say we are in index 2 
[0,1,0,2,1,0,1,3,2,1,2,1]
 0 1 2 3 4 5 6 7 8 9 10 11
 
The left highest would be 1, and the right highest is 3, so the minimum of both left and right highest is 1, which is less than height itself, so the water can not fall onto this height, 

For index 5, however, the left highest is the 2, and the right highest is the 3, so the minimum is 2, and it’s bigger than 0. So the water it can store is 2 - 0 = 2;

Group all Anagrams
["eat", "tea", "tan", "ate", "nat", "bat"],

→ Sorted → aet, aet, ant, aet, ant, abt
→ map would be like 
Key: aet       ant         abt
     Eat       tan         bat
     Tea       nat
     Ate  


Solution 1.Use HashMap, first to sort the word alphabetically, and then store every word to the map value, obtained by getting the sorted string

Set Matrix Zeros
Solution1. Brute force
O(m + n)
Set two boolean arrays one representing the row, and the other representing the columns. The and when scan the zeros, record the position as true
Then traverse the row and column, if meet the marked row or column, mark that position as 0


[
   F T F
F  [1,1,1],F
T  [1,0,1],T 
F  [1,1,1] F
    F T F
]
Row = 1, col = 1
Then row =  {F, T, F} col = {F, T, F}
Then traverse if row[i] || col[j] → mark matrix[i][j] = 0

Solution2. 
Improve the Time
O(N) N is the width of the matrix
We don’t need to compute the array row everytime since we could mark that row with a single boolean indicating that row has a zero in it. Now we take care about the column, then we move down row by row, to see whether zero shows in the matrix

Solution3. 
Improve the time
O(1)
So we could use only the first row as the indicator of predicting whether the rest of the matrix would contain zeros or not. The
Dot Product
Inputs: two arrays, output: dot product
First check if two arrays dimensions are equal, if not return -1
Then for each element in each array, multiply it with each element in that array in the same index
A [1,1,1] B[2,2,2]
Dot product = A[0] * B[0] + A[1] * B[1] + A[2] * B[2]

Moving average from data stream
Which data structure would realize that? Using Queue
Since the data stream is changing, then we could check when the queue size is the same as the size, meaning we need to poll out the first old element and offer the new element. Then we could use the queue size and the sum of the queue element to calculate the average 
Compare Version Numbers
Split the version numbers by “.”, then look at the first-level revision,
If the first-level revision of Ver1 is already less than the one of Ver2, then return -1, if greater, return 1, if equal, compare the second-level revision, repeat the same steps
But this algorithm would not handle the “1” and “01” case

What if there is no “.” in the case?, that would return 0. Example “1”, “01”
After the split, it would become the [1] and [01]

Example 
“0.1”, “1.1”
arr1 = [“0”, “1”]
Arr2 = [“1”, “1”]
Arr1[0] < arr2[0] → -1

Rotate Image
[
  [1,2,3],
  [4,5,6],
  [7,8,9]
],


Outputs 
[
  [7,4,1],
  [8,5,2],
  [9,6,3]
]
(0, 0) → (0, 2)
(0, 1) → (1, 2)
(0, 2) → (2, 2)

(1, 0) → (0, 1)
(1, 1) → (1, 1)
(1, 2) → (2, 1) 
The last row → the first column

The middle row → middle column

The first row → last column
Solution 1. Transpose and reverse each row
[
  [1,2,3],
  [4,5,6],
  [7,8,9]
],

→ Transpose
 I = 0 j = 0 →  1 <--> 1
 I = 0 j = 1 →  2 <--> 4
 I = 0 j = 2 →  3 <--> 7
 I = 1 j = 1 →  5 <--> 5
 I = 1 j = 2 →  6 <--> 8
 I = 2 j = 2 →  9 <--> 9
Transpose: keep the diagonal elements as the same, then flip the element diagonally 
[ 
  [1,4,7],
  [2,5,8],
  [3,6,9]
],

→ reverse each row
I = 0 j = 0 swap with I = 0, j = 2 → (1) <--> (7)
I = 0 j = 1 swap with I = 0, j = 1 → (4) <--> (4)
I = 1 j = 0 swap with I = 1, j = 2 → (2) <--> (8)
I = 1 j = 1 swap with I = 1, j = 1 → (5) <--> (5)
I = 2 j = 0 swap with I = 2, j = 2 → (3) <--> (9)
Reverse each row is just to flip the elements in the middle as mirror
[
[7, 4, 1].
[8, 5, 2]
[9, 6, 3]
]
N = 2

This method costs O(N^2)
Transpose Swap(matrix[i][j], matrix[j][i]) (0 <= i < n  i <= j < n)


Solution 2. Rotate four rectangle 
 ---->
1  2  3  |
4  5  6  |
7  8  9  
< ---- 








Spiral matrix 
 [ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]
Solution 1. Simulation
We have 4 parameters, 
rowBegin = 0, rowEnd = matrix.length - 1;
columnBegin = 0, columnEnd = matrix[0].length - 1;
Then simulate 
Move right → Move down → Move left → Move up

But remember to check when at the last row, need to compare if the last row is still bigger or equal to the first row
Also, when back at the first column, need to check if the column begin is still less or equal to the column end. 

Solution 2. Simulation but use directional matrix dx and dy
Now we could use a directional matrix to simulate the same process as the previous solution. The solution also uses a boolean matrix seen to mark if the cell is seen or not. Our current position is (x, y), and our matrix is R * C, , meaning we need to visit R * C cells.Our next possible positions would be x + dx[di], y + di (di starts from 0) if out of bound, the di would increase by one. Also note that the di would reset back to 1 if is 5 by modding 4


-----------------------------------------------------------------------------
Linked List

Reverse Linked List

1 -> 2 -> 3 -> null
Need three ListNodes to store its previous node, its current node, and its next node.
null       1       2 -> 3 -> null
Pre       cur    next

null   <-  1   <-    2  <-  3     null
                      Pre    cur    next
Linked List Cycle
First to ask: input? ListNode head output? Boolean value
Two pointers and check if they could meet
One is fast another is slow,
If they could → the cycle exists
It they couldn’t → the cycle not exists

Special case to consider,
 When there is guaranteed there is no cycle?
When there is not node
When there is only one node

Add two Numbers
Inputs? Two linked lists Outputs? One Linked List
Sorted in reversed order

Example
(2 -> 4 -> 3) + (5 -> 6 -> 4)
      L1              l2 
New Linked List Head = New ListNode(l1.val + l2.val)
342 + 465 = 807
7 -> 0 -> 8
Solution: Simulate the adding process
So traverse the two linked lists at the same time. So there are several
The sum would be the l1.val + l2.val + carry but here we need to check if the l1 value or the l2 value are equal or not. Say
3 -> 1 -> 9 and 4 -> 2 -> 4 -> 6
L1 doesn’t have the 4th node, so set the 4th node in l1 would be nice

If the digit needs to carry, to find the carry for each loop, we would 
Use current sum divided by 10, then the last digit will be current sum mod 10
Also we need to take care of the left carry 
Say
900
   900
   1800
1 here is the carry got left

So the 1 needs to be a new node so the new linked list would point to it

Add two Numbers II
(7 -> 2 -> 4 -> 3) + (5 -> 6 -> 4)
(7 -> 2 -> 4 -> 3) + (0 -> 5 -> 6 -> 4)
7243 + 564 = 7807
(7 -> 8 -> 0 -> 7)
Solution 1. Reverse the linked list first, and then repeat what we did on the Add Two Numbers

Solution 2. Do it without modifying the linked list, not reversing the linked lists, so we need Stack to realize the “reverse” functionality

Merge two sorted Linked List
1 -> 2 -> 4
1 -> 3 -> 4
Inputs? Two linked lists
Outputs? New linked list

Solution 
Traverse two list, compare each node to decide which would put to the next node of the new list.
If the l1.val <= l2.val new list.next = l1 l1 = l1.next else new list.next = l2, l2 = l2.next
At the end we should handle a situation where two lists are not equal in length such as 1 -> 2 -> 3 and 1 -> 2 -> 3 -> 4

Dummy -> 1 -> 1 -> 2 -> 3 -> 4 -> 4
New list 

Merge K Sorted Lists 
Merge K sorted Lists, meaning that we need to take care a more generalized case than the Merge two sorted Lists. 
Inputs? K lists
Outputs? One merged Linked List
Solution 1. Brute-Force 
Traverse all the linked lists and collect values and put them into a array
Sort and Iterate over this array to get the proper value of nodes
Create a new sorted linked list and extend it with new nodes
O(NlogN) Collect O(N) Sort O(NlogN) Iterate new linked list O(N)

Solution 2 Compare K node one by one
O(kN)

Solution 3 Use PriorityQueue<> for the comparison in algorithm 2
This would cost O(Nlogk) since we use O(logK) to insert node into priority Queue every loop. But finding the node with the smallest value would only cost O(1) time

Intersection of Two Linked Lists
Solution Two pointers
Given 
A -------------------null ptA
          |ptB 
          |
          |
          B
          ptA
Every time ptA hits the end of the linked list, make it to the beginning of the B linked list. Same with the list B, if list B hit the end of the linked list, connects it to the head of the list A. If they can meet, that means they are intersected, return a, otherwise, return 
Time O(N) Space O(1)

Copy List with Random Pointer
Example
1  -> 2  -> 3

1’ -> 2’ -> 3’

HashMap 
Key   1   2   3 
Value 1’  2’  3’
Every Time we want to find the next node, we assign that next node to the value in the HashMap

map.get(node).next = map.get(node.next)

Same for the next random pointer, we assign the next node to the value in the HashMap

map.get(node).random = map.get(node.random);



But time Complexity is O(N)

O(1) Solution

1.Build a linked list such as
[1 -> 1’ -> 2 -> 2’ -> 3 -> 3’]
2.Copy the random pointer
If known 1 >> 3
1.next -> 1’ and 1.random -> 3 
3.next -> 3’ 1’.random -> 3’
3. Split the list A
[1’ -> 2’ -> 3’]

Cur     -> 1
newNode -> 1 
newNode.next = cur.next
1.next = 1’
1 -> 1’
Cur.next = newNode
1’ -> 2
So the first section of the code is 
1 -> 1’ -> 2

Then we want to move the head forward two steps
Head = head.next.next
Step 2. CopyRandom method
[1 >> 3]
[1’ >> 3]
1 -> 1’ -> 2 -> 2’ -> 3 -> 3’

So current node’s next node’s random is the current node’s random node’s next

Cur.next.random = cur.random.next;

move pointer two steps since its counted 2 steps as one unit

3. Split the list

Split the list as the 1’ is the return value

Validate Binary Search Tree
Each check if the right.val > node.val && left.val < node.val
But not only the right child should be larger than the node but all
The elements in the right subtree,

So need two limits to keep track of the comparison between the two
The left subtree must be within the Long.MIN_VALUE and the root val, 
And the right subtree must be within the root val and the Long.MAX_VALUE


Binary tree Inorder Traversal
Basic DFS root.left -- root -- root.right

Binary Level Order Traversal
Basic BFS Implementation -
Use queue.size in the for loop

Binary Tree Zigzag order Traversal
           3
     9             20
                15      7

Idea: We could have two stacks, oldStack and newStack
To push and pop nodes exchangebly. So if old stack is not
Empty, we push left node to the new stack and then push right
Node to the new stack. Both would be non-empty
Res = {[3], [20, 9], [15, 7]}
Temp = {15, 7}
oldStack -- 3 pop {} 7 15 pop 7 
newStack -- 9 20 pop 9 pop 
So when newStack is not empty, the oldStack should add right first
Then add left first. Given both are non-empty



Populating Next Right Pointers in Each Node 






**Clone Graph**
Solution 1. DFS
Map to record the node label and the node itself
Then we recurse to find the node’s neighbors, for each neighbor, we 
Check if it has been visited, if yes, then for the tremp node would be
The value of the old node’s neighbors in the map add the tempNode to new node
If not, temp node would be a new created node with the old node’s value, 
Mark in map as visited, and dfs(oldNode.neighbor.get(i), temp)

Solution 2. Use BFS to solve this questions
need a map to record the old node and new node
Also need a queue 
Map 1 1 
Queue: 0
Example
graph
0 1 2 #

Clone nodes
pop    Queue        Map
        0          0  -> null
0      1   2        0 -> 0, 1 -> null, 2 -> null 
1        2          0 -> 0 1 -> 1 2 -> null
2        Empty      0 -> 0 1 -> 1 2 -> 2

Clone edges
    Map
    Key
    0 -> 0’          
      1 -> 1’
    2 -> 2’

                      



Construct Binary Tree from preOrder and Inorder Traversal
preOrder root left right so we know the first element is the
Root
inOrder left root right

The root found in the preOrder list splits the inOrder list into 
Two subtrees. Left_preorder and left_inorder = new int[index]
Right_preorder and right_inorder = new int[n - 1 - index]
After find these four, we could use recursion to reconstruct the 
Tree. Left subtree for the preorder would be from 1 to index - 1
Left subtree for the inorder would be from 0 to index - 1
For the right subtree for the preorder would be index + 1 + i
For the right subtree for the inorder would be index + 1 + i

Number of Islands

Solution: 
BFS in the matrix 
Use a directions arrays and the boolean matrix 
To mark as visited
Use two queues to store the coordinates respectively,
Go through each cell in the matrix and for each cell that is island, we perform the BFS to search the surrounding cells.


Longest Common Ancestor in BST
What is the LCA? The node where the p and q shares the 
If the p.val < q.val, meaning p could be q’s root or q’s left neighbor
This can be done using the method in Longest Common Ancestor in Binary Tree




Lowest Common Ancestor in Binary Tree
Solution 
Example: 
given 2’s Ancestors are {2, 5, 3} and 6’s ancestors are {6, 5, 3}
So their common ancestors are {5, 3}
Given 2’s Ancestors are {2, 5, 3} and 0’s ancestors are {0, 1, 3}
Their common ancestors are {3} only

Given 0’s Ancestors are {0, 1, 3}, and 8’s ancestors are {8, 1, 3}
Then the common ancestors are {1, 3}

So the rules are if the query nodes are both on the left subtree, the LCA must be in the same left tree, including root. 
If the query nodes are both on the right subtree, the LCA must be in the same right subtree, including the root’
If the query nodes are on the different subtree, the LCA must be the root


We could use recursion to find which subtree the query nodes are

Populating Next Right Pointers in Each Node (Perfect Binary Tree)
Level Order Traversal
Or preorder traversal, put it in every array. Time O(N), Space O(N)

Local view: Preorder view would connect the left and right subtree 
Cur.left.next = cur.right
Cur.right.next = cur.next.left
Time: O(N)
Space: O(logN) binary tree max depth is logN


Populating Next Right Pointers in Each Node II
BFS level order traversal
Level            Queue
0            {1}, 
1            {2, 3}, 
2            {4, 5, 7}

Node.next = queue.peek();
For level 1, when the current node is queue.poll(), which is 2, so the queue only has 3, which is the 2.next, given the level is greater than 1



-----------------------------------------------------------------------------
BackTracking
Letter Combination of phone numbers
You have a phone that have a numpad with corresponding letter, so you want to find when the user pressed some numbers, what letter combinations would come out?



Solution 1. DFS every number and backtracking
Input? String numbers
Output? List<String> of different characters combined
First we could create the phone book recording the letter on each digit
Then for each character in the given, like “23”
‘2’ could convert to the corresponding number, and that index can use to in the phone book array to find the String s, and for each character, we can add it to a temporary String called cur, which act like a possible candidate and would be added to the ans 
2 -> “abc” -> ‘a’, ‘b’, ‘c’
3 -> “def” -> ‘d’, ‘e’, ‘f’

Cur 
‘Ad’,’ae’,’af’
‘Bd’,’be’,’bf’
‘Cd’,’ce’,’cf’








Word Search
Find Word Trie
board = [
  ['o','a','a','n'],
  ['e','t','a','e'],
  ['i','h','k','r'],
  ['i','f','l','v']
]

words = ["oath","pea","eat","rain"]

To find a word, we could use Trie, and use DFS for the searching

WildCard Matching 
DP 
Boolean[s.length() + 1][p.length() + 1]
B[i][j] ---  
Initialize
If [i - 1] == ‘*’
[0][i] = [0][i - 1]
p\s         a        b         c
    T    F        F        F
* 1    F    \    
A 2    F            \
* 3  F                    \

2. If p match with s or p is ‘?’
The direction would be (from top left to bottom right)
S : xxxb
P: xxb

3. P[i] = ‘*’
Star can be empty character s: ab p: ab* (from up to bottom)
Star can be any sequence (from left to right)

Regular Expression Matching
First Mind goes is using recursion 
If we have ‘*’ in the string, the ‘*’ has to be in the second position of pattern. So we can ignore that part first or delete a matching character in the text

Solution DP
Boolean[][] dp = new boolean[lenS + 1][lenP + 1]
S: [abc]dc dp[3][3] = true [3][3] are the numbers of characters  
P: [abc]


-----------------------------------------------------------------------------
Sorting and Searching
Colors: [0, 1, 2]
        [r, w, b]
Example: [2, 0, 2, 1, 1, 0]
Solution 1. Brute force
        Two passes
        Count 0s, 1s and 2s
        And overwrite the array with the numbers
        O(N^2) Space: O(1)
Solution 2. Use partition twice, 
[2, 0, 2, 1, 1, 0]
          ^
[1, 0, 0, 1, 2, 2]
first partition would be choosing 1 as the pivot, but we still need another partition since right now we only partition the 1s and 2s. So now we could use 2 as pivot, so those 0s and 1s would be grouped correctly
Final -->
[0, 0, 1, 1, 2, 2]
// partition walk through
We could have 2 pointers, one left one right
[2, 0, 2, 1, 1, 0]
 ^              ^
 L              r

Exchange
[0, 0, 2, 1, 1, 2]
    ^        ^
    L        r
Continuously check if nums[l] < nums[r], given l is still smaller than r
While checking, do two whiles loops inside
First while loop → Move the left pointer to right if the nums[l] is still smaller than the pivot → find the first number greater than pivot

Second while loop → Move the right pointer to the left if the nums[r] is still bigger than the pivot → find the first number smaller than the pivot
Any nums[l] and nums[r] that are not qualified, swap the their values and move the left and right pointers

Language: JAVA
Public class Solution {
    Public void sortColors(int[] nums) {
        If (nums == null || nums.length == 0) return;

        partition(nums, 1);
        partition(nums, 2);
}

Private void partition(int[] nums, int pivot) {
    Int l = 0, r = nums.length - 1;
    While (l <= r) {
    While (l <= r && nums[l] < pivot) {
    i++;
}

While (l <= r && nums[r] >= pivot) {
    j--;
}

If (i <= j) {
    Int temp = nums[i];
    Nums[i] = nums[j];
    Nums[j] = temp;
}
}
}
}

Find Minimum in Rotated Sorted Array
Sorted, find minimum… classic binary search problem

[0, 1, 2, 3, 4, 5, 6, 7]
May become
[4, 5, 6, 7, 0, 1, 2]
 ^        ^        ^
 l        m        r
[4, 5, 6, 7, 0, 1, 2]
          ^    ^   ^
          l    m   r
[4, 5, 6, 7, 0, 1, 2]
          ^  ^  ^   
          l  m  r   
[4, 5, 6, 7, 0, 1, 2]
          ^  ^  ^   
          l  r
Nums[l] > endValue → not nums[l]
Return nums[r] → 0 which is correct 


Initialized Target value would be nums[r] == 2
And check if nums[mid] is  smaller/bigger than target

And the end only the l and r hasn’t been evaluated
If the nums[l] <= target, return nums[l]

Return nums[r] at the end

Public class Solution {
    Public int findMinimum(int[] nums) {
    Int l = 0;
    Int r = nums.length - 1;
    Int target = nums[r];

    While (l + 1 < r) {
    Int mid = l + (r - l) / 2;
    If (nums[mid] > target) {
    L = mid;
} else {
    R = mid;
}
}

If (nums[l] <= target) {
    Return nums[l];
}

Return nums[r];
}
}


Find Minimum in Rotated Sorted Array
Duplicates exist, so to resolve that we might need to remove the duplicate or think about a new approach instead of binary search

But this problem would be focus on if you can think of the worst situation where all other element in the array is 1 but left one element is 0

Search a 2D matrix
Binary search each row and each col

After searching the row (1st search), set the row to be either start or end, else return false directly
Always associate the binary search method with the search when you read a dictionary. 
If you find the target word is after what your current open page, move your left hand!
 If the target word is before what your current open page, move your right hand!

Search a 2D matrix II

[
      [1, 3, 5, 7],
      [2, 4, 7, 8],
      [3, 5, 9, 10]
    ]
    target = 3


Find the median of two sorted arrays
[1 3 7 8 18]
[2 9 12 25]

Solution 1. Merge the two sorted lists together, and to find the median
1 2 3 7 8 9 12 18 25

But how could we find the median without merging it?

Solution 2. 
Find the median of two sorted array
[1 3 7 8 18]
[2 9 12 25]

1 way to think of is to merge two sorted list together, 
So the list would be 
[1, 2, 3, 7, 8, 9, 12, 18, 25] 

But how do we know the median without merging?

So if we can find a cut where max[array1 left] < min[array2 right] && min[array1 right] > max[array2 left]
If the even numbers, median would be (Math.max(maxLeftX, maxLeftY) + Math.min(minRightX, minRightY)) / 2

If the odd numbers, median would be Math.max(maxLeftX, maxLeftY); 

Arr1 1    3    7    8    |    18 
Arr2 2                |    9    12    25

Public Solution {
    Public int findMedian(int[] arr1, int[] arr2) {
        If (arr2.length > arr1.length) {
    Return findMedian(arr2, arr1);
}
    
        Int x = arr1.length;
        Int y = arr2.length;

        Int low = 0;
        Int high = x;

        While (low <= high) {
    Int partitionX = (low + high) / 2;
    Int partitionY = (x + y + 1) / 2 - partitionX;
    
    Int maxLeftX = (partitionX == 0) ? Integer.MIN_VALUE : arr1[partitionX - 1];

Int minRightX = (partitionX == x) ? Integer.MAX_VALUE : arr1[partitionX];


Int maxLeftY = (partitionY == 0) ? Integer.MIN_VALUE : arr2[partitionY - 1];

Int minRightY = (partitionY == y) ? Integer.MAX_VALUE : arr2[partitionY];

If (maxLeftX <= minRightY && minLeftY <= minRightX) {
    // even numbers
If ((x + y) % 2 == 0) {
    Return (double) (Math.max(maxLeftX, maxLeftY) + Math.min(minRightX, minRightY)) / 2;
} else {
    Return (double) Math.max(maxLeftX, maxLeftY);
}
} else if (maxLeftX > minRightY) {
    // it is too large for the search, shrink down the right region
    High = partitionX - 1;
} else {
    Low = partitionX + 1;
}
}

Throw new IllegalArgumentException();
}
}

----------------------------------------------------------------------
Dynamics Programming
Best time to buy and sell stocks
Dp[i]: the maximum profit for the day i 
Substructure: if want to find the best time for i-th day, so it could find the best time for ith - 1 day. 

To find the best max profit, calculate the price difference between the i-th day and i-th - 1 day to get the maximum profit, so the state transit would be 
Dp[i] = Math.max(dp[i - 1] + prices[i] - prices[i - 1], prices[i] - prices[i - 1])

Initialize dp[0] as 0
Find dp[prices.length - 1];

Run example 
[3, 2, 3, 1, 2]
Dp = {0, 0, 0, 0, 0}
Dp[1] = max(dp[0] + -1, -1) = -1
Dp[2] = max(dp[1] + (3 - 2), (1)) = 1
Dp[3] = max(dp[2] + (1 - 3), (1 - 3)) = -2
Dp[4] = max(dp[3] + (2 - 1), (2 - 1)) = 1

Finally sold it on 4th day, which is 1

Longest Increasing Sequence
[5,4,1,2,3]

Dp[i] the LIS of the first i number with the end on ith number
In order to find the LIS of dp[i], the substructure would be to start off finding the dp[i - 1] which is the first LIS of the first i - 1 numbers with the end of i - 1 number. The state transit would be compare the i-th LIS length and its previous LIS length
Dp[i] = Math.max(dp[i], dp[j] + 1) given 0 < j < i and nums[j] < nums[i]

Init dp[0] = 1, dp[1] = 1

-----------------------------------------------------------------------------

Google OA predict
Minimum Domino Rotations
A = [2,1,2,4,2,2], B = [5,2,6,2,3,2]
1st position and 3rd position would swap

Solution 1. Brute Force
A [2,1,2,4,2,2]
     ^   ^
B [5,2,6,2,3,2]

Check if surrounding indices are the same or not. If the index are not the same; go to B for checking if the same rule applies, 
1 is not the same as surrounding, 2 so look at B to check if it is 2 on the same position, if yes swap

Same applies on 3rd position on A, which is 4



Public Solution {
    Public int minDominoRotations(int[] A, int[] B) {
If (A.length == 0 || B.length == 0) return -1;
Int ans = 0;
For (int i = 1; i < A.length - 2; i++) {
    If (A[i] != A[i - 1] && A[i] != A[i + 1]) {
        If (B[i] == A[i - 1]) {
    swap(A, A[i], B[i]);
    ans++;
}
}
}
Return ans;
}
}

Solution 2. Greedy 
3 situations to consider 
A can all turn to be the same equal to A[0]
B can all turn to be the same equal to B[0]

Make the all the elements of A or B to have the same A[i] and B[i]
Algo pseudo code
Take A[0] and B[0] element
Check if one can make all elements in A to be A[0], if yes, return the minimum number of rotation needed
Check if one can make all elements in B to be B[0], if yes return the minimum number of rotations needed
Otherwise return -1
A [2,1,2,4,2,2]
           ^
B [5,2,6,2,3,2]
           ^
rotationA = {0,1,1,2,2,2}
rotationB = {1,1,2,2,3,3}
Min(rotationA, rotationB) = Min(2, 3) = 2

Coding style
Public class Solution {
    Public int solution(int[] A, int[] B) {
        If (A.length == 0 || B.length == 0) {
    Return 0;
}
        Int rotationsA = 0, rotationsB = 0;
        Int a0 = A[0], b0 = B[0];
        Int n = A.length;
        Int count = 0;
// B → B[0]
}

// A → B[0]
// start: A[0]/B[0], B
Private int check(int start, int[] B, int[] A, int n) {
    Int count = 0;
For (int i = 0; i < n; i++) {
    If (B[i] == start) {
    continue;
} else if (A[i] == start) {
    count++;
} else {
    Count = n + 1;
    break;
}
}
Return count;
}
Public int main(int[] A, int[] B) {
    // edge cases
If (...)
// 3, 4
Int count1 = check(A[0], B, A, A.length);
Int count2 = check(B[0], B, A, A.length);
// 1, 2
Int count3 = check(A[0], A, B, A.length);
Int count4 = check(B[0], A, B, A.length);
Int ans = Math.min(Math.min(count1, count2), Math.min(count3, count4));
Return ans == n + 1 ? -1 : ans;

Int resultA = 0, resultB = 0;
If (A[0] == B[0]) {
    Return count;
} else {
    Return Math.min(count, check(B[0], B, A, A.length));
}
}
}







Public class Solution {
    Private int check(int start, int[] A, int[] B, int n) {
        Int rotationA = 0, rotationB = 0;
        
        For (int i = 0; i < n; i++) {
    If (A[i] != x && B[i] != x) {
return -1;
} else if (A[i] != x) {
    rotationA++;
} else if (B[i] != x) {
    rotationB++;
}
}

Return Math.min(rotationA, rotationB);
}
    Public int findDominoRotations(int[] A, int[] B) {
            Int n = A.length;
Int rotations = check(A[0], B, A, n);            
If (rotations != -1 || A[0] == B[0]) {
    Return rotations;
} else {
    Return check(B[0], B, A, n);
}
} 
}
}


Watering Plants
Go from the end to the beginning

// [4, 6] m = 3 step = 2
//  4  3  C = 0, step = 2 + 2 = 4, C = 3
//  4  0  plants[1] = 0, i = 0, C = 0
//  4  0  C = 3 step = 4 + 1 = 5 
//  1  0  C = 0 step = 5 + 1 = 6, C = 3
//  0  0  C = 2 step = 6
// ans = step * 2 - n = 12 - 2 = 10
O(nk / m) k is the average time to water a plant
Public class Solution {
    Public int solution(int[] plants, int m) {
If (plants.length == 0 || plants == null) {
    Return -1;
}

Int n = plants.length;
Int ans = 0;
Int step = n;
Int C = m;
Int i = n - 1;
While (i >= 0) {
    Int temp = Math.min(plants[i], C);
    Plants[i] -= temp;
C -= temp;
    If (plants[i] == 0) {
        I--;
        continue;
}
If (C == 0) {
C = m;
Step += i + 1;
}
}
Ans = step * 2 - n;
Return ans;
}
}



Run example
[2, 4, 5, 1, 2]
C = 6 steps = 1
I = 0 → C = 6 - 2 = 4 steps = 1 + 1 = 2
I = 1 → C = 4 - 4 = 0 steps = 2 + 2 * (1) = 4 C = 6*
I = 2 → C = 6 - 5 = 1 steps = 4 + 1 = 5
I = 3 → C = 1 - 1 = 0 steps = 5 + 2*(3) = 9 C = 6*
I = 4 → C = 6 - 2 = 4




Combinations 
N and k return the combinations 
N = 4 k = 2
[[1, 2], 
 [1, 3], 
 [1, 4], 
 [2, 1], 
 [2, 3], 
 [3, 4]]

Every time we choose a number to be the next combination, the k needs to decrease by 1, meaning our choices are decreased by 1 and the level actually increase

Convert Binary Search Tree to Circular Doubly Linked List
We can do a postorder traversal to traverse all the nodes in the tree and 

       4
       /  \
      3   5
     / \
    1   3




Task Schedule 
There must be at least n interval CPU are doing different tasks between two same tasks. Find least number of intervals for CPU to finish all the tasks

[‘A’, ‘A’, ‘A’, ‘B’, ‘B’, ‘B’] n = 1
A -> B -> A -> B -> A -> B

[‘A’, ‘A’, ‘A’, ‘B’, ‘B’, ‘B’] n = 2
A -> B -> idle -> A -> B -> idle -> A -> B
Because two same tasks must contain n intervals, each interval can either finish a task or be idle
Solution 1
Brute force
[A, A, A, A, A, B ,B] n = 2
A -> B -> idle -> A -> idle -> B -> A -> idle -> idle -> A -> idle -> idle -> A

Intervals: 14
Increase A by 2, A becomes the most frequent word, so the interval time would dependent on the most frequent word

So in further inspection, for the most frequent word, the count is repeated as count[most frequent word] - 1, and for largest interval, the size would be n + 1, and for the last interval, we need to add the numbers that from other tasks. Which would be the extreme case where each character shows up in the tasks (A~Z) minus the number of the first max count letter index



Restore IP address
DFS -- compute all possible valid ip
Tips about the Valid IP address:
Each subset would be in the range [0, 255]
Each IP address would be from 0.0.0.0 to 255.255.255.255 so after removing all the dots, the length of the IP address would be from 4 to 12. 
Each address is made up of 4 subsets

The essence to it is to use dfs to extract each segment (1~3 long) from the input String s, 
Given “25525511135”
Make the recursion call, the example would lead to 
First level → 2, 25, 255
Second level → 2, 25, 255
Third level → 1, 11, 111
Forth level → 3, 135, 35

 So the dfs process would be each time the substring from 0 to i + 1 (0 < i < s.length()), check if the level is smaller than 3 if yes → the prefix needs a ‘.’ if no → the prefix is already contains the ‘.’ 


Merge two sorted arrays
Use two pointers to compare the arrays elements
And check if there are any elements got left from the two arrays;

Maximum Subarray
Input int[] nums, output: int maximum sum
Algorithm:
If we see the previous sum is already 0, we can turn that previous sum to be 0; else we can add current number to the previous sum’
If we see the maximum is less than the previous sum, then we can replace the maximum with the previous sum; else we can keep the maximum

Dynamic Programming 
Dp[i] -- max sum from 0 to i
Dp[i] = (dp[i - 1] > 0) ? nums[i] + dp[i] : nums[i]

Dp[0] = 0

Finally find the max element of the 

N-Queens
Say the chess board is 4 by 4


































[Q Q X X X]
[X X X X X]
[Q X X X X]
[X X X Q X]
[X X X X X]
So everytime we place one queen on the board, we need to take care about the other 

Single Number 
Given you a 1d array, check the only integer that appears once in the array
Solution: 
#1
use hashset add method only add the number that never exist in the set. So if the set can’t add the number to the array then, we need to remove that number from the set since it is duplicate already. 

#2 XOR operation
Same XOR Same = 0 Same XOR else = else

Find the Celebrity
Given an integer n, find if there is a celebrity that everyone else knows him or her but he/she doesn’t know anybody else

Solution
First find a candidate that someone knows by updating the parameter if it knows about someone else
For each other person there are two situations to consider for -1
2a. The candidate knows about other people
2b. Other people doesn’t know about the candidate


Reverse a LinkedList
        //  null <- 1 -> 2 -> 3 -> 4 -> 5 -> NULL
        //   ^      ^    ^
        //   prev cur  next
        //  null <- 1 <- 2 <- 3 <- 4 <- 5 -> null
        //                              ^    ^    ^
        //                             prev cur  next
        // from where to where to reverse?
        // we want to reverse each two nodes
        // 1. take down the cur next node since we are
        // 2. break the connection between every two node and point from the current back to prev
        // 3. move the prev to current, move current to next
        // 4. after the while loop, return the prev

Equal Tree Partition
The tree’s sum can be computed using dfs algorithm, and the intermediate sums can be stored using a HashMap, the value of the map would be the visited number, if the hashmap only contains 0 sum, then we need to check if this tree’s 0 sums occurence is greater than 0; else we need to check if the sum /2 is in the map and the sum itself can be divisible by 2



Word ladder
Given? Two strings startWord and endWord
Outputs? Shortest length of transformation sequence from start to end
Only one letter can be changed at a time
Each transformed word must exist in the dictionary
Duplicates/same? No
String has digits or special characters? No it does not have any digits or special characters.
String has spaces in it? / String would be all in lowercases? All in lower cases and no spaces 
Strings have smae lengths? Different lengths? Same lengths 
Return if we can’t find short sequence? Retrn 0


Example
Begin: “hit” end: “cog”

word List: ["hot","dot","dog","lot","log","cog"]

Hit -> hot -> dot -> dog -> cog 

Solution BFS 
Treat each word as node, each of node would be changed in one characters,
For each character in our node, we can have 25 outputs in next word 

Edges cases
If the wordlist is empty -- return 0
If the start word is equal to end -- return 1



Node into a queue, 

Public class Solution {
    Public int wordLadder(String s, String t, List<String> wordList) {
        If (start.equals(end)) {
    Return 1;
}

If (wordList.length == 0 || wordList == null) {
    Return 0;
}


wordList.add(s);
wordList.add(t);
        
Queue<String> queue = new LinkedList<>();    
        Set<String> set = new HashSet<>();
Int ans = 1; // 1 stands for the initial length of the transformation sequence 
        queue.offer(s);
        set.add(s);

        While (!queue.isEmpty()) {
            // why level order traversal?
    Int n = queue.size();
    For (int i = 0; i < n; i++) {
    String cur = queue.poll();
    If (cur.equals(t)) {
        Return ans;
}

For (String nextWord : getNextWord(cur, wordList)) {
    If (!set.contains(nextWord)) {
    queue.offer(nextWord);
    set.add(nextWord);
}
}

ans++;
}

Return 0;
}

Private List<String> getNextWord(String word, List<String> wordList) {
    List<String> ans = new ArrayList<>();

    For (int i = 0; i < word.length(); i++) {
        For (char c = ‘a’; c < ‘z’; c++) {
    If (c == word.charAt(i)) {
continue;
}

Char[] sc = word.toCharArray();
Sc[i] = c;
String newWord = new String(sc);

If (wordList.contains(newWord)) {
    ans.add(newWord);
}
}
}

Return ans;
}
}
}

Time complexity? 
O(n * k * 26) -- O(nk) n stands for the length of the wordList. K stands for the length of each word
Space Complexity?
O(n + n)      -- O(n)

Follow up? Instead of outputing the shortest transformation sequence
I want you to output all sequence in List<List<String>>
ID A* search
Add a surrounding around the DFS
The outer ring would cost the most but it is ok
## 二分查找/搜索小结
二分法有4种写法，然后三个迭代写法，一个递归写法，
然后二分法一般用于一维数组，但是也可以用在二维数组的查找

二分数组的查找的最优解就是二分查找
这里有个小技巧： 如何通过一维转二维？答案如下


**240 Search a 2D matrix II**
思路就是二分，但是要按照行和列来分别进行 这里可以设置一个flag表示是行还是列，然后根据这个去进行查找


## 排序/Sort小结
**148 Sort List**
这里是比较经典的merge sort 的应用，这里的merge 和sort 是分开走的，因为这里链表是单向的，所以我们首先要考虑将一个链表一分为二, 如何办到?
这里的标准就是我们将慢指针为头结点，快指针也为头结点，然后每次快指针走两步，慢指针走一步，还有就是有一个pre 指针，每次指向慢指针，最后当快指针到链表尾部的时候，pre 的下一个节点指向null, 这样就可以分开成两个sublist {head, slow} 和 {slow, fast}. 这里的两个sublist
通过recursion 来sort，base case 是什么？ 这里的base case 是当head 为null 时，返回head，或者head.next 为空时，返回head、

Sort 完之后，我们就来merge linked list
这里我们有两个linkedList，这里的我们首先要判断哪个头结点比较小， 然后assign 完小的头结点给LinkedList 后， 我们可以将剩下的节点依次 塞入到res 中，最后还要判断左跟右是否有剩，如果有剩，那么就还是加入到res 里面

递归如下


**369 Plus One Linked List**
 Plus One 
        think of it like slow and fast runners
        
        the slow represents the last node before the tail 9s, where fast represents the last node of the list
        if fast does not equal to 9, just increase it by one and return
        
        else then set sublist of {slow, fast} by increasing the slow number by 1 and set all the nodes behind to 0s
        
        The use of Dummy node
        The dummy node would handle if the spaces expanding problems, such as 9 -> 9 -> 9
        if it's the case, then return dummy (1 -> 0 -> 0 -> 0)
        
        else return the dummy.next (9 -> 0 -> 0)

Why initialize slow and fast both dummy node first?
because we need to check if the linked list needs a extra cell when encounter 9->9->9

如何按照字典序排序呢？

 
## 数学题小结
**7 reverse number**
扩大处理, res 用long表示，然后只要越界，就返回0
每次更新res是用res = res * 10 + num % 10 , num /= 10 
比较处理
比较与原来的值和新的值，利用最大值加一些值变负数，如果新的值除以10不等于旧值，那么就返回0

**plus one 加减乘除**
这里考虑进位的问题，如果是小于9的话，正常进位并返回res，否则就将该位数字改为0，
最后如过整个for loop都结束的话，那么就需要开一个新数组，然后新数组数字多一，然后数组头设1，最后返回
String to Integer (atoi) 转换
首先将str预处理把所有空格全部拿走，然后将第一个char 拿出来看正负，然后定义第一个数字的位置为符号的后面以为，然后开始进行for loop，每次res = res * 10 + str.char(i) - ‘0’, 最先加进来的数字一定是最大的那一位，其中判断数字是否越界，这里用sign和res来判断，res实现要做一个扩大处理，最后返回int型res * sign

**367 Valid perfect square 开方**
三种解法， 1.二分法， 必须要会， 2. x * x > num 时间比二分法少一点 3. 牛顿法，非常难以解释，一般除非面试官特别要求，不需要写这个解法
67 Add Binary 加减乘除, 进位
这里用到的思想就是两个指针分别从两个字符串最后往前遍历，然后有一个sum 去加在ASCII 码的每个数字， 然后还有一个carry 去代表进位，结果中的每一位是sum % 2 得到，然后每次sum 在未进过任何加数之前是要等于carry，因为从上次计算得到的进位要算在里面，每次carry 更新carry = sum / 2。 最后出来时判断carry 是否还有剩下，如果有就将carry append 到结果中，最后返回reverse的结果

**258 Add Digits**
我们可以先看下例子
267 = 2 + 6 + 7 = 15 > 10 
15 = 1 + 5 = 6 < 10 → 输出6
第二个例子
7114 = 7 + 1 + 1 + 4 = 14 > 10
14 = 1 + 4 = 5 < 10 → 5 输出5
我们发现每一个例子都是会有一个相似的地方，就是每次都是当每个digit相加最终返回一个单个digit的结果 就返回了，所以这里就是可以用递归的思想来做，每次算出新的和然后看它是否还在10的范围外，如果是就继续算digit和


 


## 位运算小结
**136 Single Number**

**389 Find the Difference**
利用 XOR 异或 的性质去更新更长的string 的最后一个character,最后返回
**268 Missing number**

**231 Power of two**
用n & (n - 1) == 0 判断是否是2的整数次幂

**191 Number of 1 bits**
这里要统计unsigned integer 的1的个数，也就是要用到 n & 1 来末尾取一，同时更新答案，然后返回res，每次更新数字往右移一位

**318 Maximum product of two strings**
一开始想我们遍历每一个单词，然后让每一个单词和下一个单词进行一个单词比较，这里比较的是两个单词是否是unique的，当每一个单词不是unique的那么就不更新答案， 如果是unique 的就更新答案
关键在如何判断这两个单词是unique的，一个最navie的solution就是将这两个单词其中一个每一个 character判断位置是否为-1，-1就是说明这个character不在另一个单词里，如果不为-1，那么这个character 在这个character里，就返回负数
We first intitialize result to 0. We then iterate from
0 to 31 (an integer has 32 bits). In each iteration:
We first shift result to the left by 1 bit.
Then, if the last digit of input n is 1, we add 1 to result. To
find the last digit of n, we just do: (n & 1)
Example, if n=5 (101), n&1 = 101 & 001 = 001 = 1;
however, if n = 2 (10), n&1 = 10 & 01 = 00 = 0).
Finally, we update n by shifting it to the right by 1 (n >>= 1). This is because the last digit is already taken care of, so we need to drop it by shifting n to the right by 1.
At the end of the iteration, we return result.
Example, if input n = 13 (represented in binary as
0000_0000_0000_0000_0000_0000_0000_1101, the "_" is for readability),
calling reverseBits(13) should return:
1011_0000_0000_0000_0000_0000_0000_0000
Here is how our algorithm would work for input n = 13:
Initially, result = 0 = 0000_0000_0000_0000_0000_0000_0000_0000,
n = 13 = 0000_0000_0000_0000_0000_0000_0000_1101
Starting for loop:
i = 0:
result = result << 1 = 0000_0000_0000_0000_0000_0000_0000_0000.
n&1 = 0000_0000_0000_0000_0000_0000_0000_1101
& 0000_0000_0000_0000_0000_0000_0000_0001
= 0000_0000_0000_0000_0000_0000_0000_0001 = 1
therefore result = result + 1 =
0000_0000_0000_0000_0000_0000_0000_0000
+ 0000_0000_0000_0000_0000_0000_0000_0001
= 0000_0000_0000_0000_0000_0000_0000_0001 = 1
Next, we right shift n by 1 (n >>= 1) (i.e. we drop the least significant bit) to get:
n = 0000_0000_0000_0000_0000_0000_0000_0110.
We then go to the next iteration.
i = 1:
result = result << 1 = 0000_0000_0000_0000_0000_0000_0000_0010;
n&1 = 0000_0000_0000_0000_0000_0000_0000_0110 &
0000_0000_0000_0000_0000_0000_0000_0001
= 0000_0000_0000_0000_0000_0000_0000_0000 = 0;
therefore we don't increment result.
We right shift n by 1 (n >>= 1) to get:
n = 0000_0000_0000_0000_0000_0000_0000_0011.
We then go to the next iteration.
i = 2:
result = result << 1 = 0000_0000_0000_0000_0000_0000_0000_0100.
n&1 = 0000_0000_0000_0000_0000_0000_0000_0011 &
0000_0000_0000_0000_0000_0000_0000_0001 =
0000_0000_0000_0000_0000_0000_0000_0001 = 1
therefore result = result + 1 =
0000_0000_0000_0000_0000_0000_0000_0100 +
0000_0000_0000_0000_0000_0000_0000_0001 =
result = 0000_0000_0000_0000_0000_0000_0000_0101
We right shift n by 1 to get:
n = 0000_0000_0000_0000_0000_0000_0000_0001.
We then go to the next iteration.
i = 3:
result = result << 1 = 0000_0000_0000_0000_0000_0000_0000_1010.
n&1 = 0000_0000_0000_0000_0000_0000_0000_0001 &
0000_0000_0000_0000_0000_0000_0000_0001 =
0000_0000_0000_0000_0000_0000_0000_0001 = 1
therefore result = result + 1 =
= 0000_0000_0000_0000_0000_0000_0000_1011
We right shift n by 1 to get:
n = 0000_0000_0000_0000_0000_0000_0000_0000 = 0.
Now, from here to the end of the iteration, n is 0, so (n&1)
will always be 0 and and n >>=1 will not change n. The only change
will be for result <<=1, i.e. shifting result to the left by 1 digit.
Since there we have i=4 to i = 31 iterations left, this will result
in padding 28 0's to the right of result. i.e at the end, we get
result = 1011_0000_0000_0000_0000_0000_0000_0000
This is exactly what we expected to get
Divide two Integers
ToDO
Sqrt(x)
用二分的方法

## 双指针 


Window模板总结
题目概览
滑动窗口这类问题一般需要用到双指针来进行求解，另外一类比较特殊则是需要用到特定的数据结构，像是 sorted_map。后者有特定的题型，后面会列出来，但是，对于前者，题形变化非常的大，一般都是基于字符串和数组的，所以我们重点总结这种基于双指针的滑动窗口问题。

题目问法大致有这几种：

给两个字符串，一长一短，问其中短的是否在长的中满足一定的条件存在，例如：
求长的的最短子串，该子串必须涵盖短的的所有字符
短的的 anagram 在长的中出现的所有位置
...
给一个字符串或者数组，问这个字符串的子串或者子数组是否满足一定的条件，例如：
含有少于 k 个不同字符的最长子串
所有字符都只出现一次的最长子串
...
除此之外，还有一些其他的问法，但是不变的是，这类题目脱离不开主串（主数组）和子串（子数组）的关系，要求的时间复杂度往往是 O(n)，空间复杂度往往是常数级的。之所以是滑动窗口，是因为，遍历的时候，两个指针一前一后夹着的子串（子数组）类似一个窗口，这个窗口大小和范围会随着前后指针的移动发生变化。


解题思路
根据前面的描述，滑动窗口就是这类题目的重点，换句话说，窗口的移动就是重点。我们要控制前后指针的移动来控制窗口，这样的移动是有条件的，也就是要想清楚在什么情况下移动，在什么情况下保持不变，我在这里讲讲我的想法，我的思路是保证右指针每次往前移动一格，每次移动都会有新的一个元素进入窗口，这时条件可能就会发生改变，然后根据当前条件来决定左指针是否移动，以及移动多少格。我写来一个模版在这里，可以参考：


1.模板的意思是用两个指针，left或者right，它们之间代表了一个window
2. 移动右指针去找一个合适的窗口
3. 当一个合适的窗口找到之后，移动left去找到更小的窗口
要检查一个window是否合适，我们用一个map去存(char, count) 对每一个

具体的代码模板如下

```
public int slidingWindowTemplate(String[] A, ...) {
 // 输入参数有效性
 if (...) {
  ...
 }
 
 // 申请一个散列，用于记录窗口具体元素的个数情况
 // 这里用 数组的形式呈现，可以考虑其他数据结构
 int[] hash = new int[...];
 
 // left 表示指针
 // count 记录当前的条件，具体按题目要求
 // res 存结果
 
 int l = 0, count = ..., result = ...;
 for (int r = 0; r < A.length; r++) {
  // 更新元素在散列中的数量
  hash[A[r]--];
  
  // 根据窗口的变更结果来改变条件值
  if (hash[A[r]] == (...)) {
   count++;
  }
  
  // 如果当前条件不满足，移动左指针直到条件满足为止
  while (count > K || ...) {
  
   ...
   if (...) {
    count--;
   }
   hash[A[l]]++;
   l++;
  }
  
  //  更新结果
  res = ...
  
 }
 
 return res;
}

```


具体的题目例子



**239 Sliding Window Maximum**


最原始的双指针，O(Nk - 1)
PriorityQueue 放入进去， O(NlogK - 1)
deque 的写法












**992. Subarrays with K Different Integers**

**76 Minimum window substring**
模板题

**209 Minimum Size Subarray sum**
according to the template, we can set up two pointers, l and r, two are moving in the same directions, and when the right is moving along the array, the sum must add the number where r points to (prefix sum), and once the sum is greater than the s, that means after the r, the sum would always be greater than the s, assuming all the number is positive in the array.
So the algorithm will be marked down as
1 move the right pointer and add the number, where the right pointer is pointing, to the sum.
2 once the number is greater than the s, meaning, the window is valid, then we need to update our res with the window between right pointer and left pointer. To shrink the window down until we can find a minimized size window, always decrease the number where the left pointer is pointing to

**340 Longest Substring with At Most K Distinct Characters**
we still need to declare two pointers, left and right,
move the right pointer and see the window has the at most K distinct characters
then we have a maxLen defined initially as Integer.MIN_VALUE, and counter at 0,
when we see the character that right pointer is pointing to is not in the map, then we can increase our counter by 1 and increase the occurrence of the character in our map, and increase the right pointer by 1.
Then we would see if the counter has been greater than k, if yes then we can find the character where the start pointer is pointing to, then we can decrease the counter if we see the occurrence of the character,
Finally we can update our maximum substring length 

159 Longest Substring with at most two characters
Same template from minimum window is used, but right now we only care about 2 instead k substring. 

3 Longest Substring without repeating characters
1 same template from minimum window, a window is valid when there is no repeating characters inside so if there are any repeating characters by the right pointer in the map we increase our counter by 1 and decrease that character occurrence in the map, then we increase the end
2 then if the counter is bigger than 0, we need to find if the character which the start pointer is pointing to shows more than once, than we can decrease the counter by 1 and decrease the occurrence if that character by 1 in the map and increase the start pointer
3 we update the window length
438. Find All Anagrams in a String
这里我们的基本idea是

step1 用一个hashmap 去记录每一个p的character 然后他们对应的初始数字是-1， 
从左往右遍历整个s时，
step 2 先往map里面加出现的次数
step 3 然后如果当 map.get(s[i]) == 0时， 我们就要remove这个重复的character ，

step 4 如果i 已经超过p的长度时，我们需要重复step 1和step3，注意这里的i变成了
i - p.length() 因为
567. Permutation in String
992. Subarrays with K Different Integers
1055. Shortest Way to Form String
目的是找到不在source 里存在的而在target存在的指针
遍历target ，每次记录当前的target 指针 t为base， 然后在target 没到结尾之前，比较source 的当前character 是否与target 相同，相同的话，t ++
如果for 循环完发现t没动的话，就说明source 有不存在的character ，返回-1
source = "abc", target = "abcbc"
a b c
      ^ 
      s
a b c b c     
            ^
            t
      b
ans =2 exit since t = target.length()


**pow(x, n)**
这个可以用brute force 来做，就乘以多少次，但是这里就是过不了一个test case
所以用快速幂算法，

 这里可以用recursion 来做,注意这里如果power是负数，就需要设p为负数，然后将x设为倒数
整体时间复杂度为O(logn)
 
 **3sum closet**
 这里用的双指针, 在做任何动作之前，我们先要将数组排序，排序完之后我们有了一个有序的数组，然后从0到i-2遍历一遍数组，这时候我们有另一个指针 `l = i + 1` 和 `r = nums.length - 1` 这里的l 和 r 的作用是来找到3个数之和也就是 `sum = nums[i] + nums[l] + nums[r]` 如果 `sum > target` 那么我们的`r`需要减少，反之，`l` 需要增加，不断通过扫数组去找到最小的，也就是最closet的`sum`
 
 **3Sum Smaller**
 这里可以将`2Sum smaller`的思想imply 到这里，我们可以有一个`i` 指针从0到`nums.length - 2` 一直遍历，每次`left = i + 1, r = nums.length - 1` 可以用这三个指针表示当前的和，然后用这个当前的和去和`target` 比较，如果是小于`target`，那么这里的答案可以更新为`ans += r - l` 因为这里是一个窗口来走，同时可以更新`left` 指针， 反之更新`right` 指针
 
 **Move Zeros**
 这里我们用一个 `index` 来记录所有非零的元素，然后我的所有非零的元素就是放到`nums[index]` 的位置上来，最后我们从 `index` 到 `nums.length - 1` 全是0
 
 **reverse vowels**
 这里我们可以用 左右指针往中间走， 当左指针没有包含vowels， 左指针往右走，右指针也是这样， 往左走，这里直到找到了两个vowel，然后将这两个交换
 
二维数组4种交换方式 总结


 
Backtrack (通用解法) 基础 总结
对于字符串的Backtrack (通用解法) 的套路总结
每次看到字符串的Backtrack 时，首先要想到的是这个题目需要我们做什么，是需要我们排列，组合，还是其他？然后我们得去思考什么参数是需要的，而在
这里一般我们需要一个Temporal String或者StringBuilder 来改变我们递归时放入的String的格式，
然后这里还需要用一个List来存储答案
一般还需要一个index来记录当前character 所在的位置
Backtrack (通用解法) 对于For 循环里面递归的理解
递归这东西，理解比实现要难一点。
理解递归的过程，就是尝试给函数下定义的过程。简单的递归一目了然，麻烦一点的我一般是靠读代码先估计个函数的大概作用，然后去验证一下。
验证方法：当你能明确地定义这个函数的作用后，将其逻辑走一遍（不要递归地走，遇到被递归调用的函数则把它的定义代入即可），能保证逻辑的正确，就行了。
这时内心要鄙视那些类似“通过画执行图来理解函数”的方法，以免产生“妄图完整想象整个执行过程”这种毫无意义的偏执。
举个最简单的求斐波那契数的例子：
function f(n){
if n <= 1: return 1;
return f(n-1)+f(n-2);
}
看到这代码，我尝试下个定义：这函数是求第n个斐波那契数的。然后走一遍他的逻辑：将上一个和上上一个数的和返回，没错。这就可以了。别去管它具体怎么调用，调用了几层之类的。要想像自己是个领导，别去管那些啰里八嗦的细节。
最后回到题目，先看for循环，是将第level个及其之后的数字依次与第level个数交换，然后类似的去搞第level+1层。这一看就是在搞全排列嘛。那这个函数的作用估计就是“把数组中第level个至最后一个元素搞一下全排列然后塞到一个容器里”。
来走遍逻辑验算一下：进入函数体，我打算搞第level个元素开始的全排列了，进入for循环，先把第level个数保持不变，调用自己把level+1开始的元素搞个全排列；再把第level+1个数放到level位置，调用自己把level+1开始的元素搞个全排列......哎！没错呀，这确实是在搞第level个数开始的全排列啊，我的理解是正确的！
最后总结一下，理解递归一般的方法就是“定义函数”+“验证定义”。
定义的方法很多，可以靠丰富的经验；也可以靠瞎猜；简单点的一眼就读懂了甚至不需要验证。
验证就是把定义代入进去看逻辑能否走通。
PS：代码多种多样，我也不知道是否有通用的去理解有for循环的递归的方法。就比如这段代码还有个简单的理解方法，就是从最深一层开始倒推着来理解，不展开了。但换另外个同样有for循环的递归可能又有其他理解方法了。
我这里其实也没提供通用的理解方法，主要的只是给了一个“验证你的理解”的方法。因为代码写多了后，理解递归其实不是个问题，不用担心什么，就像婴儿学说话一样，看起来很难，其实不用担心以后。而“验证”这一步，我至今仍然在用，因此写出来分享一下。

 
Backtrack 模板

如何判断回溯法里怎么避开重复的值？
用一个boolean 数组去判断这个值已经访问过了，backtracking的时候把它重设回这个未访问过

什么时候加boolean 数组什么时候不加？

需要看是和否visited 过就加

什么时候用i + 1, 什么时候用i？
就是当数字/Character不能重复使用的时候，就用i + 1, 如果数字能重复使用的时候就是i

permutation 跟combination 有什么区别？
combination 是不能再选当前的数字， 只能选择后面的数字，permutation 可以重复选择当前的的数字，但是要标记当前的数字有无用过
怎么样去定长度？
  如果要定长度的话，就要去判断i 是否是等于这个candidates的长度，这样保证每次出来的答案长度都是一样的， 那么这里不需要定长度，所以就不需要这一层判断


Permutation 和Combination 模板差别

LeetCode 46. Permutations
题目描述
给定一个含有不同数字的集合，返回所有可能的全排列。
比如，
[1,2,3] 具有如下排列：
分析
全排列首先考虑深度优先搜索，每个深度的遍历都从 0到nums.length - 1
每次循环开始需要判断当前值是否已经使用过，即 if (!tempList.contains(nums[i]))

创建一个List<Integer> tempList存放临时数据
当tempList.size() == nums.length时
res.add(new ArrayList<Integer>(tempList))将tempList存入结果
此处不能直接add.(tempList)，否则改变tempList也会导致结果的改变

特别注意每次递归结束前都需要将刚加入的值从tempList中去掉
————————————————
版权声明：本文为CSDN博主「笑乾」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/genius_wenbo/article/details/79672102








LeetCode 47. PermutationsII
题目描述
给定一个可能包含重复数字的集合，返回所有可能的不同全排列。
例如,
[1,1,2] 有以下不同全排列：
分析一
此题跟上一题的区别在于给定数组可能包含重复数字，所以首先将数组排序 Arrays.sort(nums)

由于数组可能包含重复数字，所以不能用if (!tempList.contains(nums[i]))判断当前值是否使用过
需要创建一个数组boolean[] used = new boolean[nums.length]记录每个位置的值是否使用过（上题也可以使用这个方法）

但还存在一个问题，由于数组包含重复数字，tempList可能一模一样
即可能出现两个[1,1,2]
两个1分别来自两个位置

所以当每一层遍历到第二次或更多次，即i > 0时
需要判断前一个值（num[i - 1]）是否被使用过，如果没使用过，继而判断当前值与前一个值是否相等
Decision tree 如下

————————————————






2. 子集
LeetCode 78. Subsets
题目描述
给定一组不同的整数 nums，返回所有可能的子集（幂集）。
注意事项：该解决方案集不能包含重复的子集。
例如，如果 nums = [1,2,3]，结果为以下答案：

[
 [3],
 [1],
 [2],
 [1,2,3],
 [1,3],
 [2,3],
 [1,2],
 []
]
分析
子集与全排列有两大区别
输出的List长度不等
所谓子集，就是要求该集合所包含的所有集合
所以每次循环都要将tepmList加入res
而不是等tempList.size() == nums.length
List中元素不能重复
在全排列中，结果中每个List包含的元素都相同，只是顺序不一样
如[1,2,3]和[3,2,1]
子集则不同，每个List中的元素都不相同，所以循环不能再从0开始
需要重新定义一个变量start作为dfs()的输入参数

每次递归将start设为i + 1 即不会遍历之前访问过的元素
————————————————






LeetCode 90. SubsetsII
题目描述
给定一个可能包含重复整数的列表，返回所有可能的子集（幂集）。
注意事项：解决方案集不能包含重复的子集。
例如，如果 nums = [1,2,2]，答案为：
Sublist Mind Map
Choose/explore (without s)
Choose/explore (with s)
Unchoose
Remember to remove the elements in the chosen vector
 
You don’t always have a loop
[
 [2],
 [1],
 [1,2,2],
 [2,2],
 [1,2],
 []
]


3. 组合总数
LeetCode 39. Combination Sum
为什么我们需要index？
因为这里分支答案不是一个个固定长度的List，换句话来说，这些小分支是不同长度的，所以我们要用index 来越过我们已经visit 过的数字
为什么我们recursion时不是i + 1 而是i?
因为我们想重复利用数字
怎么判断我们已经到达了终点？ 每当target等于0时
题目描述
给一组候选数字（C）（没有重复）并给一个目标数字（T），找出 C 中所有唯一的组合使得它们的和为 T。
可以从 C 无限次数中选择相同的数字。

说明：

所有数字（包括目标）都是正整数。
解集合中没有相同组合。
例如，给一个候选集合 [2, 3, 6, 7] 并且目标为 7，
一个解的集合为：
[
 [7],
 [2, 2, 3]
]
分析
本题与子集的区别在于数组中的数字可以无限使用
因此递归中start参数传入的值应为i而不是i + 1
本题还需定义一个remain参数来记录动态目标值

关键点1： 要防止remain 小于0，因为在每一次的递归调用中，这里remain是会变成小于0的数，如果是小于0，这里没有限制，所以很容易的造成stack overflow
 关键点2：index 是i 为什么？因为所有数字可以重复选，然后不需要跳过这个用过的数字，所以用i 而不是i + 1, 为什么 i 不是index？ 因为这里index 代表的是每层的深度，i代表的是这个索引
 permutation 跟combination 有什么区别？
combination 是不能再选当前的数字， 只能选择后面的数字，permutation 可以重复选择当前的的数字，但是要标记当前的数字有无用过
  如果要定长度的话，就要去判断i 是否是等于这个candidates的长度，这样保证每次出来的答案长度都是一样的， 那么这里不需要定长度，所以就不需要这一层判断
还有就是需要i = index 因为这里不能选前面的数字了，但可以选择当前的数字

LeetCode 40. Combination SumII
题目描述
给定候选号码数组 (C) 和目标总和数 (T)，找出 C 中候选号码总和为 T 的所有唯一组合。

C 中的每个数字只能在组合中使用一次。

注意:

所有数字（包括目标）都是正整数。
解决方案集不能包含重复的组合。
例如，给定候选集合 [10, 1, 2, 7, 6, 1, 5] 和目标总和数 8，
可行的集合是：
[
 [1, 7],
 [1, 2, 5],
 [2, 6],
 [1, 1, 6]
]
分析
首先排序！！！
判断前后值！！！


47 Permutation II
有重复元素出现在numbers 里所以要查重，而且一般要事先对numbers进行排序处理
这里需不需要inde x 来跳过已经遇到的数字? -- 不需要因为这里是排列，每个数字可以重复运用

31 Next Permutation
Brute Force的 方法就是将每个permutation 输出来，然后找到比当前输入大的permutation就break, 这种方法是O(N!)的时间，所以并不推荐用这种方法
example 1 2 3 -> 1 3 2
        3 2 1 -> 1 2 3 (返回比较小的)
这个题思路是 需要找到一个当前第一个nums[i] < nums[i + 1] 的数字， 从后往前推，找到这个firstSmall 的值， 也就是i，break 出来
如果找不到，也就是firstSmall是-1的， 因为这里整个数组是一个descending 的数列,那么需要翻转一下,
reverse(nums, 0, nums.length - 1)

然后再for 循环找这个firstSmall大的第一个数，同样也是从后往前，这个数是firstLarge
如果找到了这个firstLarge (nums[i] > nums[firstSmall]). break 出来
然后对firstLarge 和 firstSmall 进行一次交换
然后对firstSmall + 1 和 nums.length - 1进行一次翻转




60 Permutation Sequence
这里可以用permutation算出所有的解，然后再算这个第k个permutation是多少，但是这种方法很耗时(O(N!))，并不是一个很省时间的方法, Lintcode 过的了，但Leetcode 卡在最后 n = 9, k - 13596, 这里可能需要一个更加聪明的办法，并不需要递归，
res = 1 2 3 4
这里新建一个 fact的数组，fact[i] 表示i的全排列是多少个
fact = 1 1 2 6
还要将k 更新成 k = k - 1, 因为这里k是从1 开始，但是我们这里fact是从0到n - 1
然后一个for循环
首先先算最高位
例子比如
要算k  = 18 的排列
fact 一共有4位 {6 2 1 1} 
k = 18 - 1 = 17
for 循环从后往前传
首先看最高位的全排列，然后用 k / fact[i - 1] 算剩下的全排列
index = 17 / 6 = 2 k = 17 % 6 = 5
4 3 2 1

216 Combination Sum III
这里我们就是还是根据模板，但是这里要注意就是当我们看到remain为0的时候不能立刻将cur 加入到res里，而是要进一步判断这个k有没有用光，如果用光了才会加入到res 里，
还有一个坑就是题目规定limit 是到9的，而不是到目标本身，所以有可能产生空解，

 
254 Factor Combinations
又是一道Combination 的题目，用backtracking的模板来做，这里要注意几个坑，第一个坑是到底需要不需要index 作为参数之一？ 这里是需要的，为什么？因为这里的每个小的sublist的长度各不相同，所以这是需要的
第二个坑是怎么判断什么时候加sublist到res 的list中去？这里需要知道一点，当你的factor就是这个数本身的时候，是不能当成解之一的，所以这时候怎么处理？这时候就是要去base case判断是否这个sublist的元素个数是超过1的，如果是，那么就意味着这个答案合格，
 
51 N-Queens
这里需要建立三个rules，就会是同一个row，同一个col，同一个对角线都不能有Queens，然后就是这个是需要灵魂拷问一波: 需不需要这个index？是需要的，因为index还是在代表row的index，然后for循环遍历column 的index，
也需要三个HashSet, 这里就是每次三个HashSet同时没有包含这个这个位置，说明，这个位置可以摆Queen 棋。然后这里的back track 要记得把棋子重设成空
Base case里面需要有一个新的sublist去遍历每一个board的行并且加入到character里面去，然后再将整个sublist放到res里面当成是一个解，因为这里一个解就是一个棋盘格局
 
52 N-Queens II
这里比51要简单就是每次要把unique的solution的个数输出出来，那么就是不需要去建棋盘。然后去更新答案,因为你已经将准确的答案更新了
 
17 Letter Combinations of a Phone Number
这里需要注意的坑是index不是用来作为for循环的起始位置，这里指的是我可以通过index来定位这个digit的对应letters 是什么，然后通过for 循环去遍历每一个character、
 
93 Restore IP address
这里是回溯法的实现题，想法，就是我们每次选择分割的IP address 是不断的放“.” 到input里面去的，所以就是将所有的可能性都尝试一遍，但这样太费时间，所以就应该想到一个比较好的想法就是这个segment ip address是可以预先判断是否是合格的，就比如说它是否是在[0, 255] 这个闭区间里面，
这道题也是一个用DFS找所有的可能性的问题。这样一串数字：25525511135
我们可以对他进行切分，但是根据IP的性质，有些切分就是明显不可能的：
比如011.11.11.11, 这个的问题是以0开头了，不合法，直接跳过。
比如257.11.11.11, 257大于IP里的最大数 255了，不合法，直接跳过。
然后我们把这一层切分后剩下的字符串传到下一层，继续去找。
直到最后我们得到4个section为止，把这个结果存到我们的result list。
有个坑就是还要判断说这个parameters需不需要index？这里是需要的，因为这里index代表的是我们segment的个数，也就是个数如果到了4个的话，说明有3个点已经把这长串String给分割了，比方说
a.b.c.d → 这个时候s应该是要是空的，因为我们每次去割的时候如果segment不合格那么最后s肯定不为空，那么就不更新答案，
还有就是这个recursion怎么去call？
想一下当我们分割到第三层 (index < 3) 的时候，我们用了多少个dot？在我们只分到三个section 之前,我们一直需要加 “.”，但是当index >= 3 的时候，就应该不在需要加点了，因为我们点已经用完了
处理String 的Backtracking的方法 → 还是得去理解题意
 
282 Expression add Operator
给定了一个String 的digits和一个target value， 要求出有多少种式子来得到这个target，
这里我们要想个比较naive的方法，就是我求出所有的符合条件的式子，比方说我有
“12”我想求出所有1和2 能够组成的式子
root
/\
          1        12 
         /|\
- *
      |    |    |
   2 2  2 
得到这么多式子但是很难找到规律，所以我们再看下怎么去求得这个
当我们只有+ 或者-的时候，我们的结果应该是将val + curNum/ val - curNum 对应的diff也就是+curNum/-curNum
以下这个表就是我们需要维护的
cur, prev, val

加法 就是val + curNum, 新的prev就是curNum
减法 就是val - curNum，新的prev就是-curNum
但是乘法比较麻烦，因为乘法是要先做，所以这里就要用prev 去反推乘法的值，给个例子
1 + 2 * 3 diff (prev) = 2 curNum (n) = 3 val (cur) = 3 
第二个例子
2 + 3 * 4 
(5 - 3)  + 3 * 4

所以乘法的就是 val - diff + diff * curNum 
新的prev就是 diff * curNum
每次我们的for循环怎么遍历？
就是我们每次loop的是从index + 1 开始， 然后到最后一位digit结束
坑二： 就是这个坑是要判断我们的当前subdigits是否是一个合法的值，不合法的包括了“01” “001” “012” 
坑三：我们的diff， curNum， 以及val都是Long型，因为这里我们要防止overflow
遇到String 的digits或者是一串的字符时候，怎么用backtracking去找每一个可能的单词?
应该是在for循环里去substring去切一块下来然后进行下一步的操作
params: res, path, num, target, pos, (long) val, (long) pre
val 表示当前的结果。pre表示之前的结果
num 必须遍历到结尾，并且pos 必须等于nums.length, 并且val 要等于target
for 循环从pos 开始，要对0进行处理，如果nums[pos] == ‘0’ 并且 i != pos → break
cur = Long.parseLong(nums.substring(pos, i + 1));
如果pos 是0的话，需要进行 
helper(res, path + cur, num, target, i + 1, cur, cur)
如果不是0的话， 进行
helper(res, path + “+” + cur, num, target, i + 1, val + cur, cur)
helper(res, path + “-” + cur, num, target, i + 1, val - cur, cur)
helper(res, path + “*” + cur, num, target, i + 1, val - pre + pre * cur, pre * cur)
Test case
“123” target = 6
pos = 0 cur = 1 path = “1”
    pos = 1 cur = 12                   val              pre
       pos = 1 cur = 2 path = “1 + 2”       3              2
                 path = “1 - 2”      -1             -2
                     path = “1 * 2”       1 - 1 + 1 * 2   2
                                  val           pre
pos = 2 cur = 3 path = “1 + 2 + 3”       3 + 3     3
                      path = “1 - 2 - 3”          -1 - 3     -3
                path = “1 * 2 * 3”        2 - 2        2 * 3



140 word break II
这题意思挺好理解的，但是这里呢比较不好处理的是如何去剪枝，因为首先如果暴力回溯的话，这里会TLE，因为dictionary 会有很多单词，造成这个algorithm 不能完成
所以要采用记忆化搜索，也就是要一个HashMap 叫memo，存这个单词和对应的句子
最关键的是要用这个memo去剪枝，也就是当我们看到这个s在我们的memo时

然后遍历每一个词典中的单词，每个单词先判断这个单词是否是在string 开头，如果在，那么通过recursion去找到这个string除了当前单词的对应的所有的单词list，然后分情况讨论
如果这个list里的单词是空的，那么就往res里面加入开头的单词
如果不为空，那么就是word + “ ” + item
坑： 一定要在最后往memo里面加上s 和对应已经更新的res

返回 res

351 All Android Unlocking Pattern
回溯加count， 这里是可以用symmetric的思想， 1 3 5 7 是对称的，2 4 6 8 也是对称的，然后就是我们要去判断这个pattern 是否是合法的
怎么判断合法？
就是我们会有个visited的boolean数组，还有一个棋盘表示手机的九宫格，每个九宫格实际上对应的是我们需要跨越的数字，比方说 1 → 2 → 3 我们必须去visit这个2才算合法，跨越了这个2是非法的pattern
坑二：怎么去更新这个移动的步数？
这个移动步数有个范围，要比m大于或者等于，然后最大不能超过n，所以这里我们可以定义个moves参数和count参数每当moves 大于或等于m，我们加count，然后返回count当moves > n
时
坑三： 这是一道回溯题目，怎么去做这个回溯？
这个回溯指的是我们每次在看这个num有没有被visited过，所以当for循环结束的时候，我们需要重设这个num作为unvisited
10 Regular Expression Matching

Brute-force solution recursion
        algo:
        base case
            1. pattern == "" --> text.isEmpty() 
        Recursive cases
            2. first_match: pattern[1] == "." || pattern[0] == text[0] && !text.equals("")
            3. second_match:
                a. if pattern[1] == '*' && pattern >= 2
                    -- Recurse call for pattern[2:end] with text || first match &&  text[1:end] with pattern
                b. else
                    -- first_match && Recurse call for text[1:end] and pattern[1:end]

802/37(Leetcode) Sudoku Problem

Solution: brute-force, nested for loops, each block would have 9 possible 
Numbers to pick, each cell has to check if previous has been selected

Solution 2 DFS

Rules to obey when come to the non-zero values in the matrix, we need to mark
The col and row indices, and the subgrid index as visited;

For zero value, I need to know it can’t break the rule, as the col, row, and subgrid would 
Not contain the number that I want to place

Make sure the placement would not break the board

Finally we fill the whole board

Recursion method 
checking method 
89 Gray Code
    0 -- list.size() == 2^0 = 1
        1 -- list.size() == 2^1 = 2
        2 -- list.size() == 2^2 = 4
        ...
        
        Backtracking
        
        Base Case
        n -- list.size() == 2^n
        
        recursive case
        base = 1 << (i - 1) 1 <= i <= n
        
        last = list.get(list.size() - 1)
        
        curNum = last << base 1 <= i <= n
        
        cur.add(curNum)
        
        Here needs to check the duplicates
        
        Time O(n^2)
        

401. Binary Watch
这里我们需要看下例子
比如说看这个例子，

这里我看到hours 的1 和2 都亮了，然后分钟是1 8 和16亮了，那么这怎么去计算呢？
实际上根据时间的定义 “hour : miuntes ” 这里minutes 是有点tricky 就是当minutes 少于10 分钟的话，“0minute” -- 这里 
465. Optimal Account Balancing
这道题输入是 {出资人，收款人， 钱数}
那么我们可以按照这个关系构建一个HashMap <人，总额> 如果是出资人，他的总额减去钱数，如果是收款人，他的总额是加上钱数，那么这个最后的这个map的value 可以表示每一个人对应的balance， 
backtracking 函数的参数思考：
我们首先要有这个balance list， 然后我们需要一个pos 指针去表示当前的人的位置，还有我们需要一个count 来记录交易的次数，最后我们还要有一个类变量去记录总的最小交易记录
我们这个时候要去想怎么去将这些balance 搞平？首先我们需要知道什么时候才要balance？当我们 这个 balance.get(i) (i > pos) 和 balance.get(pos) 是不同符号的时候我们才去balance
怎么balance? balance.set(i, balance.get(i) + balance.get(pos))
recursion call -- helper(balance, pos + 1, count + 1)
backtracking -- balance.set(i, balance.get(i) - balance.get(pos) )
坑1：注意balance.get(pos) == 0 为0的话我需要pos++
因为这里balance 已经平了
Flip Game II 
题目的意思是指有两个人在flip 一个字符串只包含了 ‘+’ 和 ‘-’，每次flip 的字符数是2，就是每次flip “++” 变成 “--”  或者 “--” 变成 “++”，然后这个问题就变成了： 如果我们想让第一个人win， 那么我们需要去找到第二个人的flip完的字符串，怎么找到呢？
 轮到第二个人的flip完的 字符串是什么? 
假设我们第一个人来第一轮，依旧是将头两个字符换了，那么这里这个字符就变成num[0, i] , 然后轮到第二个人来flip， 他/她 只能flip i  之后的两个字符，也就是 说这里我们要把 ‘++’ 变成 ‘--’  然后接着接上 之后的substring, 也就是 num[i + 2]
那么整个第二个人flip 完的字符串公式就是
num[0, i] + “--”  + num[i + 2]
如果我们发现这个字符串不能赢，那么就是说明对手是不能赢的，这里我们就可以返回true， 说明第一个人可以赢了，否则在循环之后我们说这人不能赢得

为了减去重复的计算，我们可以设一个HashMap 来存储immediate 的results， 这里就是一个记忆化搜索，每次的 return ans 之前我们都可以将当前的num 作为key,  true/false 作为value 存入HashMap里面去；这里会是得时间变得很快
O (n^2) ?



Additive Number
主要问题是第一个数 + 第二个数的和是否就是第三个数的值？
那么我们可以遍历第一个数和第二个数，第一个数是 从1 到len - 2, 第二个数是从第一个数+1到len - 1
那么我们是从递归的条件是什么？
我们递归的条件是找到符合第一个数和第二个数，和这里的String 之后第二个数之后 的字符串，当base case就是 s.length() == 0; 就说明这里全部验证过了，所以返回true

recursive case 是将这个第一个数和第二个数的和，那么我们就开始验证这个和是否就跟这个String 第二个数 的之后的字符串的开头相同，如果不相同，那么返回false，

之后递归的是什么？

a ------------------------------- b ------------------------------  a + b 
                solved                        about to solve
那么这里的下一个case 就是讲第二个数b，变成第一个数，然后将第二个数b 变成 a+b
这里的String 就变成了 s.substring((a + b).length(), s.length())
842. Split Array into Fibonacci Sequence
这里跟Additive Number 很像，因为这里也是斐波那契数列，主要思想是验证前两个数的和是否是和第三个数相同，然后这里的helper 的function 跟Additive Number 一样， 除了我们需要再验证第三个数是否超出整数的范围

每次找到第一个数和第二个数时候，这个第一个数和第二个数都是要检查是否整数的范围和要检查leading zero 


60 Permutation Sequence
我一开始为什么会错？
原因一： 一开始我以为就只需要一个StringBuilder 就可以了，但实际情况是一个StringBuilder 是要转换成“1...n” 的形式，然后由另一个StringBuilder 来进行对之前StringBuilder构建的String 中的数字进行一个选择，
    原因二：没有根据例子来找规律，我因为忽略掉了例子，使得自己错误的认为这个在同一个位置上的数字是可以被无限选择的，所以我没有进行对duplicate 的删除， 实际上这里需要进行重复的数字删除，used[i - 1] 为false 代表了回溯后， 之前的数字已经被选择了的状态，所以每当我们尝试在第二位选择之前选择过的数字时 （类似 11 22 33...） 我的duplicate checking可以发现这个数字已经选择过了，所以直接skip 掉
    原因三：对back tracking 中的for 循环的问题，原因是因为我不确定是否要将i 替换成index， 根本原因是对这两个东西代表的意思不理解： i 在这代表什么？i 并不是代表递归的层数，而是代表这个循环每次我要从curStr 拿走的character; 而 index 代表的是递归的层数，也就是说对这个函数定义的

Word Search
也是一道深度优先搜索的题目，是把上下左右相邻的结点看成有边联结，然后进行深度搜索就可以了，小细节是这里从每个点出发字符就可以重用，所以要重置一下访问结点。
这里流程如下：
先找到在board 有无这个word 的起始character， 作为起点，
然后就在这个起点开始往四个方向开始扩张，每次扩张时只要越界/当前charater 不是word 当前的character，就直接返回false， 这里回溯的结果是只要一个方向能候
够走到末尾就返回一个true
**Word search II**
这里的区别是现在我们要找出所有不重复的单词，而且输入不再只有一个，现在有一个单词想向量。这里就是可以有个笨方法就是说根据我上一题的说明这个单词是存在在这个矩阵的，那么就加这个单词到我的这个HashSet 中，最后放回这个Set转化成list的形式
**Palindrome Partitioning with no duplicates**
对字符进行一个预处理，也就是当出现奇数的次数超出1的话，就不可能出现
这里map 可以直接用一个256的一个数组来表示，统计完出现次数之后就可以将那些出现奇数次的character 放到一个String 里面用来表示中间的字符串是什么。
然后我们可以去做back tracking base case是当我们的cur String 达到了给的String 的长度那么，我们就可以返回这个答案，否则遍历map上所有的index，当这个index大于0的的时候，我们减去2 个occurance，因为我们要的cur String 是 “(char)i + temp + (char)i” 这里i 要加在前面和后面；所以在后面的回溯当中我们要把这个occurance 加回2
**Stickers to Spell Word**
这道题给了我们N个贴片，每个贴片上有一个小写字母的单词，给了我们一个目标单词target，让我们通过剪下贴片单词上的字母来拼出目标值，每个贴片都有无数个，问我们最少用几个贴片能拼出目标值target，如果不能拼出来的话，就返回-1
这里可以用DP的多重背包来做，或者是用记忆化搜索来做，这里我们先将记忆化搜索的方法
有点像 找到 minimum combination sum that sum up to a target with result 的题目
其中的题目的recursive case 表示
= 1 + (min combination of nums that sum up to (target - num) with result = {num}) // but we don't know num, so we guess each num


我们呢这里是要找到 minimum combination of words that cover target with res = {}
= 1 + (min combination of words that cover (target substract word) with result = {word}) // but we don't know word, so we guess each word
// target substract word: remove `ch` from target if `ch` in word
base case: target is empty

所以我们需要一个maps array 和一个 HashMap memo, maps 的每一行是每一个word的字符频数， 那么我们的memo 存的是每一个可能的target 和对应的最小值，这样我们就可以剪枝了
recursive case
当我们构建好了maps 和 target 对应的字符和频数map之后，我们就可以将这个maps的每一行跟target对应的targetMap 进行比较，如果出现在targetMap 中的字符重复了，我们往一个String Builder 里加上这个字符， 加上多少次? 对应字符在targetMap 和map的重复值之差. 然后我们通过recursion 去找到这个StringBuilder 对应的字符串的所对应的出现次数来更新这个最小贴片数，

**Stepping Numbers**
这一题的思路就是backtrack 找到所有可能的stepping number 数字个数， 这里的stepping number的相邻digits 都差1， 
第一根据例子我们可以发现这里的数字是不能重复选的，而且也不符合stepping number 的规范
第二我们发现这个里的思路跟  subsets II 很像， 也是找到不同的子集但不能重复选择数字，我们先要找到每个数的最后一位然后根据这个算出对应的多一个或者小一个

`long inc = cur * 10 + last + 1;`
`long dec = cur * 10 + last - 1;`
然后我们可以根据`last` 来判断这回我们是要将 `inc` 还是 `dec` 作为我们的 `cur`  如果`last` 等于0 的话，那么就用`inc`, 如果等于9的话，就用`dec`, 如果是在0 到9的区间内，那么就先`dec` 后`inc` 

## 2. 链表LinkedList题目小结
这里我们需要将一些基本的类型转换搞明白， 比方说让自定义链表 去转换成数组，或者是数组转化成链表


我们需要知道链表只要知道头节点，那么整个链表就知道了，也就是说我们在需要改变链表的某些值或者是顺序时，我们需要预先将表头存下来，这样我们就知道在遍历完之后，我们的头部在哪
比方说
[1 -> 2 -> 3 -> 4 -> 5]
我们要odd 和even 两个LinkedList
odd 
1 -> 3 -> 5
^
oddHead 这里需要预先存储，因为odd 本身需要改变
2 -> 4
^ 
evenHead 这里也需要存储


这里关于新的头结点有个非常重要的小常识： 每次我们翻转链表的时候，我们的prev 总是指向新的头结点

如何合并两个链表？ 
我们需要得到两个链表分别的头结点以方便我们以后可以合并两个链表， 这时候两个链表的头结点就是我们需要将dummy nodes放在前面，这样我们在合并两个链表的是候就可以将链表1的最后遍历的节点（list1）指链表2的头结点的下一个节点(list2Head.next), 因为这里链表2的头结点是dummy node


链表题目需要思考三步
头节点是否有改变？
循环的逻辑?
结束条件?

24. Swap ListNodes in pairs
这里思路是先思考需不需要改变头节点，这里是需要的，所以我们就需要去设一个新的头节点，
然后 思考 这里的循环逻辑是什么，逻辑就是我们需要先转dummy到2, 然后2的指向3的指针转到指向1， 最后1指向2的指针转到3

dummy 为l1, 1为l2, 然后就是开始进行遍历，




L1.next = l2.next
L1.next.next = l2
L2.next = l2.next.next

L1 = l1.next
L2 = l2.next

328 Odd Even LinkedList

这里需不需要dummy? 头节点并没有改变所以不一定需要
那么循环逻辑是什么？ 这里的循环逻辑就是我们可以设两个LinkedLists
一个LinkedList是奇数链表，一个LinkedList是偶数链表
然后每次跨过一个节点，
之后遍历下一个节点，
最后将odd链表连到even的头部就可以

206 Reverse LinkedList 
必须理解的代码

92 Reverse Linked List II (for a portion)

一个是走m - 1 步，把prev和cur调到1 和 2 的位置

141 LinkedList Cycle
判断链表环的方法
一个快指针， 一个慢指针， 二个都在头节点
然后当快指针不为空或者快指针的下一个不为空时，快指针走两步，满指针走一步


237 Delete node in a Linked List
删的永远知道删除节点的前一个节点是什么
交换后一个node 的值和当前的node 值，然后将后一个node给删除

 
83 Remove Duplicates from Sorted List、
扫一遍链表，然后如果遇到下一个值和自己的值相同就跳到隔一个值
如果遇到下一个值不和自己值相同的就正常迭代

82 Remove Duplicates from Sorted List II

我们判断cur.next 和cur.next.next 的价值一样就删
我们删除的时候一定要记录删除的值
while循环只要等于这个保存值，那么我们一定要这么删

234 Palindrome LinkedList
找middle 
Reverse 
一一对应
找中点的链表方法，slow 作为开始，fast作为第二个，然后每次fast走两步，slow走一步，最后fast到终点的时候，slow也走到了中点
25 Reverse List in K group
Solution 1 Using Stack Extra Space O(k)
        Use stack to push the element into the stack and poping them out would be
        in reverse order. Then we should be able to draw a digram to visualize the
        result
            k = 3
        
        D -> 1 -> 2 -> 3 -> 4 -> 5
        ^       ^
        cur  next
        
        
        | 3  |
        | 2  |    
        | 1  |
        
        cur.next -- stack.pop()
        if our stack size is not equal to k, then we should return the list directly since
        we hit the target
        
        then we need to connect the cur's next to the next, as shown in the example here
        D -> 3 -> 2 -> 1 -> 4 -> 5
                                 ^        ^
                              cur  next
                      
找到倒数第k个节点
通过设立两个指针p1 和p2, 其中p2 比p1 要多k个，那么当p2的nex为空时，p1 就是要返回的倒数第k个

203. Remove Linked List Elements
用一个prev 和cur 遍历数组，prev 初始指向dummy， cur初始指向head， 然后如果当cur 的值等于删除值，那么prev的指向cur的下一个节点。否则prev 就遍历到cur 的位置，cur 一直往后遍历
82. Remove Duplicates from Sorted List II
跟上一题203 的不同是，这里只要有值是重复的，那么就要完全删去， 
首先由快慢双指针，快指针指向头结点，慢指针指向dummy 节点，
那么这一步可以用while loop 来移动快指针， while loop之后，检查慢指针的下一个节点是否是等于快指针，如果不等，说明这里有duplicates， 需要将慢指针 的指向快指针的下一个节点，然后将快指针重设成慢指针的下一个节点。否则就没有duplicates, 直接移动双指针即可

这里需要注意的是我们在看快指针是否是duplicates的时候要判断快指针是否越界了，


148 Sort List
1. find middle point in the linked list
2. cut the list in half. - leftSection and rightSection
3. recursion call sortList to sort leftSection and rightSection
4. merge leftSection and rightSection

 /*
        Merge-sort LinkedList
        base case would be when node head is null return head, when head.next is null, also return head
        
        then we need to find two sublists like the left subsection 
        and the right subsection using slow and fast pointers
        
    */
    // merge sort all the lists

2 Add two Numbers
Inputs? Two linked lists Outputs? One Linked List
Sorted in reversed order

Example
(2 -> 4 -> 3) + (5 -> 6 -> 4)
      L1              l2 
New Linked List Head = New ListNode(l1.val + l2.val)
342 + 465 = 807
7 -> 0 -> 8
Solution: Simulate the adding process
So traverse the two linked lists at the same time. So there are several
The sum would be the l1.val + l2.val + carry but here we need to check if the l1 value or the l2 value are equal or not. Say
3 -> 1 -> 9 and 4 -> 2 -> 4 -> 6
L1 doesn’t have the 4th node, so set the 4th node in l1 would be nice

If the digit needs to carry, to find the carry for each loop, we would 
Use current sum divided by 10, then the last digit will be current sum mod 10
Also we need to take care of the left carry 
Say
900
   900
   1800
1 here is the carry got left

So the 1 needs to be a new node so the new linked list would point to it

Intersection of Two Linked Lists
Solution Two pointers
Given 
A -------------------null ptA
          |ptB 
          |
          |
          B
          ptA
Every time ptA hits the end of the linked list, make it to the beginning of the B linked list. Same with the list B, if list B hit the end of the linked list, connects it to the head of the list A. If they can meet, that means they are intersected, return a, The code will exit the while loop when curA and curB are both null. Time at most O(m + n), if they don’t have an intersection, then they will meet when both of they hit the tail nodes, which is null
Time O(N) Space O(1)
21. Merge Two Sorted Lists
分三步
找到头
处理剩下的节点
处理还有的节点

Merge K Sorted Lists 
Merge K sorted Lists, meaning that we need to take care a more generalized case than the Merge two sorted Lists. 
Inputs? K lists
Outputs? One merged Linked List
Solution 1. Brute-Force 
Traverse all the linked lists and collect values and put them into a array
Sort and Iterate over this array to get the proper value of nodes
Create a new sorted linked list and extend it with new nodes
O(NlogN) Collect O(N) Sort O(NlogN) Iterate new linked list O(N)

Solution 2 Compare K node one by one
O(kN)

Solution 3 Use PriorityQueue<> for the comparison in algorithm 2
This would cost O(Nlogk) since we use O(logK) to insert node into priority Queue every loop. But finding the node with the smallest value would only cost O(1) time
147 Insertion Sort List
Given 1 -> 3 -> 2 -> 4 - > null

dummy0 -> 1 -> 3 -> 2 -> 4 - > null
               |    |
              ptr toInsert
-- locate ptr = 3 by (ptr.val > ptr.next.val)
-- locate toInsert = ptr.next

dummy0 -> 1 -> 3 -> 2 -> 4 - > null
          |         |
   preInsert     toInsert
-- locate preInsert = 1 by preInsert.next.val > toInsert.val
-- insert toInsert between preInsert and preInsert.next

Linked List Cycle
First to ask: input? ListNode head output? Boolean value
Two pointers and check if they could meet
One is fast another is slow,
If they could → the cycle exists
It they couldn’t → the cycle not exists

Special case to consider,
 When there is guaranteed there is no cycle?
When there is not node
When there is only one node
Linked List Cycle II
这里是是一个龟兔赛跑的问题每次兔子走两步，乌龟走一步，然后当他们遇见的时候，cong head 到cycle entrance 的节点数是等于相遇点到cycle entrance 的节点数，
143. Reorder List
这个list 是一个头尾相连，往中间靠拢的一个reorder list
L0 -> Ln -> L1 -> Ln -1 
我们先想什么是否cycle 停止？
 遇到middle node时
然后我们根据middle node分隔；两个sublists， 然后将第二个sublists reverse 成Ln - > Ln-1
最后我们reorder 两个sublist



3. 二叉树的题目小结
298. Binary Tree Longest Consecutive Sequence
PreOrder
这一题的思路是用preOrder，然后每次在这preOrder的时候我们要去检查节点的左右子节点是否为空，
如果左右子节点不为空的话，分别递归求这个节点的左右子节点比节点差1，就说明我们可以更新当前的步数，
不停更新res，最后返回最大值

100 Is Same Tree

双Pre

思路就是判断二叉树不是一样的有两种情况
二个节点值 不等
其中一个树比另一个树早遍历完
判断二个二叉树是一样的有一种情况
两个树同时遍历完

就根据这三种情况，遍历左右子树，

101 Symmetric Tree

双Pre
1. 左边没有，右边有
2. 右边没有，左边有
    对称有两个情况
    1. 左右都有
    2. 左右都没有
    特殊情况
    没有或者只有一个节点，是对称的


思路就是有一个二叉树然后，从root的左边和右边出发，有一种情况二叉树是不对称的，
当二叉树的左右节点有一个点为空另一个点不为空的时候，不对称

当二叉树的左右节点都不为空的时候，对称，

但这里还有一个条件非常重要，就是判断左右节点值是否相同，如果相同才能叫对称，要不然就算有两个节点，值不相同也不能叫对称

还有就是我们递归调用的时候是对称的，也就是左子树和右子树，右子树和左子树

129 Sum root to leaf numbers

双P
这里的思路是我们可以设一个类变量，然后我们可以设一个temp 变量，作为计数器，然后用一个helper函数，每一次当我们到root 为空的case 时，直接返回0，然后当这个node是叶子节点的时候，就可以往res加 temp,其余就是temp = temp * 10 + root.val
最后左子树，右子树递归

111 Minimum height of binary tree
双Pre
思路就是我们可以求出每条从根节点到叶子节点的高度，通过递归，然后依次比较哪个是最小的


98 Validate Binary Search Tree
双Pre
思路就是我们每次有两个值一个最小，一个最大，每次遍历二叉数节点的时候需要去判断这个节点值是否符合BST的规律

坑： 这里注意在设最大最小值的时候要设成Long整型，因为这里有可能出现Integer 的最大最小值，如果设Integer会造成Overflow


107. Binary Tree Level Order Traversal II (Bottom Up)
BST
思路就是二叉树先正常order遍历一遍，然后再开多一个List，从尾到头塞到新的List里去

226. Invert Binary Tree
BST 
BST 这里不需要按size来遍历，我们就把当前的节点的左子节点存下来，
然后交换左右子节点，然后就一直遍历下去
DFS
根据level 和res list 的 size值来做 
这是固定套路
root 1 res 0 level 0
root 3 res 1 level 1
root 4 res 2 level 2
 
因为是看右边所以这里 右
res 1, 3, 4
走left的res list size 和 level是不一样的，所以这里左边递归不加东西



103. Binary Tree Zigzag Level Order Traversal
BST
这里BST 我们需要一个flag 变量，从第0层打上-1，然后第1层打上1，然后第二层打上-1… 所以我们在for循环之后如果发现flag是1，那么reverse list，
给flag *= -1, 然后最后加list 到res


104. Maximum Depth of Binary Tree
POSTORDER
这里思路是，我们可以求出左子树和右子树的高度然后找出较大值，然后加1得到新的最大深度

250. Count Univalue Subtrees
POSTORDER
思路就是postorder 得到左右子树是否有， 如果都有，那要判断如何左右子树是不合法的，
左子树有，但是左子节点值和根节点值不一样
右子树有，但是右子节点值和根节点值不一样
其次就是合法的
然后加res
最后所有不是左右子树都有的情况全部返回false

428 Serialize and Deserialize N-ary Tree
要把children 的size 存入到String 当中
这里用 递归来求




Count Total numbers of nodes on the complete binary tree

        because we want to count the nodes
        utilize the defintion of the complete tree
        it would be possibly becoming a perfect tree where 
        all the nodes number would become the 2^(H) - 1
        
        so we are going to count the left subtree height and with a 
        left subtree pointer to update our left subtree
        we are going to count the right subtree height and with a 
        right subtree pointer to update our right subtree
 
572 Subtree of another tree
 把t 和s 的子树一一对比

256 Find Leaves of Binary Tree
POSTORDER
思路不是要找出每一个叶子节点，思路是找到叶子节点所在的层数（高度），然后再加入到每一个节点值到答案里去，
怎么找到这个高度？ 根据定义是1 + 左子树高度和右子树高度的较大值，然后找到这个高度的时候，我们应该要检查那个res的层数是否到达了这个高度？如果到达了，说明这个res可以加一个新的list，然后我们就可以加新的节点值到我们的res中深度位置的那个list中了

这个是一个，涉及到postorder

Binary Tree Tilt
这个题目是postorder 的题目，每一个节点都有一个左子树的节点和和右子树的节点和，tile 的定义就是这个两个和之差的绝对值，然后将这些全加起来就是等于结果，

这里要注意的的是子函数recursion 返回的是 left + right + root.val 并不会加入到最后的结果里面去，因为这里我们只需要tilt的和，这个节点和是用来进行递归算左右子树节点和准备的

Diameter of Binary Tree
最长路经 = 左子树的最大深度 + 右子树的最大深度， 子树的 最大深度 = 左子树和右子树的较大值 + 1， 这里res是一个全局变量，然后这个函数的返回值是子树的最大深度，这两者是不一样的



337 House Robber III
POSTORDER
Naive'的思路：
我们这样子想，取决于下一层是否要被robbed的条件是否就是看我们的root是否被robbeed了，如果root被robbed了我们就看孙子树，也就是root.left.left, root.left.right和root.right.left, root.right.right。 这里我们就去更新一个val, 然后去看说我们这里的最后值是这个val + root.val 和左右子树的较大值

236 Lowest Common Ancestor of a Binary Tree 
POSTORDER
这里思路是postorder找到左右子树的LCA，然后分情况讨论
如果只有左子树的LCA → 都在左边
如果只有右子树的LCA → 都在右边
如果两个都有 → 就是root

235 LCA of BST
BST
这里思路是将两个点的值和根节点处作比较，分情况讨论
如果两个点都小于根节点 → 都在左子树
如果两个点都大于根节点 → 都在右子树
如果两个点一大一小 → 就是根节点本身

270 Closet Value of BST
BST
思路先可以把这个数组升序算出来，然后用Collections.min lambda 表达式找到最小值

110. Balanced Binary Tree
我们要判断一个数是否为balanced tree 的条件就是我们要去判断node 于node 之间是否相差1？如果相差1的话，那么我们 的这个返回一个flag 值表示是否是一个balanced binary tree
然后我们去想这个是否是



173 Binary Tree Iterator
BST + Inorder
思路就是用inorder 将所有节点值存到一个queue中，然后next() 就返回queue的poll，hasNext()

就返回!queue.isEmpty()

230 K smallest element in BST
BST + Inorder
思路就是用中序遍历，设一个global count 为k和global res，然后每次中序中间count--, 当count为0时设res 为当时的root值,

314 Vertical Order Traversal of a Binary Tree
这里需要d
  3
  /\
 /  \
 9  20
    /\
   /  \
  15   7
-2 0   2 
这里的层数，怎么把这个负号变成自己的index？比如-1坐标将自己的坐标对0的对称的变成+1
. _ * _ *_*_._*
-2 -1   0 1 2 3
可以将dfs 确定左边右边的界限 min 和 max，这里怎么确定？ dfs 的recursive cases 是  
min = Math.min(min, index)
max = Math.max(max, index)
然后recursion call 左子树index - 1
   “”              右子树index + 1
 知道min 和 max之后， 要对结果进行一个初始化, 然后如果我们还要多一个queue 去存当前的转换后的坐标indices (也就是把坐标转换)
然后调用BFS
这里每次我们怎么加?
cur = queue.poll()
res.get(idx).add(cur.val)
然后我们在检查左子树是否为空时，我们需要去往坐标index 里加入idx - 1，因为是左子树所以我们要把它放在左边
在检查右子树是否为空时，我们需要网坐标index 里加入idx + 1， 因为是右子树所以我们要把它放在右边
这里最复杂的是坐标转换
时间复杂度 O(n) 空间复杂度O（n）

285 Inorder Successor in BST
BST + Inorder
思路是中序遍历，然后在判断i 之后的一个数
坑：判断这个i + 1是否超出数组范围
更好的做法
直接递归去找， 如果root 的值比p 的值要大，那么就往左边找
public TreeNode successor(TreeNode root, TreeNode p) {
  if (root == null)
    return null;

  if (root.val <= p.val) {
    return successor(root.right, p);
  } else {
    TreeNode left = successor(root.left, p);
    return (left != null) ? left : root;
  }
}

拓展： BST去找前继节点
public TreeNode predecessor(TreeNode root, TreeNode p) {
  if (root == null)
    return null;

  if (root.val >= p.val) {
    return predecessor(root.left, p);
  } else {
    TreeNode right = predecessor(root.right, p);
    return (right != null) ? right : root;
  }
}


272 Closest Binary Search Tree Value II
BST + Inorder
思路就是我们需要一个Priority 用lambda 表达式排好序的一个结构，然后inorder存下来list，最后要用一个新的list去存我们前k个数字， 这些数字从PriorityQueue poll出来

99 Recover Binary Search Tree
BST + Inorder
这里我不他清楚怎么找到两个swapped的数字，就是在外面设两个变量来代表两个数，然后如果发现有不符合条件的两个数字，那么就把正确的数字顺序放到对应的变量中，然后返回变量的数字

最后重新遍历一下二叉树，遇到不合法的将之前调整好顺序的数字放回去就好

116. Populating Next Right Pointers in Each Node
preorder 遍历
root.left.next → root.right | root.left != null
root.right.next → root.next.left | root.next != null && root.right != null
117 Populating Next Right Pointers in Each Node 非满二叉树



501 Find Mode in Binary Tree

思路是将利用HashMap 和一个global max 变量，将中序遍历中每一个数字记好，并且将这个数字出现的次数和max去比，直到找到出现次数最多的几个数字，
然后遍历这个map的keySet, 当key出现的次数多于1次的时候，需要将这个加入到list 里面并且，将这个list转成integer array


297. Serialize and Deserialize Binary Tree

BST + BFS

思路 Serialize 用的是postOrder的思路，当得到左右子树的serialize 的String时，我们就可以组成最终的序列

deserialize用的是BFS的方法，但是在加完所有String到queue里面时，我们就用helper函数进行递归， 
坑：注意这里只要遇到之前设下的不能走到的路”X”标记的，那么就提前返回null


108 Convert Sorted Array to Binary Search Tree Binary Search 
思路是找到这个数组的中点，中点的数字就是这个树的root，然后用递归去求出左子树和右子树的节点， 这里用到了二分的递归写法

109 Convert Sorted Linked List to Binary Search Tree Binary Search 
BST + LinkedList find middle point
这里用了链表快慢指针找中点的方法和递归找左右子树，然后每次我们在找到中点时我们要判断有没有一种情况，当链表只有一个节点的时候，
Considering there is only one node in the linked list now, and this node should be the root of the subtree.
The root.left and root.right should all be null, since it is the only node in the linked list.
If you keep use head at the start of this recursion, this can actually cause stack over flows since it is a endless recursion.
So we need to set head as null to avoid that problem.

113 path sum (get all the valid root-leaf paths whose sum equals to target)
这里需要回溯，为什么？因为当我们遇到一个叶子节点时我们要回溯返回上层去visit 另一个叶子节点重复相同的操作

513. Find Bottom Left Tree Value
这里需要去找最后的深度以及更新对应的最左边的值，这里假设最左边的值一定是最深的，那么我们可以用BFS 去做层次遍历然后最后一层去得到最左边的值，或者是用DFS 来找最深的 深度
同时更新res

N-ary Tree Preorder Traversal
跟二叉树前序遍历差不多，但是这里是需要从每个节点的child 节点进行递归调用

Google | Check if a node exists in a complete tree

Put target into stack and divide target /= 2 and repeat as long as target > 1
While poping element from the stack, check if I should go left child or right child. If the value is not found, return false
 
Minimum/Maximum Subtree
    use two global variables for the subtree node and the min/max sum. then use recursion to find the sum of left and right subtree and the root node. Then we should compare whether this sum is smaller/greater than the subTreeSum, replace it if yes, 
107 Binary Tree Level order traversal print the order bottom up
Simplify Path

394 Decode String





4. 全排列，子集，combination sum 小结
这类排列问题首先考虑深度优先搜索
如果结果的List中不能包含相同元素，需要引入变量start来标记循环起始位置，使每次递归不再遍历之前访问过的元素
如果数组含有重复数字首先排序
然后每次循环前判断当前值(nums[i])与前一个值(nums[i - 1])是否相等
考虑res.add(new ArrayList<Integer>(tempList))的时机，是在每次循环中添加还是等达到某个条件是再添加
要排除相同的List，可以利用List中的contains()方法，但这是线性搜索，时间复杂度为O(n),不适合在递归中使用，所以不推荐
————————————————
版权声明：本文为CSDN博主「笑乾」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/genius_wenbo/article/details/79672102

————————————————
版权声明：本文为CSDN博主「笑乾」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/genius_wenbo/article/details/79672102

 
5. 图的算法 小结




图的算法跟树一样是准备面试中必不可少的一块，不过图的方法很容易概括，面试中考核的无非就是两种搜索算法：深度优先搜索和广度优先搜索。LeetCode中关于图的问题有以下几个：
Clone Graph
Word Ladder
Word Ladder II
Longest Consecutive Sequence
Word Search
Surrounded Regions
Count Complete Binary Tree Node
because we want to count the nodes utilize the defintion of the complete tree it would be possibly becoming a perfect tree where all the nodes number would become the 2^(H) - 1 so we are going to count the left subtree height and with a left subtree pointer to update our left subtree we are going to count the right subtree height and with a right subtree pointer to update our right subtree
now if the left subtree height is the same as the right subtree height then we can determine it is a perfect binary tree
if not the perfect binary tree we can return the recursive results for the number of the nodes
先来看看最基础的Clone Graph，很简单就是要复制一个图，常见的两种搜索算法（深度和广度）都可以用，具体细节就不在这里解释了，不熟悉的朋友可以看看相关资料。建议大家还是两种都要练一练，因为在解决具体问题中这两种方法还是很常用的。

接下来的这些题都是基于图算法的应用，Word Ladder和Word Ladder II是比较典型的，看起来好像是字符串操作的题目，实际上这里得转换成图的角度来考虑，因为字符集比较小的缘故（26个小写字母），也就是说对于一个单词来说，改变其中一个字符可以有25条边（除去他自己），所以总共有（25*单词的长度L）条边。找到是否有满足一个单词转成另一个单词就是在这个图中找到一条路径。所以我们可以把问题转换成图用广度优先搜索来解决，找到即可停止。

Word Ladder是广度优先搜索的应用，而Longest Consecutive Sequence则是深度优先搜索的应用。题目要求是找出最长的连续整数串，如果把数字看成结点，与它相邻的整数连有边，那么找到最长的连续串就是在这个图中找最长路径。因为是最长路径，这里用深度优先搜索是比较适合的。

Word Search也是一道深度优先搜索的题目，是把上下左右相邻的结点看成有边联结，然后进行深度搜索就可以了，小细节是这里从每个点出发字符就可以重用，所以要重置一下访问结点。

Surrounded Regions要用一个图形学中很常用的填充算法：Flood fill 算法，其实本质还是一个深度优先搜索，跟Word Search一样是把相邻的上下左右看成连边，然后进行搜索填充。

图的问题其实本质都是两种搜索算法，难点主要在于对于具体问题如何想到转换成图的问题，然后用这两种搜索来解决，这也是算法中的一个分支，面试中也是常客哈。
 Is Graph Bipartite
这个是二分图染色的模板题
通过黑白染色我们可以判断一个无向图是否二分图:

遍历整个图, 将相邻的节点染成不同的颜色, 如果可以完成这个遍历(即染色过程没有冲突), 说明是二分图.
可以用BFS或DFS来实现, 只需要根据当前节点的颜色设定下一个节点的颜色即可, 如果下一个节点已经被染成了相同的颜色, 说明发生了冲突.

迷宫型的图上搜索，
490 The Maze 

这里就是迷宫，但是不是走一步，而是要不停走直到撞墙， 最后如果能到destination 就可以返回true， 否则返回false
时间复杂度？ 这里的时间复杂度为 O(m * n *k) 因为在最差的case内，每个格子都要走一遍, 每个格子都要去转成1
空间复杂度？ 这里的空间复杂度为O(m * n)

505 The Maze II

这里跟迷宫I 差不多，但是要返回的是最短的步数，这里就是要多一个len field在你的 Point class里表示的是一个步数的长度，而还需要一个val Maze 2d数组 来记录我每一个cell 的当前步数，这里每当我的当前点的步数要大于或等于当前的valMaze 的cell的步数，continue 然后设这个cell 的步数
时间复杂度？surrounded 
O（m * n）

499 The Maze III
这里比迷宫III 更复杂， 这里的要求是说我们现在迷宫有个球还有个洞，然后这个球呢需要到达这个洞里，那么怎么在The Maze II 的基础上再加上一层，这就牵涉到我们point的定义了，

坑一：在这里point 的定义指的是我们不仅有x，y，和len，我们还有多一个String 叫path，代表了当前cell的路径是怎么样的。 

坑二： 那么在比较距离之外，The Maze III 还要加上一层对path 的比较，这里可以在class Point里面写上这个比较函数，就是当两个cells的距离都是一样的时候，如果前一个cell 的path 比后一个cell的path要小，那么这前一个就是更小的

坑三： 一定要在while 循环里面判断当前的点是否已经是在洞里了，如果是在的话，那么这里的洞就要跳过

坑四： 在判断是否越界的时候，这里还要判断是否这个点的x或y坐标在hole[0] 和hole[1]这里了,如果不是就可以继续移动球

其他的话就基本跟The Maze差不多，因为这里还有一个点是res是一个Point矩阵了，每次都是更新这个cell上的Point 

286. Walls and Gates
这里用的Flood Fill 思想，每次的我看到gate的时候我向外扩张，只要遇到不是-1 的点，那么我就填上当前的步数和，如果不是的话我们就返回一个值，然后recusion向四个方向扩张

也可以用两个Queue 一个代表x 一个代表y来做

743. Network Delay Time
这道题是一个Dijkstra 的模板题， Dijkstra算法是经典的求解一个顶点到其他顶点的最短距离的算法， 这里我们要理解Dijkstra 的意思
设置起点v
审视参考以前去过的路径，找出可以到达没去过顶点的所有路径
若有，则选择其中代价最小的路径。并标记本次去过的顶点，执行步骤2
若没有，则结束

953. Regions Cut By Slashes
Union-Find 


332. Reconstruct Itinerary
See also here
All the airports are vertices and tickets are directed edges. Then all these tickets form a directed graph.
The graph must be Eulerian since we know that a Eulerian path exists.
Thus, start from "JFK", we can apply the Hierholzer's algorithm to find a Eulerian path in the graph which is a valid reconstruction.
Since the problem asks for lexical order smallest solution, we can put the neighbors in a min-heap. In this way, we always visit the smallest possible neighbor first in our trip.
DFS 递归找从Heap所有的City 的neighbors
但是这里因为用的是最小根，所以最后需要倒序

694. Number of Distinct Islands
这里unique island 的定义指的是当两个island 形状不相同都可以算作是unique islands
那么当我们在想的时候，我们需要做什么？
首先1的个数要相同，但是形状不同的话就算作是不同的岛屿即使1的个数一样，那么这里就可以用到点的相对位置来计算，也就是说我算【基点---右边的点---下面的点---左边的点】就可以出来到底这两个岛屿是否相同。 这里用什么来存，用HashSet 来存这个岛屿的相对位置，利用它自身的除重的特性就可以来做

 
6. 图的转化思想(隐士图) 小结
127 Word Ladder
这里不像是一个图的问题，但实际上这里是一个hidden的graph的问题，我们可以去建立一个图，也可以不建一个图，这里找的是最短距离，这里就是要用BFS/Bi-BFS, 就为了找最短距离，这里有个转化不同单词的小Trick： 咱们不是有一个个单词吧，然后可以通过toCharArray()d来转成新的单词，然后每一个新单词去wordDict里面去验证,然后继续BFS

然后每次遍历更新这个转换次数， 直到到达终点返回这个转换次数，然后这个如果不能达到终点，返回-1

时间复杂度多少啊？ 令L为单词长度, N为单词数目, 那么Set.contain()的平均复杂度一般是O(L). 所以, jiuzhang的算法的复杂度是26 * O(L) * O(L) = O(26L^2)



399 Evaluate Division
这里是一个数学问题，但是这里可以用图的思想来做，
这里我们有一个Equations 的double List，然后有一个对应的结果值，这里要求的是根据我们的query，这里的新的结果是什么

这里可以以图的思想来做‘

构建两个HashMap，一个叫variablesMap ，另一个叫valuesMap，这里variablesMap存的是这个equation的variables。 valuesMap存的是variables对应的values，注意这里如果是反过来的话，那么就返回value的倒数， 到这一步图的建立是关键

当图建好后，如何去遍历？BFS/DFS 即可，新建一个double数组，每个cell通过DFS 算出这个当前的value 乘以valueMap的当前的value，

126 Word Ladder II
这一题比127要复杂很多，因为他是要求输出所有可能路径的单词组成，所以我们要去先用BFS来找到所有的从出发到结束的就最短距离，然后将每一个单词的neighbor加入到HashMap中

2). Use DFS to output paths with the same distance as the shortest distance from distance HashMap: compare if the distance of the next level node equals the distance of the current node + 1.

310. Minimum Height Trees
First let's review some statement for tree in graph theory:
(1) A tree is an undirected graph in which any two vertices are
connected by exactly one path.
(2) Any connected graph who has n nodes with n-1 edges is a tree.
(3) The degree of a vertex of a graph is the number of
edges incident to the vertex.
(4) A leaf is a vertex of degree 1. An internal vertex is a vertex of
degree at least 2.
(5) A path graph is a tree with two or more vertices that is not
branched at all.
(6) A tree is called a rooted tree if one vertex has been designated
the root.
(7) The height of a rooted tree is the number of edges on the longest
downward path between root and a leaf.
Our problem want us to find the minimum height trees and return their root labels. First we can think about a simple case -- a path graph.
For a path graph of n nodes, find the minimum height trees is trivial. Just designate the middle point(s) as roots.
Despite its triviality, let design a algorithm to find them.
Suppose we don't know n, nor do we have random access of the nodes. We have to traversal. It is very easy to get the idea of two pointers. One from each end and move at the same speed. When they meet or they are one step away, (depends on the parity of n), we have the roots we want.
This gives us a lot of useful ideas to crack our real problem.
For a tree we can do some thing similar. We start from every end, by end we mean vertex of degree 1 (aka leaves). We let the pointers move the same speed. When two pointers meet, we keep only one of them, until the last two pointers meet or one step away we then find the roots.
It is easy to see that the last two pointers are from the two ends of the longest path in the graph.
The actual implementation is similar to the BFS topological sort. Remove the leaves, update the degrees of inner vertexes. Then remove the new leaves. Doing so level by level until there are 2 or 1 nodes left. What's left is our answer!
The time complexity and space complexity are both O(n).
Note that for a tree we always have V = n, E = n-1.

7. 拓补排序 小结
207 Course Schedule I

这里涉及拓补排序的知识点，步骤如下
新建一个Map叫graph，一个Map叫indegree
遍历0到number of courses, 每次遍历这个点的neighbors， 如果indegree有这个点的话，就把出现的次数加1， 如果没有的话，就是1
在统计完indegree之后，从0到number  courses 遍历，每次遍历遇到没在indegree出现过的点，就把点加入到queue里
BFS，每次遍历当前点在图中的邻居，然后给这个邻居所在的indegree减一，如果这个indegree是0的话，那么就讲这个neighbor 加入到queue里
最后判断这完成的课程数目是否等于number of courses

210 Course Schedule II
本质跟207是一样的，但是这里需要返回所有课程ordering， 然后只需要一个数组ans和count, ans[count] 当前的课，然后当count等于number of courses的话，就返回ans数组，如果不能到达，就返回一个空数组
  


Alien Dictionary
这里也是一个拓补排序的应用，这里的用处就是将每次character 进行一个dependency的排序，比方说给定一个String的数组，
[
  "wrt",
  "wrf",
  "er",
  "ett",
  "rftt"
]
这里的结果出来就是“wertf“，每个Character 的指向就要用到拓补排序来实现
但怎么建设一个图？
比较好想的就是，每次我们找到当前的单词，然后将没一个单词的Character 放到Map里面，然后这个每一个单词和下一个单词的不同Character 加到graph的value Set里面；
这里有个大坑：就是这里whileloop找到当前单词和下一个单词的不同Character时，加完这个Character到res map里面之后，要直接break，

怎么构建入度？
这里入度的构建就是首先将graph 的每一个Character key填入到res的里面，对应的value是1；
然后这里再第二个for循环里面遍历每一个graph的Character key的邻居nei，然后将nei的对应入度值加1

8. Flood Fill算法 小结
Flood Fill的输入就是一个普通的矩阵，与之相反，图中的DFS 或者BFS都是adjacent matrix，或者是HashMap<V, E>
200 Number of Islands
这里是典型的Flood Fill 算法的实现，每次我们遍历每一个cell，然后对每一个cell我们判断这个cell装的是否是’1’， 如果是，那么我们就用Flood Fill 算法去填这个cell周围所有的cells 并把合法的‘1’ cells变成’0’ 

这里要判断是否越界
然后递归调用四个方向
130 Surrounded Regions
跟200 Number of Islands 很像，但是他这里要多一个坑就是要去判断这个’O’ cell是否是在边界上，如果是在边界上那么我们就要Flood Fill 算法去把这些能convert成’X‘的cell 保持不变，然后将需要surrounded的cell 变成另一个Character,  比方说 ‘W’
然后我们再遍历一次，这次只要看到有之前标记的不能convert的cell，我们把它们变回’O’,否则就改成’X’ cells
286 Walls and Gates
417 Pacific Atlantic Water Flow
我们首先要判断一下一个点能否流入Pacific 或者Atlantic，怎么判断？
这里的高度值要高于或者等于4周围的节点值高度
这里能达到第一列（pacific） 或者最后一列（atlantic) 
或者能达到第一行 （pacific）或者最后一行 (atlantic)
判断完这些之后，我们要去想怎么去找到两个都能流去的点呢？也就是说我们得构建两个boolean 矩阵，一个代表pacific能流的节点，另一个代表了atlantic 能流的节点
想完这些之后我们再想这里怎么去实现这个遍历呢？
用DFS 或者叫Flood fill 可以帮助我们去实现这个遍历，从第一行、第一列开始遍历，用pac 的矩阵记录，然后从最后一行和最后一列用atl 矩阵做记录， 最后得到更新后的矩阵，算出当同一个节点都表示true的话，那么就把坐标加入到res List 中


双指针 算法汇总
Container with Most Water
solution 1 bruteforce
use two loops (0 <= i < heights.length, i + 1 <= j < heights.length)
each loop the area is computed area = min(heights[i], heights[j]) * (j - i)

Time: O(n^2)
Space: O(1)

solution 2 two pointers
two pointers 
    use two pointers to keep track of water area
    the height of the area would be determined as lower height 
    the length between the left and right would be determined 
    during the moving, we need to update our shorter height,
    if the left height is shorter  -- l++
    if the right height is shorter -- r--
O(N) time because each number only visit once
O(1) space
sum of sliding window
use a sum array to record every k’s sum 
so sum[0] is from 0 to k - 1
and sum[i] (1 <= i < sum.length) is sum[i - 1] + nums[i + k - 1] (k is the length of the window) - nums[i - 1]
image a window sliding through the array

2 Sum  - difference equals target
this would be 

3 Sum
since a + b + c = 0. the equation can be re written as 2sum + c = 0
c = -2sum. so we can transfer this problem into 2 sum difference equals target problem. now our targets would be -c 

recover rotated sorted array
use 3 flips method, first find the changed point where the previous number is bigger than the current number, then we flip from the 0th to i - 1th number, from i to nums.size() - 1, and flip from the 0 to nums.size() - 1 number
so we can return the whole thing flipped in-place

---------------------------End of Two Pointer Algorithm-------------------
Remove Duplicate Letters
用一个栈来维护答案，从左往右扫描字符串，当栈顶元素字典序小于当前扫描的字符，并且栈顶元素在s未被扫描到的部分中还有出现时，栈顶元素出栈，并继续比较新的栈顶元素与当前字符字符，重复上面的过程，直到不符合上述条件时，再让当前字符入栈。最后答案就是栈底到栈顶元素组成的字符串。

String Multiplication/Multiply String
We can split the whole problem into several sub problems,
we can first times each characters’ converted integer into each elements of a new array whose size is the sum of the two strings’ lengths
then we can check from the tail of the new array to see if there has a number that is greater than 10, if it is greater than the 10, then add the carry to the previous number, and convert that digit to the last digit it would have. 
Then we can use a StringBuffer to append all the digits, but before that we need to find the first non-zero index in the array. So while the index is still smaller than the new array’s length. append each digit to the StringBuffer


Plus One
this numbers would only increase the last digit by 1 if the digit is smaller than 9, so we can use a for loop to continuously check if the digit is smaller than 9, starting from the tail of the array

then we mark the number to be 0 since it is equal to 9, and 9 + 1 = 10 so the last digit would become 0. 

Then after the nums, we could conclude that only if the numbers are 99, 999, 9999, ... would come to this step, so we can open a new array with nums.length + 1 in length since the carry becomes in the new position.

Excel Column Number 
this problem is like transforming from decimal system to the 26 进制. the res would add (Character - ‘A’ + 1) every iteration, but before we add anything, we need to multiply the numbers by 26 so that we can carry that digit to the next number.
so for example 
“AB”
i = 0, res = 0 -> res = ‘A’ - ‘A’ + 1 = 1
i = 1, res = 1 * 26 -> res = 26 + ‘B’ - ‘A’ + 1 = 26 + 2 = 28


BFS DFS 算法
矩阵上找最小路的深搜模板 (有可能会TLE， 需要记忆化搜索)
res = Integer.MAX_VALUE;
int[][] dirs = {{1, -1}, {1, 0}, {1, 1}};
public void solution(int[][] matrix) {
    int sum = 0;
for (int j = 0; j < matrix[0].length; j++) {
    helper(matrix, sum, 0, j);
}
}
private void helper(int[][] matrix, int sum, int x, int y) {
    sum += matrix[x][y];
    if (x == matrix.length - 1) {
        res = Math.min(res, sum);
        return;
}
// now look for all directions
for (int k = 0; k < dirs.length; k++) {
    int nx = x + dirs[k][0];
    int ny = y + dirs[k][1];

    if (nx < 0 || nx > matrix.length || ny < 0 || ny > matrix[0].length) {
        continue;
}

helper(matrix, sum, nx, ny);
}
    // back tracking
    sum -= matrix[x][y];
}
Knight Shortest Path 
since it’s asking for the shortest path of the knight, so BFS would come in handy, note in this the res needs updated before the level order traversal, and using HashSet would be cause TLE, only can we use the boolean array. Another thing that we need to consider is the direction of the knights, it can move in 8 directions, be sure to clarify that when encounter such problem

K sum II
it is asking if the any k numbers’ sum can be the target. So this problem very much like the subset problem that we did a couple days ago. The main difference is that the subset problem is that when the level reaches the end of the numbers, while this problem has two situations to consider: the k needs to be 0 since that means we run out of our choices, then we need to check if the remain (target after each iteration) equals to 0, if so, that means the numbers in the cur List fits the requirement and are ready to add to the res List. Be sure to recursively call the i + 1 instead of i as we are selecting the next numbers, each number can not be selected twice in this case

Sales man travelling problem
all permutations, but need to transform the n by 3 array to 2 d array, 

each of the costs is N by 3 array. Each element represents the city, so for each array we want to find the x and y’s cost (x = costs[i][0] - 1, y = costs[i][1] - 1), so it would be used to compare if the last city on each row in the costs array. The visited array is needed to record if the characters have been used

The dfs(previous, current city, current cost)
the base case for the dfs is to update the res if the current city cost is larger than current cost, replace it with the current cost

The recursive case would be from 0 to the numbers of the cities, if the traversing city is not been visited or the cost from the 2d array is not equals to MAX_VALUE, then we should consider mark it as visited, and we should begin our next recursion with the current cost + cost between the previous city to the traversing city (2d_array[p][i]). after the recursive calls, we need to do the back tracking

Palindrome Partitioning
the essence of this problem is to every time make choices splitting the given input string s, and the each choice we want to check if the substring is a valid palindrome, so we also need a sub function to check each candidates is palindrome or not, for each recursive call, we need to call the i + 1 since we are iterating the next letter so that we would proceed to the next level
区间小结 
Missing Ranges 
Merge Ranges 
Summary Ranges
Insert Interval


----------------------------END OF BFS + DFS---------------------------------
BFS小结
BFS分层遍历与否的问题
我想问一下BFS的算法在实现过程中，怎么哦按段该不该分层遍历，我知道在二叉树的BFS时，分不分层的区别体现在答案的要求，如果答案要的是将每一层分别存在一个list的话那就需要分层遍历（即循环当前queue的size），但是如果题目只是要求我们写出BFS的结果存在一个大list里面就好的话，那么就不需要做分层遍历了。
我在做Knigh Shortest Path这道题的时候遇上了这个问题，答案是有做分层遍历的，但是我开始在做的时候没有考虑到分层遍历，代码如下：
首先你应该提高一些debug的技巧，这个很重要，如果这是在面试呢？如果我是面试官，我提示你：你的程序有问题，不对，你该怎么办？
抛开前面的问题，你仔细想一下steps究竟表示了什么意思？
如下图：


所以一个点BFS 出去的最短路经就是他的BFS 树形成 的层数

[Leetcode] 994. Rotting Oranges
Bfs 网格题，这里先将rotten orange 的位置放入到queue里，然后算fresh orange 的数量，然后就是BFS 四个方向搜索，然后每次先加rotten的数量，然后当遇到合法的fresh orange， 就将fresh orange的数量减一，最后判断fresh orange 的个数是否为0， 如果是0, 那么就返回rotten的count 减一，为什么要减一？Imagine you are doing BFS on a tree, starting from depth = 0, you do depth++ every level as you go down, adding child nodes into the queue, and when you reach the last level where all the nodes are null(for instance) you are still doing depth++. But essentially nulls are not required so you just do depth - 1 in the end and return. I hope you understood.
[Leetcode] 
317. Shortest Distance from All Buildings
  1 -- Building 
  2 -- Obstacle 
    
    middle 0 to all 1s
    3 + 3 + 1 = 7
    int[][] dist -- every node's distance to building
        int[][] numbers -- every buildings the node can reach
        for 
        for 
            if (grid[i][j] == '1')
                BFS
    Time O (m ^ 2* n ^ 2 )
    Space O(m * n)
 这里我们从一开始搜，用BFS ，这里我们还需要进行一个分层的遍历，因为我们要控制1到周围0的距离，也就是说我们要来填每一个empty space 能到达的buildinig数然后将dist[][] 控制在最小
1 - 0 - 2 - 0 - 1
|   |   |   |   |
0 - 0 - 0 - 0 - 0
|   |   |   |   |
0 - 0 - 1 - 0 - 0
我们要在这里看
这一步，size 控制我们的BFS树的距离，queue 不断poll 出来的值就是控制了我们的距离
如果我们不去控制这个size 的话，(1, 1) 也会在当前遍历， 是不对的

(1, 0) (0, 1) size = 2
(1, 1) comes in 
(1, 1) (1, 0) size = 2, queue poll (0,1)
控制了这size的话，我们就知道这里的(1, 1)是要走两步才能到达，所以这就是为什么size 需要用到的原因



PriorityQueue 小结
如果你想找到第K大的值，你用小根堆， 如果你想找第K小的值，你用大根堆， 因为小根堆是先把小的poll 出去，最后只会剩下大的元素； 大根堆与之相反，它会先将大的poll出去，最后只会剩下小的元素
但是你也可以反过来想，如果对小根堆只加入k次元素的话，那么这时小根堆就是储存从小到大的k个元素


Rearrange String k Distance Apart
这里我们看例子得到一个信息就是，最高频的总是排在前面的，然后第二高频，第三高频...
这里我们很快想到要统计这里的频率和对应的character 的index ，所以用Map，然后这里假设所characters 都是lowercase, 那么这里我们可以不用HashMap，可以用一个26个位置的数组来表示map ，
关键点1： 表示index 和真正character 的关系是 根据 map[c - ‘a’]++ 得到index，
          然后通过 index i, i + ‘a’ 得到对应的character

得到统计之后，我们应该去想怎么去separate k distance from each character 呢？ 这涉及一个排序的问题，因为最大的总在前面，所以我们需要一个大根堆来实现，如果当两个character 频率一样怎么办？ 那么我们可以去按照字典序也就是 (a[0] - b[0]) 来排

之后塞完所有的character 进heap 之后，我们需要开始遍历，这里有个很关键的点
关键点二： 什么时候结束? 当我们的heap size 为0的时候，但是要注意这里不是所有heap size为0的例子都可以，这里如果i 不为k - 1 或者StringBuilder 的长度不是等于String 的长度，意味着这里不能rearrange 所以要return “”, 

关键点三：另外需要注意的是我们map 里面可能有很高频的character 还没被用完，这里就需要一个 list 去将每次放入到heap里面的cur[0] 放进去，然后在heap poll 完k 次之后，再次检查是否map还有没用完的character， 最后将没用完的重新加入到heap里面

时空复杂度分析：O(klogM) —  M 代表了不同字符的个数，至多有26个，所以这里也可以说是O(1) 
空间复杂度 O(logM) 开了一个heap — O(1)
373. Find K Pairs with Smallest Sums
最正常的，根据example 我们可以想到这里有一个排序的问题，也就是说我们的pairs 的combinations 是有一个sum 从大到小的顺序在的，所以这里我们用堆来实现。这里堆的排序是倒序的，然后我们维护一个大小为k的 大根堆 ， 最后更新我们的答案
时空复杂度是O(n^2logK) — 空间是 O(logK) 比较naive 的做法

有没有可能会更加好的想法呢？
因为我们继续观察发现，这里的array 都是sorted 的，意味着我们可以从每次从nums1 中拿数字出来时，nums2[0] 是最小的partner，所以下一个candidate是 this candidate number + nums2[current_associated_index + 1]  除非out of bound

如上图所示，每当我们去存int[] 到heap的时候，一开始for 循环存的是 {nums1[i], nums2[0], 0}
然后我们每次while 循环去找candidate的时候，需要 去存
我们需要去存{current number (nums[1]), nums[asscoiated_ number_index + 1], asscoiated_number_index}  
378 Kth Smallest Element in a Sorted Matrix
这里跟373相类似，也可以用二分搜索的方法去做，这里我们维护一个大根堆，然后每次让这个堆得元素个数等于k， 最后返回堆顶元素就是第k小的
218. The Skyline Problem 

264. The Ugly Number II
PriorityQueue
        关键点一：将ugly number放入到heap 中每次如果有相同的数字，就pop 出来
        每次poll 一次 temp = heap.poll(), 然后将 temp * 2, temp * 3, and temp * 5 放入
        
        关键点二：这里是将 Long 整形作为Heap 的数字，这样避免overflow
        
        关键点三： 最后heap 顶是要转化为int 数组()
215 Kth Largest Element in an Array
维护一个小根堆，每次堆的size 大于k， poll 出来，最后堆顶的就是第k大的数
347 Top K Frequent Elements
这里可以通过Lamda表达式对HashMap里面的value进行一个排序，PriorityQueue<Integer> heap = new PriorityQueue<>((a, b) -> (map.get(a) - map.get(b))) 小技巧： map.put(num, map.getOrDefault(n, 0) + 1) 就跟以前没有Key value 放0 else 放更新值是一样的
295 Find Median from Data Stream
用两个堆来维护中位数， 大根堆来维护较中位数小的排序，小根堆来维护中位数较大的数字，每次讲数字加到大的数字堆里，然后将大的数字堆顶元素放到小的数字堆里，然后检查是否小的数字堆比大数字堆大，如果是，那么就将小数字堆顶元素放入到大的数字堆里
 

Binary Search 二分法小结
[]

DFS 算法小结
[Leetcode] 733 Flood fill 
网格搜索DFS 模板题，
[Leetcode] 257. Binary Tree Paths
这里是我用了StringBuilder去更新current String，但是如何限制String 的长度将它放到答案里需要在recursion之前拿到StringBuilder 的长度，最后在设置递归后的StringBuilder 的长度为之前的长度，
[Leetcode] 100. Same Tree
递归分析三种情况， 当p和q都为null时， 返回true， 当只有一条树为null时，返回false，当两个树的值不一样时返回false，递归返回左右子树值
[Leetcode] 108. Convert Sorted Array to Binary Search Tree
找到这个sorted array 的中点，然后将这个树的root的值定义为这个数组的中点值，然后，根据递归调用，分别使用从左端点到中点减一求左子树，从中点加一到右端点求右子树，
[Leetcode] 690. Employee Importance
首先用map去记录每一个employee的id和employee；然后根据id 去找到对应的employee， 然后通过递归去找到每一个subordinate 的importance 然后加到sum
[Leetcode] 872. Leaf-Similar Trees
先用递归方法将每个node 的值放入到一个List中，然后对比这两个List的值，如果有不一样直接返回false， 否则返回true
[Leetcode] 897. Increasing Order Search Tree
先通过递归中序遍历拿到node的值存到List中，然后根据这个List的每个值重建一条只有右子节点二叉树
[Leetcode] 559. Maximum Depth of N-ary Tree
这里通过递归调用去比较每个node 的children node的maxdepth 求最大depth
[Leetcode] 101 Symmetric Tree
这里有三种情况，1. 这Tree要symmetric的条件是两个子树的root的值相同
                           2. 这两个subtree 其中一个子树的左子树和另一个子树的右子树相同
                           3. 当这两个子树中其中一个不为null时，直接返回false
[Leetcode] 110. Balanced Binary Tree
检查两个子树返回的height 之差大于1 如果大于1， 那么结果就是false，否则就是true
[Leetcode] 111. Minimum Depth of Binary Tree
检查每次当到叶子节点时候，这个sum 和全局变量res的比较
[Leetcode] 979. Distribute Coins in Binary Tree
if node has greater one coin, 
we can have the coins number on the subtrees dfs(node) would return the [# of nodes in the subtree, # number of coins in the subtree]. the moves of the subtree can be abs(# of nodes - # of coins). then return the [number of nodes of subtrees + 1, # number of coins of subtrees + root.val]
[Leetcode] 993. Cousins in Binary Tree
The nodes are considered as cousins if their parents are not the same and they are at the same height. So we could use dfs or bfs to check if this criteria is satisfied or not
DFS:
    Our helper function parameters are root, x, y, depth, parent
    if we define four global variables: xParent, yParent, xDepth, and yDepth denoting the parents and the depths of x and y nodes. We could begin recursion on the binary tree: when the node.val is equal to x, then we set xParent to be parent node and xDepth will be depth. If the node’s value is equal to y, then our yParent will be parent node, and yDepth will be depth
at the end, we need to check the criteria 

BFS:
we can set up two boolean values checking if x or y is existing, then every time we do our level order traversal, we check the current node’s left and right child is x and y or y and x → false if yes, then we set xExist = true if current node value is x or yExist = true if current node value is y 

DFS + 回溯法 算法小结
123. Word Search
135. Combination Sum
570. Find the Missing Number II
152. Combinations
913. Flip Game II
1020. All Paths From Source to Target
426. Restore IP addresses
[Leetcode] 531. Lonely Pixel I
1612. Smallest Path
164. Unique Binary Search Trees II


--------------------------------------END------------------------------------

10. Stack & Queue小结

Implement a Queue using two stacks
Divide the task into several sub problems
initialized the two stacks
realize the push method: input? an integer element, output? void
realize the pop method: input? nothing, output? int
realize the top method: input? nothing, output? int

first we can initialize the two stacks, called s1 and s2, in the 
inorder to solve the subtask we can start off the push method. The element is just pushed into one of the stacks, maybe s2.

then we can start off by our pop method, recall the characteristic of stack is LIFO (last in first out), and our queue is FIFO (first in first out), so given a example
we have s1 []   s2 [1,2,3], we want our queue pop method to pop 1 when we call pop method, but stack can only pop 3 in this case. So we can first put all the numbers from s2 to s1, so that now we have s1 = [3, 2, 1] s2 = []
return s1.pop() as the pop result for the custom queue structure. but there is one more thing to consider, because when our new updated s1 is not empty, then we can just pop out the elements from the first stack. Only when the s1 is empty, we need to push our s2 elements to s1.

for the peek method we could try to get on the top of the queue what would be the element? input? nothing output? int 
also because when our new updated s1 is not empty, then we can just pop out the elements from the first stack. Only when the s1 is empty, we need to push our s2 elements to s1.
return s1.peek()

Implement Stack by two Queues
Q1 = []
Q2 = [1, 2, 3]
stack.push      → Q2.offer()
stack.pop()     → 3 → move all the Q2 elements to Q1, return the Q1.poll() but need to check if the Q1 is empty, only do the move() when Q1 is empty
stack.isEmpty() → check if Q1.isEmpty()
which method do we need to realize?
push, pop, and isEmpty
sub-tasks
initialize the Queues
realize the pop method
realize the push method 
realize the top method
realize the isEmpty method
The pop() and push methods needs to swap the queues as to maintain the stack structure. For example, given the stack that we want to realize is stack = [1 2 3(top)] 
push → Q1 = [1 2 3], Q2 = []
Move() do things shown below
Q1 1 2 3
Q2

Q1 2 3
Q2 1

Q1 3
Q2 1 2 
pop() would do the things below
after move()
Q1 3
Q2 1 2 
Q1.poll() → 3
Q1 
Q2 1 2
swap()
Q1 1 2
Q2 

top 
Q1 3
Q2 1 2
move --
val = Q1.poll()
Q1 
Q2 1 2
swap()
Q1 1 2
Q2 
Q1.offer(val)
Q1 1 2 3
Implement Queue using Circular Array
Some basic operations from the Queue, the Queue has three properties needed to define at first
The front pointer: the pointer displaying the front of the queue
The rear pointer: the pointer displaying the rear of the queue
The size pointer: the size of the queue
There are four methods that we need to realize:
isFull(), isEmpty(), enqueue(), and dequeue()
so the isFull() and isEmpty() are judged based on the size pointer and the actual length of the CircularQueue()

isFull() is just determined on whether the size is equal to the length of the circular array

isEmpty() is determined on whether the size is 0

enqueue() we need to determine our rear pointer position by moding the (front + size) with CircularQueue’s length, and put input element to the rear position in the array, and we update our size to be size + 1

dequeue() we need to determine our front element by using frontElement = CircularQueue[front]; and we need to update our front pointer to front + 1 position since we are polling out the first element which got into the Queue
and we  update the size to size - 1
Min Stack
there are three methods: push(), pop(), and min()
we need another stack to maintain the smallest element from the original stack

stack  [2 3 1 1 0]

minStack [2 2 1 1 0]

pop the two stacks at the same time

Largest Rectangle in Histogram
单调栈的应用
traverse the array and use a minstack
Every time check if the current number is smaller or equal to the element on the top of the stack. Then we should think about how to find the height and the width of the rectangle. The height must be the shorter one, whose index is saved in the stack, and the width is retrieved between the index i and the top element of the stack 
Evaluate Reverse Polish Notation
What is Reverse Polish Notation?
whenever we meet an operator, we do that multiplication 
[4 13 5 / +]
13 5 / would be turned as (13 / 5)
and push back into the stack
[4 (13 / 5) +]
then we had 4 + (13 / 5) as our return value
So we can use stack to push all the digits to the stack, when we encounter the operators, we then can pop two numbers from the stack and do the operation. Then we can do 
Binary Tree Iterator
这是一个非常通用的利用 stack 进行 Binary Tree Iterator 的写法。
stack 中保存一路走到当前节点的所有节点，stack.peek() 一直指向 iterator 指向的当前节点。
因此判断有没有下一个，只需要判断 stack 是否为空
获得下一个值，只需要返回 stack.peek() 的值，并将 stack 进行相应的变化，挪到下一个点。
挪到下一个点的算法如下：
如果当前点存在右子树，那么就是右子树中“一路向西”最左边的那个点
如果当前点不存在右子树，则是走到当前点的路径中，第一个左拐的点
访问所有节点用时O(n)，所以均摊下来访问每个节点的时间复杂度时O(1)



------------------------------Queue & Stack--------------------------------

Swap two array once to make their sum equal, if can return true if not return false
Input? -- two int[] arrays      Output? -- boolean
clarifying questions? 
1 are there any negative numbers in the array?
2 duplicates in the array?

if need equal sum, then each of sum should be the total / 2
a = [4, 1, 2, 1, 1, 2] sum = 11
b = [3 6 3 3] sum = 15

total = 26 each array needs to be 13
for this example we need to swap 1 from a and 3 from b
a’ = [4 3 2 1 1 2] sum = 13
b’ = [1 6 3 3]     sum = 13

Or 
A’ = [6 1 2 1 1 2] sum = 13
b’ = [3 4 3 3] sum = 13
Each array needs to be total / 2
So let say a is less than total/2 then we can define a difference variable as diff =  total/2 - sum of the array
Then we can start off by doing a navie approach
For each element in a called a[i], we want to check if the element in b called b[j] is diff greater/less than the a[i], so if we check find a pair (a[i], b[j]) that abs(a[i] - b[j]) == diff, we can just return true
If we can’t find the pair at the end we can return false
Solution Naive Approach, try every pair and see if the new sums are equal
newASum = oldASum - A[i] + B[j] 
newBSum = oldBSum - B[j] + A[i]
if (newASum == newBSum) return true
Time Complexity? O(m * n) (m is the length of array A, n is the length of array B)

package com.yifeng;

/*
* params: int[] A, int[] B
*
* output: boolean isPairFound
* */

public class FindSwappedPair {
   private static int getSum(int[] nums) {
       int m = nums.length;
       int res = 0;
       for (int i = 0; i < m;i++) {
           res += nums[i];
       }
       return res;

   }
   public static boolean swapTwoArrays(int[] A, int[] B) {
       if (A.length == 0 || B.length == 0) {
           return false;
       }

       if (A.length == 0 && B.length == 0) {
           return true;
       }


       int m =  A.length, n = B.length, total = 0;
       int ASum = getSum(A);
       int BSum = getSum(B);

       // try each pair
       int newASum = 0, newBSum = 0;

       for (int i = 0; i < m; i++) {
           for (int j = 0; j < n; j++) {
               newASum = ASum - A[i] + B[j];
               newBSum = BSum - B[j] + A[i];
               if (newASum == newBSum) {
                   return true;
               }
           }
       }

       return false;
   }

   public static void main(String[] args) {
       int[] A = {4, 1, 2, 1, 1, 2};
       int[] B = {3, 6, 3, 3};

       int[] A2 = {0, 1};
       int[] B2 = {0};


       System.out.println("Correct Test Case: ");
       boolean result = swapTwoArrays(A, B);

       System.out.println(result);

       System.out.println("Incorrect Test Case: ");
       boolean incorrectResult = swapTwoArrays(A2, B2);

       System.out.println(incorrectResult);
   }
}
Find Duplicates in an array
use HashMap to record the number in the numbers array and the occurances
Time (O(N)) Space (O(N))

10/02/2019
Longest Increasing Subsequence
each time we can use a hash set to add all the numbers into the set
then we can find a range [l, r] where the l and r are on the left and 
right number on each number of the numbers interval is the biggest then if we check the set if l is in if it is in then we remove the number in set and we move our pointer to the left then we check if r is in the set if r is in the set, then we remove the number from our set and move r to the right

Subarray Sum Equals To K
what if we have a map to store all nums and its counterpart?
like we want to store prefixSum and its occurance into the map
also we need the 
and we would define a ans and a prefixSum 

Find Right Interval
This problem should use binary search for the interval searching, but before that we should use a hashmap to store all the intervals and their indices. Then we should also sort the list by its start positions. Then we should traverse the whole interval list so that we can do the binary search for each of our interval. so how to use the binary search? 

The binary search that we have is that the target should be each current interval’s end position, and our start and end would be 0 and intervals.size() - 1 and we need to find that interval from the intervals list so it’s a parameter as well. The 
Non-overlapping intervals
essentially we want to find the number of intersection of intervals
        greedy approach
        since we are working on the intersection problem
        choose the next interval that definitely have the intersection
        two intervals have intersections [a, b] [c, d] if 
        c < b will be the condition
        or in other words
        when a >= d we can guarantee there would be no intersection between
        the intervals
        and the number of removal would be the total number of intervals - 
        the number of the intervals with no intersection

Flipping Game II
将博弈问题转化成重叠子问题
if the number of '+' or '-' are not even that means the game cannot be winned with less than 2 string, it is impossible to even form a consecutive symbols, so we should return false directly
then we need to think about playing the game if it's my turn and I saw there is a consecutive characters in the given string, "++", then we need to think about change that state into a new state by flipping those consecutive characters into the "--", so we could pass this new state to our opponent 
to see if this is a win for him/her, if they can't get a win, then we must secure the win, at the end of the loop if there is no win showing up, then we should return false

Campus Bike
Time Complexity: O(m * n * log(m * n)), inserting into heap would cost log(m * n)
Space Complexity: 

Use a priority queue to store {dist, worker index, and bike index} and use comparator to handle situations such as when dist is the same, and when worker is the same. 



DP小结
怎么样去得到DP的解？
考虑最后一步来想状态， 很想数学上的数列问题，


---序列型DP---
Leetcode 357 Count all unique numbers from digits
This is a digit combination problem. Can be solved in at most 10 loops.
When n == 0, return 1. I got this answer from the test case.
When n == 1, _ can put 10 digit in the only position. [0, ... , 10]. Answer is 10.
When n == 2, _ _ first digit has 9 choices [1, ..., 9], second one has 9 choices excluding the already chosen one. So totally 9 * 9 = 81. answer should be 10 + 81 = 91
When n == 3, _ _ _ total choice is 9 * 9 * 8 = 684. answer is 10 + 81 + 648 = 739
When n == 4, _ _ _ _ total choice is 9 * 9 * 8 * 7.
...
When n == 10, _ _ _ _ _ _ _ _ _ _ total choice is 9 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1
When n == 11, _ _ _ _ _ _ _ _ _ _ _ total choice is 9 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1 * 0 = 0
 

Longest Increasing Subsequence
dp[i] means the LIS length ends with i, 
dp[i] = max(dp[i], dp[j] + 1)  0 <= j < i, if nums[i] > nums[j] 
dp[0] = 1

Jump Game
dp = boolean[nums.length]
dp[i] whether can we jump to i position
dp[i] = true if dp[j] is true and nums[j] > i - j
dp[0] = true
return dp[nums.length - 1]
Jump Game II
dp = int[nums.length]
dp[i] means smallest jump steps to i, initialize dp[i] = MAX_VALUE
dp[j] -> dp[i]: if dp[j] is reachable (dp[j] + 1 < dp[i]) and nums[j] >= i - j
dp[i] = dp[j] + 1

return dp[nums.length - 1]
494 Target Sum
跟314 有点像，全部加上正数，sum就是+sum, 全部加上负数， sum 就是-sum 
-sum                  0                     +sum
转移坐标轴之后就是
0                  sum                    2sum + 1 （为什么？从0开始）
dp 
定义dp[i]： 这个第i个的数字组合的sum
transit function
dp[]

 this problem has sub-structure that 
if we want to know whether it can make up to S
we need to decide whether we want to choose the number, say nums[i]
        
then we could convert it into 01 knapsack problem
initialize
int[][] dp = new int[nums.length + 1][S + 1]
transit
if we don't select nums[i] 
        
this means the previous state already is make up to S
dp means how many ways we can select for first i-th elements to make up weight j
        
dp[i][j] = dp[i - 1][j]
        
if we do select nums[i]
that means we the previous state is dp[i - 1][j - nums[i]] given j >= nums[i]
but the problem has something different with the classic knapsack problem
Here the question is presenting whether we need to add or minus the numbers
        
so the transit function really becomes 
dp[i][j] = dp[i - 1][j + nums[i]] + dp[i - 1][j - nums[i]]
Triangle
相邻问题: 
i: j
i + 1: j, j + 1

从下往上走，一层层扫，
因为从上往下扫是不能找到这个结果的
还有一个规律是最后一层的元素个数跟这个dp 的长度是一样的


---坐标型DP/网格DP---
Maximum Squares
10 1 0 0
1 0 1 1 1
1 1 1 1 1
1 0 0 1 0

当前状态需要看上一个状态
能否构成正方形需要看左上方，上方，左方的格子都是1，就能form rectangle
dp[i][j] -- side length of the maximum square whose bottom right corner is the cell with index (i, j) in the matrix
经过一次dp[i][j] 每次看这个cell的左边，左上边，以及上边的cell 是否为 1？
比方说matrix[2][3] 为‘1’，符合这个条件，这里的dp[3][4] (从1 开始)
 
        1 0 1 0 0 
        1 0 1 1 1
        1 1 1 2 1
        1 0 0 1 0
就变成了2， 这个2代表什么意思？ 这个2代表的意思是我的这个正方形的边长是为2
所以这里的res 代表的是这个最大边长，所以这个最后返回结果是 res * res 表示面积

Unique Paths 
Unique Paths II
Unique Paths III
Climbing Stairs
Dungeon Game
Maximum Square

Minimum Path Sum
the dp[i][j] represents the minimum path sum at (i j), the transit would be 
dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]
for initialization 
dp[0][0] = grid[0][0]
then 
dp[0][j] = dp[0][j - 1] + grid[0][j]
dp[i][0] = dp[i - 1][0] + grid[i][0]

finally we want to find the dp[grid.length - 1][grid[0].length - 1]
O(m * n) space O(m * n)
Unique Path
dp[i][j] = the number of unique path when at (i, j)
dp[i][j] = 0 (grid[i][j] == 1) or dp[i - 1][j] + dp[i][j - 1] (grid[i][j] == 0)
for initialization
dp[0][0] = 1 
dp[0][j] = 0 (grid[0][j] == 1) or dp[0][j - 1]
dp[i][0] = 0 (grid[i][0] == 1) or dp[i - 1][0]

finally return dp[m - 1][n - 1]
knights shortest path II
direction matrix {{-1, -2}, {1, -2}, {-2, -1}, {2, -1}}
initialize dp[0][0] = 0 and all other cells as Integer MAX VALUE
then dp[i][j] -- the shortest path of knight at {i, j}
the transit would be dp[i][j] = min(dp[i][j], dp[nx][ny] + 1) where x = i + dirs[k][0], ny = j + dirs[k][1]
of course, this problem can also use BFS to do 

Bomb Enemy


匹配  DP
Decode Ways
非常重要
这个就是DP
肯定要 优化到最优解
给我们一个String, String 里面是数字，1 代表a， 2 代表b，“12” 就可以代表 “AB” 或者“L” 最后可以返回一个数值为2，
version1 空间复杂度为O(N)
当前这个第一个数字只能是1或者2， 第二个数只能小于或者等于6
dp = new int[nums.length + 1]
dp[0] means the empty string would only have 1 way to decode
dp[1] means the way to decode a string with size 1
...
dp[i] means the way to decode a string with size i
就判断前一位和前二位的范围来加
so dp[i] += dp[i - 1] if String.substring(i - 1, i) is between 1 and 9
   dp[i] += dp[i - 2] if String.substring(i - 2, i) is between 10 and 26
finally return dp[len]

这里可以简化
可以用 c1 和 c2 来控制这个组合个数
"1231"
“12” --> "AB" "L"
C1 = 1 + 1 = 2
C2 = 2 - 1 = 1
cc 
21 

"123"
c1 = 3
c2 = 2
            
“ABC”
"AW"
 "LC"
            
i = 3
"31"
只执行else
c2 = 3      
"1231"
c2 "123"  
最后只返回c1， 判断c1 或者c2 能否组合成1个数的次数， 
Edit Distance
从word1 到word2 我们需要一个二维的数组
DP基础题目， 必须要会
dp[i][j] - 从字符串1 i 的位置转换到字符串2的j的位置，所需要的最小步数
题目的状态转移公式，
1 如果两个单词是相等的，那么三种操作都不用用，所以相等于跳过
dp[i][j] = dp[i - 1][j - 1]
2 如果不等， 可以通过a.增加 b. 删减 c. 替换
    a ar -- ar ar 
    INSERT: dp[i][j] = dp[i][j - 1] + 1 
    ae ar -- ar ar
    REPLACE: dp[i][j] = dp[i - 1][j - 1] + 1
    ar a -- ar ar
DELETE: dp[i][j] = dp[i - 1][j] + 1
如果不等的话，dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
Test Case the walk through
   /*
                a  b  c  d
             0  1  2  3  4
          a  1  0  1  2  3
          e  2  1  1  2  3
          f  3  2  2  2  3    */
    



这道题乍一看还想不到用DP解
子问题：
上一步是什么样的？
状态转移方程：
f[i][j] 代表把一个长度为i的word1转换成长度为j的word2所需步数
我们来考虑最后一步的操作：
从上一个状态到f[i][j]，最后一步只有三种可能：
添加，删除，替换（如果相等就不需要替换）
a、给word1插入一个和word2最后的字母相同的字母，这时word1和word2的最后一个字母就一样了，此时编辑距离等于1（插入操作） + 插入前的word1到word2去掉最后一个字母后的编辑距离
f[i][j - 1] + 1
例子： 从ab --> cd
我们可以计算从 ab --> c 的距离，也就是 f[i][j - 1]，最后再在尾部加上d，距离+1
b、删除word1的最后一个字母，此时编辑距离等于1（删除操作） + word1去掉最后一个字母到word2的编辑距离
f[i - 1][j] + 1
例子： 从ab --> cd
我们计算先删除b，先+1，再加上从 a --> cd 的距离： f[i - 1][j] ，
c 、把word1的最后一个字母替换成word2的最后一个字母，此时编辑距离等于 1（替换操作） + word1和word2去掉最后一个字母的编辑距离。
这里有2种情况，如果最后一个字符是相同的，即是：f[i - 1][j - 1]，因为根本不需要替换，否则需要替换，就是
f[i - 1][j - 1] + 1
以上abc三种情况取其最小值
d. 最后一个字母相同，不需要操作： f[i][j] = f[i-1][j-1]
初始条件
当word1和word2都为空：f0][0] = 0
当word1为空，f[0][j] = j , 挨个添加word2字母
当word2为空，f[i][0] = i, 挨个删除word1字母
计算顺序
从小到大
Distinct Subsequence
DP 基础
几维？ 2维
dp[i][j] means the number of Subsequences of S which  equals to T

2 种情况
a S[i] != T[j] → dp[i][j] = dp[i - 1][j]
b S[i] == T[j] → dp[i][j] = dp[i - 1][j] + dp[i - 1][j - 1]
“r”  ra
S → T
矩阵表示如下
  1  r a b b i t
  1  0 0 0 0 0 0
r 1  1 0 0 0 0 0
a 1  1 1 0 0 0 0   -- “ra” “ra”

b 1  1 1 1 1 0 0
b 1  1 1 2 1 0 0
b 1  1 1 3 3 0 0
i 1  1 1 3 3 3 0
t 1  1 1 3 3 3 3
dp[i][0] = 1 
最后return dp[S.length()][T.length()]

97. Interleaving String
可以用backtracking + memoization
/*
        if s1[i] is the same with s3[k] -- keep backtracking 
        i + 1 and k + 1
    
         if s2[j] is the same with s3[k] -- keep backtracking 
        j + 1 and k + 1
    
        
        if i hits the s1 end -- check if the substring of s2 from j .. end is same with the k .. end: 
        
        if j hits the s2 end -- check if the substring of s1 from i .. end is same with the k .. end
        
        remember to check with memo if memo[i][j] >= 0 -- return if memo[i][j] == 0 ? false : true
    */
Regular Expressions Matching



特别典型DP
思路想明白就可以
几维？二维
s1 = “aabcc”, s2 = “dbbca”
when s3 = “aadbbcbcac” return true 
when s3 = “aadbbbaccc” return true 
corner case is s1.length + s2.length != s3.length -- return false;

dp[i][j] means the s2[i] and s1[j] 可否组成一个s3的valid character？

[f, f, f, f, f, f]
[f, f ,f ,f ,f ,f]


背包问题DP
01背包
416 Partition Equal Subset Sum
01 背包问题 这里我们需要用到背包思想: 对于一个背包和物品，我们有两种选择：选这个物品还是不选这个物品，然后对这个问题来说，我们需要将这个总的和分成两等分，然后每等份,
所以我们用一个boolean数组dp[][] 来表示这个和为j 当我们是从前i个物品的时候，如果是这个和j的话，那么这个dp[i][j]为true，反之为false
transfer function
怎么去定义这个transfer function 呢？
如果我们不去选, 意味着这个之前i - 1 个元素的和已经达到这个和j
dp[i][j] = dp[i - 1][j]
如果我们去选，意味着这个状态还差nums[i] 才能到j，
dp[i][j] = dp[i - 1][j - nums[i - 1]]
这两种情况二选一
总的transfer function 就是
dp[i][j] = dp[i - 1][j] || dp[i - 1][j - nums[i - 1]] | j >= nums[i - 1]
初始条件
当没有物品时，重量为0的一定是正确的 — dp[i][0]  = true 0 <= i < n
当没有物品时，非0的重量一定是不可能的 — dp[0][i] = false 1 <= i < sum / 2
边界条件 
最终返回dp[nums.length][halfSum] — 所有物品能否拼出半个重量
 
1049  Last Stone Weight II
跟上题思路差不多 经典 01背包问题
先算出所有石头的总的重量S是多少
要找出这最后一个石头的重量是多少之前，我们要去想怎么去优化这个最后一个石头的重量？这里我们给出了说两个set, 第一个是S1， 第二个是S2
S = S1 + S2
S1 - S2 = diff, 
diff = S - 2 * S2
这里diff 表示的就是最后一个石头的重量，我们要diff越小越好，那么表明我们要S2越大越好, 所以这个问题就是去最大的S2。
dp[i][j] — 前i个物品能拼出j的重量
然后transit function分情况讨论
不选, dp[i - 1][j] 已经是达标的， 那么就是算作一种
选, dp[i - 1][j - stones[i - 1]], 需要前一个石头的重量来进行达到重量j的操作，
这两个情况任意一个满足的话，那么我们的dp[i][j] 就可以设成true， 并且更新S2
初始条件
dp[i][0] = true (0 <= i <= stones.length)
 
边界条件
return S - 2 * S2
 
 
322 Coin Change
01 背包问题 — 
 
我们这里要找到最小的coin 的使用数去凑一个值，
State 
dp[i]  i价值的最小硬币数目，
state transfer function
然后我们有一种选择就是不选选当前的coin，这样就是dp[i]
或者是前面 的价值就差了 coin[j]，今天i 天加上这个coins[j]就可以得到想要的和了
 
dp[i] = min(dp[i], dp[i - coins[j]] + 1) |  1 <= i < amount, 0 <= j < coins.length && i >= coins[j]
初始条件
当我们将每一个dp格子设成amount + 1， 除了dp[0], 代表了没有价值有多少种硬币？ 有1中方式就是没有硬币， 
边界条件
判断dp[amount] 是否大于amount; 如果是就直接返回-1， 如果不是就返回dp[amount]
518. Coin Change 2
377 Combination Sum IV
474 Ones and Zeros
139 Word Break
140 Word Break II
494 Target Sum
 
完全背包问题DP
Coin Change II
Ones and Zeros
多重背包问题DP
Target sum

一维DP问题
maximum subarry
maximum product subarry
字符串问题
10. Regular Expression Matching
dp = dp[s.len + 1][p.len + 1]
 情况分类
1 s[i  - 1] = p[i - 1] dp[i][j] = dp[i - 1][j - 1]
2 p[i - 1] == ’.’ dp[i][j] =dp[i - 1][j - 1]
3 p[i - 1] == ‘*’
  3a p[j - 1] != s[i - 1] && p[j - 2] != ‘.’: dp[i][j] = dp[i][j - 2]
      3b p[i - 2] == s[i - 1] || p[i - 2] == ‘.’
         3b1 dp[i][j] = dp[i][j - 1]   a* -- a
         3b2 dp[i][j] = dp[i - 1][j]  a* -- aa or more
         3b3 dp[i][j] = dp[i][j - 2]  a* -- empty
3b1 || 3b2 || 3b3 means any condition is true
 
 
Best Time to Buy and Sell Stock问题
Best time to buy and sell stock
state dp[i] max profit at i day
state transit
dp[i] = what does max profit on ith day stands for?
ith day max profit is either 
previous day’s profit + prices difference of selling the stock
OR 
I only sell this stock on this day. So my profit would only be 
prices difference
so I only pick the maximum from these two

final return my ans
Best time to buy and sell stock II
greedy find the positive prices difference every day and add it to the array

Best time to buy and sell stock III
only perform at most 2 times at the end of season
substructure 子问题
at ith day max profit with j transaction
dp[i][j] -- at ith day max profit with j transaction 第j个transaction 在前i天的最大收益

state transfer
前一天以持有
dp[i][j] = dp[i - 1][j - 1] + prices[i - 1]
dp[i][j] = max(dp)


Best time to buy and sell stock with 1 day Cooldown



 


