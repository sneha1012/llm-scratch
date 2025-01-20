import sys

def solution(points, tokens):

    indexes =  [i for i, c in enumerate(tokens) if c == 'T']
    ans = sum(points[i] for i in indexes)
    adjacent_count = sum(1 for i in range(1, len(indexes)) if indexes[i] == indexes[i - 1] + 1)
    return ans + adjacent_count

points= [3, 2, 1, 2, 2]
tokens = "ETTTE"
print(solution(points, tokens))