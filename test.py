def count_codes(S):
    """
    Count all 4-digit numbers whose digits sum to S.
    """
    count = 0

    # Iterate through all possible values for four digits
    for d1 in range(10):
        for d2 in range(10):
            for d3 in range(10):
                d4 = S - (d1 + d2 + d3)  # Calculate the fourth digit
                if 0 <= d4 <= 9:        # Ensure it's a valid digit
                    count += 1

    return count

# Example usage
S = 4
print(count_codes(S))  # Output: 35