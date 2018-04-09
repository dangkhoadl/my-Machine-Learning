############################################################### Sort ###############################################
data = [1,4,2,3,5,9,8,7,0,6]

# Sort
data.sort()
data.sort(reverse=True)

# Create a new sorted data without sort old data
d = sorted(data)


############################################################### Lambda ###############################################
# Lambda
my_function = lambda a, b, c : a + b + c
my_function(1, 2, 3)
    # 6

# Square all numbers in the list
squares = list(map(lambda num: num**2, [1,2,3]))
    # [1,4,9]


# Create a list of tuples
tups = list(map(lambda num1, num2: (num1, num2), [1,2,3], ['1','2','3']))
    # [(1, '1'), (2, '2'), (3, '3')]


############################################################### Sort pair ###############################################
data = [(1,5), (2,4), (3,1)]

# Sort by second element
data.sort(key=lambda pair: pair[1])

# Sort by product
data.sort(key=lambda pair: pair[0] * pair[1])
