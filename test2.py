import random

def generate_valid_fraction(a, ranges):
    results = []
    for x, y in ranges:
        while True:
            b = random.randint(0, a)
            fraction = b / a
            if x <= fraction < y:
                results.append(fraction)
                break

    return results

# 示例使用
# case 1:
d = 100
s = 10
a = d * d - s * s
starts = [0.75, 0.59, 0.92, 0.79, 0.03, 0.07, 0.12]
rags = []
for start in starts:
    rags.append((start, start + 0.005))
res = generate_valid_fraction(a, rags)
print("Generated values:", res)

# # case 2:
# s = 10
# a = s * s
# starts = [0.98, 0.95]
# rags = []
# for start in starts:
#     rags.append((start, start + 0.009))
# res = generate_valid_fraction(a, rags)
# print("Generated values:", res)
