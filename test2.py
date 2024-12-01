import random

r = random.randint(1, 100)

u = int(input())

count = 1

while u != r:
    count += 1
    if r > u:
        print('もっと大きいです！')
    else:
        print('もっと小さいです！')
    u = int(input())

print(f'正解です！{count}回で正解しました！')