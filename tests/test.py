import sys

# 将 stderr 指向一个文件
sys.stderr = open('global_error.log', 'a', encoding='utf-8')

# 以下代码产生的任何未捕获报错都会自动进入 global_error.log
print(1 / 0)