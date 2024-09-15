import time
from concurrent.futures import ThreadPoolExecutor

# 要在线程中运行的函数
def func(num):
    time.sleep(num)
    return num * num


# 主程序
def main():
    # 创建字典用于存储返回值
    return_values = {}

    # 创建线程列表
    threads = []

    # 创建线程

    pool = ThreadPoolExecutor(max_workers=3)
    future1 = pool.submit(func, 3)
    future2 = pool.submit(func, 8)
    # for i in range(5):
    #     t = threading.Thread(target=thread_function, args=(i, return_values))
    #     threads.append(t)

    # 启动线程
    for t in threads:
        t.start()

    # 等待所有线程完成
    for t in threads:
        t.join()

    # 打印返回的结果
    print(return_values)


if __name__ == "__main__":
    main()