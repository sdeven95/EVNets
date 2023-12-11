"""
time 包

time() 返回当前时间秒数，不精确
prf_counter() 返回当前时间描述，较为精确
process_timer() 调用系统功能返回进程使用的时间
timeit() 可以统计小段代码的执行时间，例子：timeit.timeit('"-".join(str(n) for n in range(100))', number=10000)

已经存在精确到纳秒的方法
time.perf_counter_ns()
time.process_time_ns()
time.time_ns()

"""
