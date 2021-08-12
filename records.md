# 1.2_PGELLA_Final~^_^~

### :triangular_flag_on_post:PG-ELLA验证 成功！:gem:

在正式开始将matlab代码转移到python之前，需要先确认现有的python代码的正确性。

以下，将针对以pg-ella算法，进行验证。

```
验证步骤：（完全模仿Bou Ammar的原始matlab代码）
1. 6个随机生成的任务
2. 用pg-regular学习\alpha
3. 用pg-ella学习
4. 比较二者曲线
```

可以看到效果还是不错的。

- 纵轴值在-5～-6之间，原因可能是具体的device的task的设置不同，比如sigma，mean，variance等等。

![Screen Shot 2021-08-05 at 10.29.43 AM.png](https://i.loli.net/2021/08/05/yb97Ce2DUVpLgQl.png)

```
后续工作：
1. 阅读pg-ella类，梳理fixme
2. 找到纵轴值不同的原因，确定合适的设定参数值
```





# 2.0_TaDeLL

本文件用于对TaDeLL函数进行全python转移。

### 2021.08.11

错了。不知道出了什么错误呀。恨别鸟惊心。明天再看吧。

![Screen Shot 2021-08-11 at 6.22.39 PM.png](https://i.loli.net/2021/08/11/Z8k9IdpP7CgVqcM.png)

### 2021.08.12 :gem:

重新生成了新的tasks library

运行成功！

可以用pkl文件，快速绘制结果图。

(k=3的pkl文件同理)

```
# Library generation function: main()-pg_task_generation(30)
# 30 tasks with k=2
# task.policy is the policy learned after 30 niters
with open('TaDeLL_Tasks_lib_k_2.pkl', 'wb') as f:
	pickle.dump(tasks0, f)
```



```
# Learn with TaDeLL:main()-main()
# Use the above generated library, 20 tasks for training, 10 tasks for testing
# 以下有一些变量在运行过程中已经发生了变化。
with open('TaDeLL_result_k_3.pkl', 'wb') as f:
    pickle.dump([means_pg, means_tadell, niter, TaDeLL_Model, tasks0, tasks, testing_tasks, testing_tasks_pg, testing_tasks_TaDeLL], f)
```



**k = 2**

![Screen Shot 2021-08-12 at 3.24.04 PM.png](https://i.loli.net/2021/08/12/KGFaAZl1L24qgiX.png)

**k = 3**

![Screen Shot 2021-08-12 at 4.47.39 PM.png](https://i.loli.net/2021/08/12/AOcHzBGWx3mQ1u9.png)

