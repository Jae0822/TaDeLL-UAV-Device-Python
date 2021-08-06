## 预备阶段：1.2_PGELLA_Final~^_^~

### :triangular_flag_on_post:PG-ELLA验证 成功！

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

