- **MPS相关情况介绍**

  1.允许来自不同进程的内核和内存复制操作在 GPU 上重叠，从而实现更高的利用率和更短的运行时间，

  2.MPS 服务器在其所有客户端之间共享一组调度资源，从而消除了 GPU 在这些客户端之间进行调度时的交换开销。

  3.MPS 服务器分配一份由其所有客户端共享的 GPU 存储和调度资源。
  ​	当每个应用程序进程未生成足够的工作以使 GPU 饱和时，MPS 很有用。如果应用程序由于每个网格的线程数较少而显示出较低的 GPU 占用率，则可以使用 MPS 实现性能改进。
  ​	MPS 控制守护程序会将来自不同用户的 MPS 服务器激活请求排队，从而导致用户之间对 GPU 的序列化独占访问，而不管 GPU 独占性设置如何。

  - **新版本情况**

    ![img](https://api2.mubu.com/v3/document_image/30829030-3a13-45b9-b2c2-b288b3656982-15661181.jpg)

    ​	原本的MPS 服务器分配一份由其所有客户端共享的 GPU 存储和调度资源。新版本的Volta MPS 支持增加 MPS 客户端之间的隔离，因此资源减少的程度要小得多。

    - 1、Volta MPS 客户端直接向 GPU 提交工作，无需通过 MPS 服务器。2、每个 Volta MPS 客户端都拥有自己的 GPU 地址空间，而不是与所有其他 MPS 客户端共享 GPU 地址空间/3、Volta MPS 支持为服务质量 (QoS) 提供有限的执行资源。


    - **节点情况：**可产生48个节点，如果超过连接限制，CUDA 应用程序将无法创建 CUDA 上下文并从 cuCtxCreate() 或触发上下文创建的第一个 CUDA 运行时 API 调用返回 API 错误。MPS 服务器将记录失败的连接尝试。（原来的情况是创建超过16个之后，导致启动延迟增加，并且通常会减慢客户端-服务器通信，这是由于 Linux CFS（完全公平调度程序）调度线程的方式。）


    - **停止方式**：正确的停止方式terminate_client <server PID> <client PID>；通过 CTRL-C 或信号终止 MPS 客户端，这将导致未定义的行为。可能会使 MPS 服务器处于未定义状态并导致意外故障、损坏或挂起。


    - **调度**
      - taskset –c 0 nvidia-cuda-mps-control –d  ，在有多个GPU和多个MPS控制守护进程一起使用的时候，把不同的守护进程绑定到不同的cpu上，可以通过MPS继承关联处理器，增加访问速度。（局部性原理）


    - **内存保护方面**

      - （写入）CUDA 内核中的越界写入可以修改另一个进程的 CUDA 可访问内存状态，并且不会触发错误。


      - （读取）CUDA 内核中的超出范围读取可以访问由另一个进程修改的 CUDA 可访问内存，并且不会触发错误，从而导致未定义的行为。


    - **错误隔离方面**

      （原来的情况是1、任何客户端产生的 GPU 异常都会报告给所有客户端，但不会指明是哪个客户端产生了错误。
      2、由一个客户端触发的致命 GPU 异常将终止 MPS 服务器和所有客户端的 GPU 活动。
      ）

      - **现在的情况**客户端进程生成的致命 GPU 异常  将报告给在包含致命异常的 GPU 子集上运行的所有客户端，而不指示哪个客户端生成了错误。请注意，受影响的客户端有责任在收到致命 GPU 异常通知后退出。其他 GPU 上运行的客户端不受致命异常的影响，并将正常运行直到完成。![img](https://api2.mubu.com/v3/document_image/bbac1a72-a17a-429f-8943-94b0641a1f9e-15661181.jpg)


      - 同一mps服务器（同一个GPU上）


  - **抢占问题和分片策略**

    不保留专用资源，只是限制使用的资源，所以由于负载均衡的原因还是可能会有不同客户端的提交在同一个SM上运行

    - 支持有限的执行资源配置，目的是1、通常不需要让每个客户端都可以访问所有线程，因此分配完整的上下文存储是一种浪费。减少可用线程数将有效地减少上下文存储分配大小。2、配置机制可以用作经典的 QoS 机制来限制可用的计算带宽。减少可用线程的部分也会将一个客户提交的工作集中到一组 SM，减少对其他客户提交工作的破坏性干扰。


    - **活动线程百分比**（向下舍入到最接近的硬件支持的线程计数限制）

      - **均匀分区**

        - 一种常见的配置策略是将可用线程均匀地分配给每个 MPS 客户端进程（即，将活动线程百分比设置为 100%/n，对于 n 个预期的 MPS 客户端进程）。此策略将分配接近最小数量的执行资源，但它可能会限制偶尔使用空闲资源的客户端的性能。


        - 更优化的策略是将部分均匀划分为预期客户端数量的一半（即设置活动线程百分比为 100% / 0.5n），以便在有空闲资源时让负载均衡器更自由地在客户端之间重叠执行。


      - **非均匀分区**

        - 接近最佳的供应策略是根据每个 MPS 客户端的工作负载对可用线程进行非均匀分区（即，如果客户端 1 的比率为客户端 1，则将活动线程百分比设置为 30%，将客户端 2 的活动线程百分比设置为 70%工作量和 client2 工作量为 30%: 70%)。该策略将把不同客户提交的工作集中到不相交的 SM 集合中，并有效地减少不同客户提交的工作之间的干扰。


        - 最佳供应策略是在知道每个客户端的执行资源需求的情况下，精确地限制每个 MPS 客户端使用的 SM 数量（即，在具有 84 个 SM 的设备上，客户端 1 有 24 个 SM，客户端 2 有 60 个 SM）。与活动线程百分比相比，此策略对工作将在其上运行的 SM 集提供了更细粒度和更灵活的控制。


    - **编程接口**（内部四舍五入为最接近的硬件支持的 SM 计数限制）
      -  cuCtxCreate_v3() 创建的客户端 CUDA 上下文配置的CUexecAffinity参数它指定上下文限制使用的 SM 数量。上下文的执行限制可以通过查询cuCtxGetExecAffinity().


- 实验一：nbody（./nbody -benchmark -numbodies=512000）

  - 未开启MPS

    - 只运行单个程序（5387）![img](https://api2.mubu.com/v3/document_image/8792613b-5b7a-4d9e-840c-980d89abb741-15661181.jpg)


    - 同时运行两个程序（11629-11637）![img](https://api2.mubu.com/v3/document_image/8326f6fe-fe2c-42bc-84d9-01caff381475-15661181.jpg)


    - 同时运行四个程序（23054-23250）![img](https://api2.mubu.com/v3/document_image/bf705cfb-32c6-48ba-a9cf-4d893fd32fa3-15661181.jpg)


  - 开启MPS（开启之后可以看到后台后Mps守护进程，进程变成M+C进程）

    - 只运行一个程序（5404）![img](https://api2.mubu.com/v3/document_image/2d001819-fc11-4953-b3de-615762cb81a6-15661181.jpg)


    - 同时运行两个（9122-9151，提升了21%）![img](https://api2.mubu.com/v3/document_image/3444dac7-115c-4ffc-a91f-041bf8a08901-15661181.jpg)![img](https://api2.mubu.com/v3/document_image/ba0c64b3-8778-48f0-8a21-e1bfcaa5ae7b-15661181.jpg)
      显卡状态


    - 同时运行四个（时间18395-18438，提升了20%）![img](https://api2.mubu.com/v3/document_image/d957fafe-19c5-4631-9ce9-a667392bf386-15661181.jpg)![img](https://api2.mubu.com/v3/document_image/b63f39a7-9ab5-44eb-a01f-b011c1d04c0b-15661181.jpg)![img](https://api2.mubu.com/v3/document_image/020acdf7-ed46-44ad-adc8-94bffa5967dc-15661181.jpg)![img](https://api2.mubu.com/v3/document_image/aa193699-b6ba-4bd0-a2a2-9271b3407cd9-15661181.jpg)
      GPU利用率这一块也有提高，不开启MPS就会平均分配，各占25%利用率，开启后同时运行就会有90以上利用率，但时间提高远没有达到90%


    - 使用不同的停止方式停止MPS客户端


- 实验二：LeNet训练cifar-10

  - 单独跑一次![img](https://api2.mubu.com/v3/document_image/97b23d93-88fc-4247-89f1-6fcfb8d5c6dc-15661181.jpg)![img](https://api2.mubu.com/v3/document_image/0075db1e-fea6-499e-9835-2aebc5575ff8-15661181.jpg)


  - 不开启MPS运行四个![img](https://api2.mubu.com/v3/document_image/d8470649-a87c-4c5f-97dc-e09e46abba5a-15661181.jpg)![img](https://api2.mubu.com/v3/document_image/b98dc53e-bd29-49cc-a85e-3b6189ab99ee-15661181.jpg)![img](https://api2.mubu.com/v3/document_image/640546e3-c0fe-41b7-b174-075057a42ede-15661181.jpg)


  - 开启MPS运行四个![img](https://api2.mubu.com/v3/document_image/898bf424-2f37-4286-9ebd-ed0291ed6e49-15661181.jpg)![img](https://api2.mubu.com/v3/document_image/29cc2d5c-b314-446b-b483-56c60007b97a-15661181.jpg)
    可以看到时间上并没有提高很多，但计算的利用率提升了不少。

- 实验三：错误隔离情况（手动停止，通过命令停止和其他错误停止）

  - 多次测试效果

    - 一百次测试，产生八次错误错误类型都为Sl+![img](https://api2.mubu.com/v3/document_image/af9f5b76-568e-4725-b724-958f63c7bfe2-15661181.jpg)


    - 一千次测试，产生67次错误，错误类型都为Sl+![img](https://api2.mubu.com/v3/document_image/ce596015-1bf0-481b-bacd-7973969f72ae-15661181.jpg)





