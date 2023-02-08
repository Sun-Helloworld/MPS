#mps错误测试
#在mps开启的情况下使用一个GPU执行两个相同的程序，在开始执行一段时间后中断其中一个，等待一段时间看另一个是否会中断
#没有中断则等待短暂时间后手动停止，然后重复执行，有产生中断则整个循环退出，或者执行一定数量此操作后程序退出


#!/bin/bash
count=0
for i in {1..1000}
  do
    
    nohup python mps_train.py > mps_train.log 2>&1 &
    nohup python train_mps.py > train_mps.log 2>&1 &

    first=$(ps -ef | grep "mps_train.py" | grep -v grep | awk '{print $2}')
    #需要等待到使用GPU的阶段
    sleep 5s
    # kill -9 $first
    kill -2 $first  #类似ctrl+c的功能
    #等待几秒钟看另外一个进程是否会出错结束
    sleep 5s
    #获得进程ID
    second=$(ps -ef | grep "train_mps.py" | grep -v grep | awk '{print $2}')
    #获得进程状态
    status=$(ps -aux | grep "train_mps.py" | grep -v grep | awk '{print $8}')
    # status2=$(ps -aux | grep "train_mps.py" | grep -v grep | awk '{print $8}')
    # echo $status
    if [[ $status =~ "Rl" ]]; then
        # kill -9 $second
        kill -9 $first
        kill -9 $second  
        echo "第"$i"次"
    else
        count=$(($count+1)) 
        echo "产生错误"$status"#################################################################"
        kill -9 $first
        kill -9 $second 
    fi
    
  done
  echo $count


