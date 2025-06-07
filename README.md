# qiqi-radar_2025

### 1. 推荐使用虚拟环境conda配置python环境

### 2. 相对于PFA的改动如下：

1. config.yaml中的改动：
<br>
camera_mode: 增加mv选项，即启动迈德威视相机
<br>
mv_camera: 描述迈德威视相机参数
<br>
blind_zone 中增加<br>
hero_blind_mode: 'base'<br>
enemy_state: 'B'<br>
maxtime: 12<br>
each_guess_time: 6<br>
以及三号四号步兵预测点位(建议删除，因为最后一场跟齐工大的比赛中报过错)


2. 代码中的改动：
<br>
迈德相机驱动
<br>
比赛视频内录
<br>
拓展读取的裁判系统信息量(0x0003, 0x0001)
<br>
英雄预测功能(class TowerBaseHp)
<br>
视频测试功能
<br>
修改双倍易伤触发逻辑
<br>
四号五号步兵的暴力预测(建议删除, 理由同上)

ps: 相比PFA的原生开源，main.py, calibration.py, ser_api.py中均有改动


