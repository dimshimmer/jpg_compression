# README

该脚本实现了将图片压缩为`.jpg`文件的功能，并在其中添加隐写信息

`.jpg`文件的基本格式为：

```assembly
SOI + APP0 + DQT + SOF0 + DHT + (DRI) + SOS + EOI

start of image: 0xFF,0xD8
APP0: 图像识别信息
DQT: 定义量化表
SOF0: 图像基本信息
DHT: 定义Huffman表
DRI: 定义重新开始间隔
SOS: 扫描行开始
end of image:0xFF,0xD9

The format of a segment:
段标识  0xFF
段类型  xx
段长度  2 byte, big ending
```



```python
tag_map = {
    "ffd8": "SOI",
    "ffc4": "DHT",
    'ffc8': "JPG",
    'ffcc': "DAC",
    "ffda": "SOS",
    "ffdb": "DQT",
    "ffdc": "DNL",
    "ffdd": "DRI",
    "ffde": "DHP",
    "ffdf": "EXP",
    "fffe": "COM",
    "ff01": "TEM",
    "ffd9": "EOI"
}
```

APP0

```assembly
--------------------------------------------------------------------------
名称      字节数     值                   说明
--------------------------------------------------------------------------
段标识       1         FF
段类型       1         E0
段长度       2         0010                    如果有RGB缩略图就＝16＋3n
　　（以下为段内容）
交换格式      5         4A46494600          “JFIF”的ASCII码
主版本号      1
次版本号      1  
密度单位      1         0＝无单位；1＝点数/英寸；2＝点数/厘米
X像素密度     2                             水平方向的密度   
Y像素密度     2                             垂直方向的密度
缩略图X像素  1                           缩略图水平像素数目  
缩略图Y像素  1                           缩略图垂直像素数目
（如果“缩略图X像素”和“缩略图Y像素”的值均＞0，那么才有下面的数据）
RGB缩略图  3×n         n＝缩略图像素总数＝缩略图X像素×缩略图Y像素
--------------------------------------------------------------------------
```

DQT

```assembly
--------------------------------------------------------------------------
名称  字节数 值       说明
--------------------------------------------------------------------------
段标识   1     FF
段类型   1     DB
段长度   2     43      其值＝3＋n（当只有一个QT时）
（以下为段内容）
QT信息  1     0－3位：QT号
4－7位：QT精度（0＝8bit，1字节；否则＝16bit，2字节）
QT        n             n＝64×QT精度的字节数
--------------------------------------------------------------------------
```

SOF0

```assembly
--------------------------------------------------------------------------
名称  字节数 值       说明
--------------------------------------------------------------------------
段标识   1     FF
段类型   1     C0
段长度   2             其值＝8＋组件数量×3
　　（以下为段内容）
样本精度 1      8       每个样本位数（大多数软件不支持12和16）
图片高度 2
图片宽度 2
组件数量 1      3       1＝灰度图，3＝YCbCr/YIQ 彩色图，4＝CMYK 彩色图
　　（以下每个组件占用３字节）
组件 ID     1             1＝Y, 2＝Cb, 3＝Cr, 4＝I, 5＝Q
采样系数 1              0－3位：垂直采样系数
                        4－7位：水平采样系数
量化表号 1
---------------------------------------------------------------------------
```

DHT

```assembly
--------------------------------------------------------------------------
名称  字节数 值       说明
--------------------------------------------------------------------------
段标识   1     FF
段类型   1     C4
段长度   2             其值＝19＋n（当只有一个HT表时）
　　（以下为段内容）
HT信息  1             0－3位：HT号
                                4位：   HT类型, 0＝DC表，1＝AC表
　　　　　　　　　　  5－7位：必须＝0
HT位表  16            这16个数的和应该≤256
HT值表  n             n＝表头16个数的和
--------------------------------------------------------------------------
```

SOS

```assembly
--------------------------------------------------------------------------
名称          字节数 值       说明
--------------------------------------------------------------------------
段标识            1    FF
段类型            1    DA
段长度            2    000C    其值＝6＋2×扫描行内组件数量
　　（以下为段内容）
扫描行内组件数量 1  3       必须≥1，≤4（否则错误），通常＝3
　　（以下每个组件占用２字节）
组件ID               1    1 = Y, 2 = Cb, 3 = Cr, 4 = I, 5 = Q
Huffman表号      1    0－3位：AC表号 (其值＝0...3)
                                    4－7位：DC表号(其值＝0...3)
剩余3个字节          3           最后３个字节用途不明，忽略
--------------------------------------------------------------------------
```

```assembly
JPEG压缩编码实例
DC是指直流系数，是8×8个像素的平均值；AC是交流系数，是8×8个像素的其它值。压缩数据的排列方式是：亮度DC，AC，色差DC，AC，色差DC，AC。

1、每个分量如Y分量（DC+AC）完成后，如果还剩下位数，应该舍弃，后面的Cb分量是从下一个字节重新计算。

2、如果编码到后面没有压缩数据了，后面实际编码数据用0填充。

3、如果编码已经完成，那么剩余压缩数据用1填充。
```

