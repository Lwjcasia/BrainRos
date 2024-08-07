import serial
 
ser = serial.Serial('COM3', 9600, timeout=1)  # 根据实际情况修改串口名称和波特率
file_path = r"data.txt"# 更改为保存数据的文件路径
file = open(file_path, "w")
 
while True:
    # 读取串口数据
    data = ser.readline()
    print(data)
    if data:
        # 将字节数据转换为字符串并打印
        # print(data)
        # print(int(data.decode('utf-8')))
        file.write(data.decode('utf-8'))
        file.flush()