import os

txtsavepath = '/disk1/ybh/data' 
flist = [ 'testval']
version = '2014'
for i in flist:
    total_f = os.listdir(txtsavepath + '/' + i + version + '/' + 'images')
    f = open(txtsavepath + '/' + i + '.txt', 'w')
    for j in total_f:
        name = txtsavepath + '/' + i + version + '/' + 'images' + '/' + j + '\n'
        f.write(name)
    f.close()
