
list1=[[75,	85,	85,	85,	80,	72],
      [88,	68,	88,	88,	85,	92],
      [82,	58,	72,	72,	78,	86],
      [90,	78,	82,	58,	65,	68],
      [56,	82,	78,	82,	75,	82]]


list = [[] for _ in range(len(list1)) ]

for i in range(len(list1)):
    for j in range(len(list1[i])):
        list[i].append(list1[i][j]/sum(list1[i]))
print(list)

#本文方法
wb=[0.199,0.218,0.210,0.155,0.116,0.102]
wb1=[]
for i in range(len(list)):
    wb_son=0
    for j in range(len(list[i])):
        wb_son=wb_son+wb[j]*list[i][j]
    wb1.append(wb_son)

wa=[0.199,0.214,0.203,0.159,0.122,0.103]

wa1=[]
for i in range(len(list)):
    wa_son=0
    for j in range(len(list[i])):
        wa_son=wa_son+wa[j]*list[i][j]
    wa1.append(wa_son)

wc=[0.129,0.238,0.250,0.156,0.100,0.127]

wc1=[]
for i in range(len(list)):
    wc_son=0
    for j in range(len(list[i])):
        wc_son=wc_son+wc[j]*list[i][j]
    wc1.append(wc_son)


#令三篇方法结果
wb2=[1.79,2.4,2.08,1.61,1.45,1.52]
# wb=[ i/sum(wb2) for i in wb2]
# wb1 = []
# for i in range(len(list)):
#     wb_son = 0
#     for j in range(len(list[i])):
#         wb_son = wb_son + wb[j] * list[i][j]
#     wb1.append(wb_son)
#
# wa = [0.112,0.279,0.250,0.144,0.093,0.122]
#
# wa1 = []
# for i in range(len(list)):
#     wa_son = 0
#     for j in range(len(list[i])):
#         wa_son = wa_son + wa[j] * list[i][j]
#     wa1.append(wa_son)
#
# wc = [0.1208,0.2708,0.2519,0.1472,0.0919,0.1174]
#
# wc1 = []
# for i in range(len(list)):
#     wc_son = 0
#     for j in range(len(list[i])):
#         wc_son = wc_son + wc[j] * list[i][j]
#     wc1.append(wc_son)

print(wb1)
print(wa1)
print(wc1)