############# make session ###########################
# session_datas = open('ml-20m.txt', 'r').readlines()
#
# session_dict = {}
# user = []
# item = []
# for data in session_datas:
#     data = data.split('\n')[0].split(' ')
#     user.append(int(data[0]))
#     item.append(int(data[1]))
#     if data[0] not in session_dict:
#         session_dict[data[0]] = []
#     session_dict[data[0]].append(int(data[1]))
# print('max_user:', max(user))
# print('max_item：', max(item))
# session = list(session_dict.values())
# length = []
# for i in session:
#     length.append(len(i))
# print(max(length))
# with open('processed/ml_20m_visit/random/ml_20m_visit_session.csv', 'w') as f:
#     [f.write('{0}  {1}\n'.format(key, value)) for key, value in session_dict.items()]
########################################################################################
# session_datas = open('ml_20m.csv', 'r').readlines()
# all_item_dict = {}
# for data in session_datas:
#     data = data.split('\n')[0].split('\t')
#     neb_item_datas = data[0].split('  ')
#     user = int(neb_item_datas[0])
#     nebs = list(eval(neb_item_datas[1]))
#     all_item_dict[user] = nebs
# session_m = list(all_item_dict.values())
# x = len(session_m)
# num = 0
# new_session = []
# for i in session_m:
#     num += len(i)
#     if len(i) <= 144:
#         new_session.append(i)
# print(len(new_session))
# print(num/x)
#
# new_session_dict = {}
# for pos, i in enumerate(new_session):
#     new_session_dict[pos] = i
#
# with open('processed/ml_20m_visit/random/ml_20m_visit_session.csv', 'w') as f:
#     [f.write('{0}  {1}\n'.format(key, value)) for key, value in new_session_dict.items()]




############################################################################
# session_datas = open('ml_20m.csv', 'r').readlines()
#
# all_item_dict = {}
# item_order = 0
# for data in session_datas:
#     data = data.split('\n')[0].split(' ')
#     item_list = data[1:]
#     for item_id in item_list:
#         if item_id not in all_item_dict:
#             all_item_dict[item_id] = item_order
#             item_order += 1
#
# user_session_dict = {}
# max_len_session = 0
# for data in session_datas:
#     data = data.split('\n')[0].split(' ')
#     user_id = data[0]
#     user_session_dict[user_id] = [all_item_dict[i] for i in data[1:]]
#     if len(data[1:]) > max_len_session:
#         max_len_session = len(data[1:])
#
# item_num = len(all_item_dict)
# print(item_num)
# session_m = []
# for user_id in user_session_dict:
#     session_list = user_session_dict[user_id]
#     session_m.append(session_list)
#
# num = 0
# x = len(session_m)
# new_session = []
# for i in session_m:
#     num += len(i)
# print('length_session_m:', x)
# print('average_length:', num/x)
# for i in session_m:
#     num += len(i)
#     if len(i) <= 8:
#         new_session.append(i)
# new_session_dict = {}
# for pos, i in enumerate(new_session):
#     new_session_dict[pos] = i
#
# with open('toys.csv', 'w') as f:
#     [f.write('{0}  {1}\n'.format(key, value)) for key, value in new_session_dict.items()]
#
###########################################################################################
# # coding=utf-8
#
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MultipleLocator
#
# plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：SimHei
# plt.rcParams['axes.unicode_minus'] = False  # 显示负号
#
# x = np.array([0.5,1,1.5,2])
# #Beauty
# # VGG_supervised = np.array([0.0698, 0.1008, 0.1116, 0.0983])
# # VGG_supervised = np.array([0.3151, 0.3165, 0.3182, 0.2949]) #HR@10
# # VGG_supervised = np.array([0.1689, 0.1838, 0.1882, 0.1750]) #NDCG@10
#
# #steam
# # VGG_supervised = np.array([0.1095, 0.1137, 0.1194, 0.1192]) #HR@1
# # VGG_supervised = np.array([0.3784, 0.3851, 0.4026, 0.3912])  #HR@10
# VGG_supervised = np.array([0.1974, 0.2033, 0.2257, 0.2141]) #NDCG@10
#
# # label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
# # color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
# # 线型：-  --   -.  :    ,
# # marker：.  ,   o   v    <    *    +    1
# plt.figure(figsize=(7, 5))
# plt.grid(linestyle="--")  # 设置背景网格线为虚线
# ax = plt.gca()
# ax.spines['top'].set_visible(False)  # 去掉上边框
# ax.spines['right'].set_visible(False)  # 去掉右边框
#
#
# # plt.plot(x, VGG_supervised, marker='*', mec='r', mfc='w', color="green", label="BertR", linewidth=2, markersize=8)
# plt.plot(x, VGG_supervised, marker='o', mec='r', mfc='w', color="orange", label="BertR", linewidth=2, markersize=8)
#
#
# group_labels = ['32', '64', '128', '256']  # x轴刻度的标识
# plt.xticks(x, group_labels, fontsize=16, fontweight='bold')  # 默认字体大小为10
# plt.yticks(fontsize=16, fontweight='bold')
# # plt.title("example", fontsize=12, fontweight='bold')  # 默认字体大小为12
# # plt.xlabel("Performance Percentile", fontsize=13, fontweight='bold')
# # plt.ylabel("HR@10", fontsize=16, fontweight='bold')
# plt.ylabel("NDCG@10", fontsize=16, fontweight='bold')
# plt.xlabel("Dimensionality", fontsize=16, fontweight='bold')
# plt.title("steam", fontsize=16, fontweight='bold')
# # plt.title("steam", fontsize=16, fontweight='bold')
# plt.xlim(0.4, 2.1)  # 设置x轴的范围
# y_major_locator=MultipleLocator(0.01)
# ax.yaxis.set_major_locator(y_major_locator)
# # plt.ylim(0.060, 0.120)
# # plt.ylim(0.105, 0.120)
# # plt.ylim(0.2800, 0.3200)
# # plt.ylim(0.370, 0.410)
# plt.ylim(0.190, 0.230)
# # plt.ylim(0.160, 0.190)
# # plt.legend()          #显示各曲线的图例
# plt.legend()
# leg = plt.gca().get_legend()
# ltext = leg.get_texts()
# plt.setp(ltext, fontsize=16, fontweight='bold')  # 设置图例字体的大小和粗细
#
# plt.savefig('./filename.svg', format='svg')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
# plt.show()
#############################################################################################################################


# class Solution():
#     def solution(self, s):
#         tab = {'a': 1, 'b': 2, 'c': 4, 'x': 8, 'y': 16, 'z': 32}
#         now = 0
#         maxlen = 0
#         before = {0: -1}
#         for i in range(len(s)):
#             now = now if s[i] not in tab.keys() else now ^ tab[s[i]]
#             if now not in before.keys():
#                 before[now] = i
#             maxlen = max(maxlen, i - before[now])
#             print(maxlen)
#         return maxlen
# ss = Solution()
# s = 'efamacdcbx'
# res = ss.solution(s)
# print(res)

####################################################################################
# if __name__ == '__main__':
#     def dfs(x, graph_dict, visited, left):
#         b_members = graph_dict[x]
#         for b_m in b_members:  # 同一次 增广路寻找中，若v曾经被达到过，则跳过。
#             if visited[b_m] == 0:  # 若x能到达的 右部结点 b_m 为非匹配点，则找到了一条长度为1的增广路，记：left[b_m] = x
#                 visited[b_m] = 1
#                 if b_m not in left.keys():
#                     left[b_m] = x
#                     return True
#                 else:
#                     # 若 b_m 为匹配点，则递归的从left[b_m]出发寻找增广路，回溯时记：left[b_m] = x
#                     dfs(left[b_m], graph_dict, visited, left)
#                     left[b_m] = x
#                     return True
#         return False
#
#
#     # a_company = list(map(int, input().split(' ')))
#     # b_company = list(map(int, input().split(' ')))
#     # n = int(input())
#     # projects = []
#     # for _ in range(n):
#     #     temp = list(map(int, input().split(' ')))
#     #     projects.append(temp)
#     a_company = [0, 1, 2]
#     b_company = [3, 4, 5]
#     projects = [[0, 4], [0, 3], [1, 3], [1, 4], [2, 5], [2, 4]]
#     graph_dict = {}
#     for p in projects:
#         if p[0] not in graph_dict.keys():
#             graph_dict[p[0]] = []
#         graph_dict[p[0]].append(p[1])
#     # 根据建立的二分图，寻找 最大匹配数 = 最小点覆盖数
#     visited = {}  # 记录右部节点是否被匹配过
#     for b_m in b_company:
#         visited[b_m] = 0
#     left = {}  # 匹配右部i点的左部节点
#     ans = 0
#     for a_m in a_company:  # 从a公司的任一节点出发，依次寻找增广路，并查找返回结果
#         if dfs(a_m, graph_dict, visited, left):
#             ans += 1
#     print(ans)

#########################################################
# n, m = 3, 3
# num = 0
# num_water = 0
# p = ['**.', '.*.', '***']
# for i in p:
#     x = p.split()
#     for j in range(m):
#         if x[j] == '*':
#             num += 1
#         if x[j] == '.':
#             num_water += 1
# print(num, num_water)
##########################链表##############################
l1 = [2, 4, 3]
def Ltable(l1):
    head = ListNode(l1[0])
    cur = head
    for i in range(1, len(l1)):
        tem = ListNode(l1[i])
        cur.next = tem
        cur = tem
    return head
class ListNode:
   def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def traverse(head):
    if head != None:
        print(head.val)
        traverse(head.next)

head = Ltable(l1)
traverse(head)
#################################################



















