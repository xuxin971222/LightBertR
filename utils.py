def get_train_test_data(name, token, p):
    session_path = r'processed/{}/{}/{}_session.csv'.format(name, token, name)
    session_datas = open(session_path, 'r').readlines()

####################################################################
    # get all_item_dict to instead item's id
    all_item_dict = {}
    item_order = 0
    for data in session_datas:
        data = data.split('\n')[0].split(' ')
        item_list = data[1:]
        for item_id in item_list:
            if item_id not in all_item_dict:
                all_item_dict[item_id] = item_order
                item_order += 1

    # get user_session_dict
    user_session_dict = {}
    max_len_session = 0
    for data in session_datas:
        data = data.split('\n')[0].split(' ')
        user_id = data[0]
        user_session_dict[user_id] = [all_item_dict[i] for i in data[1:]]
        if len(data[1:]) > max_len_session:
            max_len_session = len(data[1:])

    item_num = len(all_item_dict)
    session_m = []
    for user_id in user_session_dict:
        session_list = user_session_dict[user_id]
        session_m.append(session_list)

    split_pos = int(len(user_session_dict) * p)
    train_data = session_m[:split_pos]
    test_data = session_m[split_pos:]
#############################################################################

    # all_item_dict = {}
    # for data in session_datas:
    #     data = data.split('\n')[0].split('\t')
    #     neb_item_datas = data[0].split('  ')
    #     user = int(neb_item_datas[0])
    #     nebs = list(eval(neb_item_datas[1]))
    #     all_item_dict[user] = nebs
    # session_m = list(all_item_dict.values())
    # split_pos = int(len(all_item_dict) * p)
    # train_data = session_m[:split_pos]
    # test_data = session_m[split_pos:]
    # item_num = 13044                   #  steam: 13044
    # max_len_session = 12             #   ave_length_steam:12
    # print(len(train_data))
    # print(len(test_data))
    return train_data, test_data, item_num, max_len_session
