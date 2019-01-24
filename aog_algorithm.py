'''
https://arxiv.org/pdf/1711.05847.pdf
'''

# init
o_0, o_1, o_2 = ['o', 0, 0], ['o', 0, 1], ['o', 0, 2]
or_nodes = [['o', 0, 0], ['o', 0, 1], ['o', 0, 2]]
v = [['o', 0, 0], ['o', 0, 1], ['o', 0, 2]]
q = [['o', 0, 0], ['o', 0, 1], ['o', 0, 2]]
e = []
t_nodes = []
and_nodes = []


while(len(q) is not 0):
    v_i_j = q.pop()
    k = v_i_j[2] - v_i_j[1] + 1

    # if v_i_j is an OR-node
    if v_i_j[0] == 'o':
        # i) add a terminal-node
        t_i_j = ['t', v_i_j[1], v_i_j[2]]
        t_nodes.append(t_i_j)
        v.append(t_i_j)
        e_v_t = [v_i_j, t_i_j]
        e.append(e_v_t)
        # ii) create AND-nodes for valid splits 0<=m<k
        for m in range(k-1):
            a_i_j_m = ['a', v_i_j[1], v_i_j[2], m]
            and_nodes.append(a_i_j_m)
            q.append(a_i_j_m)

    # if v_i_j is an AND-node with split m
    elif v_i_j[0] == 'a':
        m = v_i_j[-1]
        # create 2 OR-nodes
        o_i_im = ['o', v_i_j[1], v_i_j[1]+m]
        or_nodes.append(o_i_im)
        o_im1_j = ['o', v_i_j[1]+m+1, v_i_j[2]]
        or_nodes.append(o_im1_j)
        e_v_o = [v_i_j, o_i_im]
        e.append(e_v_o)
        e_v_o = [v_i_j, o_im1_j]
        e.append(e_v_o)
        if o_i_im not in v:
            v.append(o_i_im)
            q.append(o_i_im)
        if o_im1_j not in v:
            v.append(o_im1_j)
            q.append(o_im1_j)


print('vertices: ', v)
print('or-nodes: ', or_nodes)
print('t-nodes: ', t_nodes)
print('and-nodes: ', and_nodes)
print('edges: ', e)
