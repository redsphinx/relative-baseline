'''
https://arxiv.org/pdf/1711.05847.pdf
'''

by_first = True

# init
n = 3
or_nodes = [['o', 0, 0], ['o', 0, 1], ['o', 0, 2]]
v = [['o', 0, 0], ['o', 0, 1], ['o', 0, 2]]
q = [['o', 0, 0], ['o', 0, 1], ['o', 0, 2]]
e = []
t_nodes = []
and_nodes = []

if by_first:
    q.reverse()

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
        for m in range(k):
            a_i_j_m = ['a', v_i_j[1], v_i_j[2], m]
            and_nodes.append(a_i_j_m)
            e_v_a = [v_i_j, a_i_j_m]
            if e_v_a not in e:
                if [a_i_j_m, v_i_j] not in e:
                    e.append(e_v_a)
            if a_i_j_m not in v:
                v.append(a_i_j_m)
                if by_first:
                    q.insert(0, a_i_j_m)
                else:
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
            if by_first:
                q.insert(0, o_i_im)
            else:
                q.append(o_i_im)
        if o_im1_j not in v:
            v.append(o_im1_j)
            if by_first:
                q.insert(0, o_im1_j)
            else:
                q.append(o_im1_j)


print('vertices: ')
for i in range(len(v)):
    print(v[i])

print('edges: ')
for i in range(len(e)):
    print(e[i])
