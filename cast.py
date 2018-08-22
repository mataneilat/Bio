
import numpy as np
from bio.simple_heap import SimpleHeap

def CAST(affinity_matrix, t):
    (n, m) = affinity_matrix.shape
    if n != m:
        return None
    clusters = []
    elements = [x for x in range(n)]
    C_open = SimpleHeap()
    U_heap = SimpleHeap(elements=elements.copy(), is_max=True)
    while not U_heap.is_empty() or not C_open.is_empty():
        u_affinity, u = U_heap.pop_element()
        if u is not None and u_affinity >= t * len(C_open):
            C_open.add_element(u)
            print("LEN" , len(C_open))
            print("U is: " , u_affinity, u)
            for x_affinity, x in U_heap:
                print(x_affinity)
                U_heap.add_element(x, priority=x_affinity + affinity_matrix[x,u])
            for x_affinity, x in C_open:
                C_open.add_element(x, priority=x_affinity + affinity_matrix[x,u])
        else:
            if u is not None:
                print("ADDING BACK", u, u_affinity)
                U_heap.add_element(u, priority=u_affinity)
            v_affinity, v = C_open.pop_element()
            if v is not None and v_affinity < t * len(C_open):
                print("NOW V")
                U_heap.add_element(v)
                for x_affinity, x in U_heap:
                    print("X: " + x_affinity.__repr__())
                    U_heap.add_element(x, x_affinity - affinity_matrix[x,v])
                for x_affinity, x in C_open:
                    C_open.add_element(x, x_affinity - affinity_matrix[x,v])
            else:
                if v is not None:
                    C_open.add_element(v, v_affinity)
                clusters.append([e for e in C_open.entry_finder])
                C_open = SimpleHeap()
                #print(len(U_heap))
                U_heap.reset_priority()
                print(U_heap.is_empty(), C_open)
    return clusters


def test_cast():
    affinity_matrix = np.matrix([[1, 0.55, 0.1, 0.62, 0.5, 0.7],
                                 [0.55, 1, 0.39, 0.9, 0.35, 0.37],
                                 [0.1, 0.39, 1, 0.66, 0.3, 0.5],
                                 [0.62, 0.9, 0.66, 1, 0.68, 0.75],
                                 [0.5, 0.35, 0.3, 0.68, 1, 0.1],
                                 [0.7, 0.37, 0.5, 0.75, 0.1, 1]])
    print("CLUSTERSSS", CAST(affinity_matrix, 0.5))

if __name__ == '__main__':
    test_cast()




