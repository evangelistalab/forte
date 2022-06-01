def find_ordering(ints,wfn):
    nmo = ints.nmo()
    print(f'Localizing {nmo} orbitals')
    L = np.zeros((nmo,nmo))
    mos = list(range(nmo))
    print(mos)
    tei = ints.tei_ab_block(mos,mos,mos,mos)
    for i in range(nmo):
        for j in range(nmo):
            if i != j:
                L[i,j] += -tei[i,j,j,i]
        for j in range(nmo):
            L[i,i] += np.abs(tei[i,j,j,i])

    L1 = np.zeros((nmo,nmo))
    oei = ints.oei_a_block(mos,mos)
    for i in range(nmo):
        for j in range(nmo):
            if i != j:
                L1[i,j] += -oei[i,j]
        for j in range(nmo):
            L1[i,i] += np.abs(oei[i,j])

    print(f'L = {L}')
    A,B = np.linalg.eigh(L)
    print(f'A = {A}')
    print(f'B = {B}')

    fiedler_eval = A[1]
    deg = 1
    for a in A[2:]:
        if np.abs(fiedler_eval - a) < 1.0e-6:
           deg += 1

    print(f'Degeneracy of the second Fiedler eigenvalue {deg}')
    x = B[:,1]
    order = []
    for i in range(nmo):
        order.append((x[i],i))
    order = sorted(order)
    print(order)
    U = np.zeros((nmo,nmo))
    k = 0
    for _,i in order:
        print(f'MO{k + 1:4d} -> {i + 1:4d}')
        U[i,k] = 1.0
        k += 1

    L2 = np.dot(B.transpose(),np.dot(L1,B))
    with np.printoptions(precision=3, suppress=True):
        print(L2)

    L2p = L2[1:4,1:4]
    print(L2p)
    A1,B1 = np.linalg.eigh(L2p)
    print(f'A1 = {A1}')
    print(f'B1 = {B1}')

    Deg = B[:,1:1+deg]
    NonDeg = np.dot(Deg,B1)

    order = []
    for i in range(nmo):
        order.append((NonDeg[i,2],i))
    order = sorted(order)
    print(order)
    U = np.zeros((nmo,nmo))
    k = 0
    for _,i in order:
        print(f'MO{k + 1:4d} -> {i + 1:4d}')
        U[i,k] = 1.0
        k += 1

    Ca = wfn.Ca().np
    Cap = np.dot(Ca,U)
    wfn.Ca().np[:]= psi4.core.Matrix.from_array(Cap)
