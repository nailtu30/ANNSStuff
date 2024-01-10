# The Right way of Computing ANNS (Approximate Nearest Neighbor Search) TopK Ground Truth

When I was doing the experiment, I found that the Top-1 (I. e. K = 1) result obtained by fais `IndexFlatL2` was used as Ground Truth to test `IndexIVFflat`, `IndexIVFflat` could never reach 100% `recall-at-1`. Even `nprobe == nlist`, `recall-at-1` could not get 100%.   

I found that the problem is the Ground Truth of `sift1M` or `gist1M` or `deep1B` datasets. They have N base vectors with the same distance to a query vector, and the Top-1 vector will return a random Index among them. Therefore, there is a situation where the distance is the same but the Index is different, resulting in the structure not matching.   

Therefore, the following code finds all base vectors of the same distance for a query vector as TopK.   
For example, for query vector q1, base vector b1,b2,b3, if `Dis(q1,b1)==Dis(q1,b2)<Dis(q1,b3)`, the Top-1 Ground Truth file includes the index of b1 and b2.  

Computing Ground Truth:
```python
def compute_gt(xb, xq, k):
    nb, d = xb.shape
    index = faiss.IndexFlatL2(d)
    index.add(xb)
    nq, d = xq.shape

    new_I = [[] for _ in range(nq)]
    for nq_idx in range(nq):
        print('========handle {} nq========'.format(nq_idx))
        D, I = index.search(xq[nq_idx].reshape((1, d)), nb)
        k_idx = 0
        nb_idx = 0
        while k_idx < k and nb_idx < nb:
            k_idx = k_idx + 1
            D_min = D[0][nb_idx]
            new_I[nq_idx].append(I[0][nb_idx])
            nb_idx = nb_idx + 1
            while nb_idx < nb:
                if D[0][nb_idx] == D_min:
                    new_I[nq_idx].append(I[0][nb_idx])
                    nb_idx = nb_idx + 1
                else:
                    break

    return new_I
```   

Save Ground Truth:
```python
def compute_groundtruth(DB_DIR, data_name, k, gt_file_name):
    if data_name == 'sift1M':
        xb, xq, xt = load_sift1M(DB_DIR, compute_gt=True)
    elif data_name == 'gist1M':
        xb, xq, xt = load_gist1M(DB_DIR, compute_gt=True)
    elif data_name == 'deep10M':
        xb, xq, xt = load_deep10M(DB_DIR, compute_gt=True)
    new_I = compute_gt(xb, xq, k)
    with open(gt_file_name, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for row in new_I:
            writer.writerow(row)
```

Load Ground Truth:
```python
    gt = []
    with open(os.path.join(DB_DIR, gt_file_name), 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            row = list(map(int, row))
            gt.append(row)
```

Computing Recall:
```python
def compute_recall(result, ground, ncandidate):
    nq = result.shape[0]
    count = 0
    missed_query = []
    for nq_idx in range(nq):
        flag = False
        for res in result[nq_idx, :ncandidate]:
            if res in ground[nq_idx]:
                count = count + 1
                flag = True
                break
        if not flag:
            missed_query.append(nq_idx)
    recall = count / (ncandidate * nq)
    print(missed_query)
    return recall
```