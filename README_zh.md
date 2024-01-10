# 近似最近邻搜索 TopK Ground Truth计算方式
我在做实验时发现，用faiss的`IndexFlatL2`得到的Top-1（即K=1）的结果作为Ground Truth，用于测试`IndexIVFFlat`，`IndexIVFFlat`永远达不到100%的`recall-at-1`，即使`nprobe == nlist`，`recall-at-1`也不能100%。    

我发现问题是`sift1M`或者`gist1M`或者`deep1B`数据集，它们存在对于一个query向量相同距离的n个base向量，而Top-1向量会返回其中随机的一个Index，因此，出现了距离相同但是Index不同，导致结构匹配不上的情况。   

因此，下面的代码把对于一个query向量所有相同距离的base向量都找出，作为TopK。   
举个例子，对于query向量q1，base向量b1,b2,b3，如果`Dis(q1,b1)==Dis(q1,b2)<Dis(q1,b3)`，则Top1为b1和b2的索引。   

计算Ground Truth：
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
保存Ground Truth：
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

读取Ground Truth：
```python
    gt = []
    with open(os.path.join(DB_DIR, gt_file_name), 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            row = list(map(int, row))
            gt.append(row)
```

Recall计算方式：
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