[proba, ids] = readData();
[pgold, igold] = getGold(proba, ids)
type = getType(pgold);
hgr = compute(ids, 0);
