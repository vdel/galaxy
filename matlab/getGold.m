function [proba, ids] = getGold(proba, ids)
    I = max(proba(:, 1:3), [], 2) > 0.8;
    proba = proba(I, :);
    ids = ids(I);
end