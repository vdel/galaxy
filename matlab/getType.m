function type = getType(proba)
    [~, type] = max(proba(:, 1:3), [], 2);
end