function [proba ids] = readData()
    file = '../data/training_solutions.csv';
    proba = csvread(file, 1, 1);
    fid = fopen(file, 'rt');
    fgetl(fid);
    ids = cell(size(proba, 1), 1);
    i = 0;
    while ~feof(fid)
        i = i + 1;
        line = fgetl(fid);
        k = strfind(line, ',') - 1;
        ids{i} = line(1 : k);
    end
end