function data = compute(ids)    
    nimg = length(ids);
    data = struct('gr', cell(nimg, 1), 'ri', []);
    
    crop = 160;
    
    cropCenter = 120;
    I = (1 : cropCenter) - cropCenter / 2 - 0.5;
    [x, y] = meshgrid(I, I);
    x = x(:);
    y = y(:);
    xx = x .^ 2;
    yy = y .^ 2;
    xy = 2 * x.* y;
    X = [xx yy xy];
    
    for i = 1 : nimg
        fprintf('Processing %d/%d\n', i, nimg);
        [mag cmag img] = readImg(ids{i}, crop, cropCenter);
        
        central = centralPoint(mag);
        
        data(i).gr = central(1) - central(2);
        data(i).ri = central(2) - central(3);
        
        % DeVaucouleurs
        Y = (cmag - central(3)) .^ 8;        
        Y = Y(:);
        I = ~isinf(Y);
        
        p = X(I, :) \ Y(I);
        A = [p(1) p(3); p(3) p(2)];
        
        if 1 
            [v, d] = eig(A);
            d
            d = 1 ./ sqrt(diag(d));
            d = d * 20;
            a = atan2(v(2, 1), v(1, 1));            
            imagesc(img);
            hold on;
            ellipse(d(1), d(2), a, 80.5, 80.5);  
            hold off;
            pause;
        end        
    end
end

function c = centralPoint(mag)
    center = floor(size(mag, 1) / 2);
    I = center : (center + 1);
    c = mean(reshape(mag(I, I, :), [4, 3]), 1);
end