function [mag, magcenter, img] = readImg(id, crop, cropCenter)
    img = imread(fullfile('../data/images_training', [id '.jpg']));    

    cropAdjust = 40;
    pad = (424 - cropAdjust) / 2;
    I = (pad + 1) : (424 - pad);
    magcenter = imfilter(double(img(I, I, :)), fspecial('gaussian', [3, 3], 1), 'replicate');
    magcenter = mean(bsxfun(@times, double(magcenter), reshape([1 1.176 1.818], [1 1 3])), 3);
    
    if 0
        dx = 0;
        dy = 0;
    else
        [m, dy] = max(magcenter);
        [~, dx] = max(m);
        dy = dy(dx) - cropAdjust / 2;
        dx = dx - cropAdjust / 2;    
    end
    
    pad = (424 - cropCenter) / 2;
    Ix = ((pad + 1) : (424 - pad)) + dx;
    Iy = ((pad + 1) : (424 - pad)) + dy;
    magcenter = imfilter(img(Iy, Ix, :), fspecial('gaussian', [3, 3], 1), 'replicate');
    magcenter = mean(bsxfun(@times, double(magcenter), reshape([1 1.176 1.818], [1 1 3])), 3);

    pad = (424 - crop) / 2;
    Ix = ((pad + 1) : (424 - pad)) + dx;
    Iy = ((pad + 1) : (424 - pad)) + dy;
   
    img = img(Iy, Ix, :);
    mag = bsxfun(@times, double(img), reshape([1 1.176 1.818], [1 1 3]));
    
    mag = max(mag, 1);
    mag = -log(mag);
    magcenter = max(magcenter, 1);
    magcenter = -log(double(magcenter) * 1.818);
end