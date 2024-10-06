I = imread('Aerial04.jpg');
J = imnoise(I, 'gaussian', 0.25);

function f = lowPassBlockFilterGenerator(n)
    f = ones(n, n)/(n*n);
end

function f = binomialFilterGenerator(n)
    res = zeros(n, 1);
    for k = 0:n-1
        res(k + 1 , 1) = factorial(n - 1)/(factorial(k)*factorial(n - 1 - k));
    end
    m = res*res';
    f = m/sum(m, 'all');
end

function f = imageConvolver(colorImage, filter)
    rChannel = colorImage(:, :, 1);
    gChannel = colorImage(:, :, 2);
    bChannel = colorImage(:, :, 3);
    f = uint8(cat(3, conv2(rChannel, filter), conv2(gChannel, filter), conv2(bChannel, filter)));
end

function f = prewittFilterGenerator(n, orientation)
    if orientation == 'h'
        
    else

    end;
end


%imwrite(imageConvolver(I, lowPassBlockFilterGenerator(3)), "3x3LowPassOriginal.jpg");
%imwrite(imageConvolver(J, lowPassBlockFilterGenerator(3)), "3x3LowPassNoisy.jpg");
%imwrite(imageConvolver(I, lowPassBlockFilterGenerator(7)), "7x7LowPassOriginal.jpg");
%imwrite(imageConvolver(J, lowPassBlockFilterGenerator(7)), "7x7LowPassNoisy.jpg");
%imwrite(imageConvolver(I, lowPassBlockFilterGenerator(9)), "9x9LowPassOriginal.jpg");
%imwrite(imageConvolver(J, lowPassBlockFilterGenerator(9)), "9x9LowPassNoisy.jpg");
%imwrite(imageConvolver(I, lowPassBlockFilterGenerator(11)), "11x11LowPassOriginal.jpg");
%imwrite(imageConvolver(J, lowPassBlockFilterGenerator(11)), "11x11LowPassNoisy.jpg");

%imwrite(imageConvolver(I, binomialFilterGenerator(3)), "3x3LowPassGaussianOriginal.jpg");
%imwrite(imageConvolver(J, binomialFilterGenerator(3)), "3x3LowPassGaussianNoisy.jpg");
%imwrite(imageConvolver(I, binomialFilterGenerator(7)), "7x7LowPassGaussianOriginal.jpg");
%imwrite(imageConvolver(J, binomialFilterGenerator(7)), "7x7LowPassGaussianNoisy.jpg");
%imwrite(imageConvolver(I, binomialFilterGenerator(9)), "9x9LowPassGaussianOriginal.jpg");
%imwrite(imageConvolver(J, binomialFilterGenerator(9)), "9x9LowPassGaussianNoisy.jpg");
%imwrite(imageConvolver(I, binomialFilterGenerator(11)), "11x11LowPassGaussianOriginal.jpg");
%imwrite(imageConvolver(J, binomialFilterGenerator(11)), "11x11LowPassGaussianNoisy.jpg");

imwrite(imageConvolver(I, [1, -1]), "basicVerticalBorderDetectorOriginal.jpg");
imwrite(imageConvolver(J, [1, -1]), "basicVerticalBorderDetectorNoisy.jpg");
imwrite(imageConvolver(I, [1; -1]), "basicHorizontalBorderDetectorOriginal.jpg");
imwrite(imageConvolver(J, [1; -1]), "basicHorizontalBorderDetectorNoisy.jpg");