% Se lee la imagen
I = imread('Aerial04.jpg');

% Se le agrega ruido a la imagen
J = imnoise(I, 'gaussian', 0.25);

% Filtro paso bajo 5x5
lowPassFilter5x5 = lowPassBlockFilterGenerator(5);

% Difuminar la imagen sin ruido
blurredImageWithoutNoise = imageConvolver(I, lowPassFilter5x5);

% Difuminar la imagen con ruido
blurredImageWithNoise = imageConvolver(J, lowPassFilter5x5);

% Convertir la imagen con ruido a escala de grises
J_gray = rgb2gray(J);

% Función generadora de filtros paso bajas de bloque
function f = lowPassBlockFilterGenerator(n)
    f = ones(n, n)/(n*n);
end

% Función que genera un vector compuesto por los coeficientes binomiales.
function f = binomialCoefficientVectorGenerator(n)
    res = zeros(n, 1);
    for k = 0:n-1
        res(k + 1 , 1) = factorial(n - 1)/(factorial(k)*factorial(n - 1 - k));
    end
    f = res;
end

% Función que genera un vector compuesto por la primera derivada de los coeficientes binomiales.
function f = binomialCoefficientVectorGeneratorFirstDerivative(n)
    nMinusOne = binomialCoefficientVectorGenerator(n - 1);
    f = paddata(nMinusOne, n) - paddata(nMinusOne, n, Side='leading');
end

% Función que genera un vector compuesto por la segunda derivada de los coeficientes binomiales.
function f = binomialCoefficientVectorGeneratorSecondDerivative(n)
    nMinusOne = binomialCoefficientVectorGeneratorFirstDerivative(n - 1);
    f = paddata(nMinusOne, n) - paddata(nMinusOne, n, Side='leading');
end

% Función generadora de filtros gaussianos de segunda derivada.
function f = secondDerivativeBinomialFilterGenerator(n)
    v = binomialCoefficientVectorGeneratorSecondDerivative(n);
    f = v*v';
end

% Función generadora de filtros gaussianos de primera derivada.
function f = firstDerivativeBinomialFilterGenerator(n)
    v = binomialCoefficientVectorGeneratorFirstDerivative(n);
    f = v*v';
end

% Función generadora de filtros gaussianos.
function f = binomialFilterGenerator(n)
    coefficients = binomialCoefficientVectorGenerator(n);
    m = coefficients*coefficients';
    f = m/sum(m, 'all');
end

% Función aplicadora de filtros.
function f = imageConvolver(colorImage, filter)
    rChannel = colorImage(:, :, 1);
    gChannel = colorImage(:, :, 2);
    bChannel = colorImage(:, :, 3);
    f = uint8(cat(3, conv2(rChannel, filter, 'same'), ...
                  conv2(gChannel, filter, 'same'), ...
                  conv2(bChannel, filter, 'same')));
end

% Función generadora de filtro Prewitt según la orientación dada.
function f = prewittFilterGenerator(orientation)
    if orientation == 'x'
        f = [1 ; 1; 1]*[1, 0, -1];
    else 
        f = [1; 0; -1]*[1, 1, 1];
    end
end

% Función generadora de filtro Sobel según la orientación dada.
function f = sobelFilterGenerator(orientation)
    if orientation == 'x'
        f = [-1; -2; -1]*[1, 0, -1];
    else
        f = [1; 0; -1]*[-1, -2, -1];
    end
end

% Función simple de enfoque de enmascaramiento
function f = unsharpMasking(originalImage, blurredImage)
    f = originalImage + (originalImage - blurredImage);
end

imwrite(imageConvolver(I, lowPassBlockFilterGenerator(3)), "3x3LowPassOriginal.jpg");
imwrite(imageConvolver(J, lowPassBlockFilterGenerator(3)), "3x3LowPassNoisy.jpg");
imwrite(imageConvolver(I, lowPassBlockFilterGenerator(7)), "7x7LowPassOriginal.jpg");
imwrite(imageConvolver(J, lowPassBlockFilterGenerator(7)), "7x7LowPassNoisy.jpg");
imwrite(imageConvolver(I, lowPassBlockFilterGenerator(9)), "9x9LowPassOriginal.jpg");
imwrite(imageConvolver(J, lowPassBlockFilterGenerator(9)), "9x9LowPassNoisy.jpg");
imwrite(imageConvolver(I, lowPassBlockFilterGenerator(11)), "11x11LowPassOriginal.jpg");
imwrite(imageConvolver(J, lowPassBlockFilterGenerator(11)), "11x11LowPassNoisy.jpg");

imwrite(imageConvolver(I, binomialFilterGenerator(3)), "3x3LowPassGaussianOriginal.jpg");
imwrite(imageConvolver(J, binomialFilterGenerator(3)), "3x3LowPassGaussianNoisy.jpg");
imwrite(imageConvolver(I, binomialFilterGenerator(7)), "7x7LowPassGaussianOriginal.jpg");
imwrite(imageConvolver(J, binomialFilterGenerator(7)), "7x7LowPassGaussianNoisy.jpg");
imwrite(imageConvolver(I, binomialFilterGenerator(9)), "9x9LowPassGaussianOriginal.jpg");
imwrite(imageConvolver(J, binomialFilterGenerator(9)), "9x9LowPassGaussianNoisy.jpg");
imwrite(imageConvolver(I, binomialFilterGenerator(11)), "11x11LowPassGaussianOriginal.jpg");
imwrite(imageConvolver(J, binomialFilterGenerator(11)), "11x11LowPassGaussianNoisy.jpg");

imwrite(imageConvolver(I, [-1, 1]), "basicVerticalBorderDetectorOriginal.jpg");
imwrite(imageConvolver(J, [-1, 1]), "basicVerticalBorderDetectorNoisy.jpg");
imwrite(imageConvolver(I, [-1; 1]), "basicHorizontalBorderDetectorOriginal.jpg");
imwrite(imageConvolver(J, [-1; 1]), "basicHorizontalBorderDetectorNoisy.jpg");

imwrite(imageConvolver(I, prewittFilterGenerator('x')), "prewittXOriginal.jpg");
imwrite(imageConvolver(I, prewittFilterGenerator('y')), "prewittYOriginal.jpg");
imwrite(imageConvolver(J, prewittFilterGenerator('x')), "prewittXNoisy.jpg");
imwrite(imageConvolver(J, prewittFilterGenerator('y')), "prewittYNoisy.jpg");

imwrite(imageConvolver(I, sobelFilterGenerator('x')), "sobelXOriginal.jpg");
imwrite(imageConvolver(I, sobelFilterGenerator('y')), "sobelYOriginal.jpg");
imwrite(imageConvolver(J, sobelFilterGenerator('x')), "sobelXNoisy.jpg");
imwrite(imageConvolver(J, sobelFilterGenerator('y')), "sobelYNoisy.jpg");

imwrite(imageConvolver(I, firstDerivativeBinomialFilterGenerator(5)), "gaussianDerivativeOriginal5x5.jpg");
imwrite(imageConvolver(J, firstDerivativeBinomialFilterGenerator(5)), "gaussianDerivativeNoisy5x5.jpg");
imwrite(imageConvolver(I, firstDerivativeBinomialFilterGenerator(7)), "gaussianDerivativeOriginal7x7.jpg");
imwrite(imageConvolver(J, firstDerivativeBinomialFilterGenerator(7)), "gaussianDerivativeNoisy7x7.jpg");
imwrite(imageConvolver(I, firstDerivativeBinomialFilterGenerator(11)), "gaussianDerivativeOriginal11x11.jpg");
imwrite(imageConvolver(J, firstDerivativeBinomialFilterGenerator(11)), "gaussianDerivativeNoisy11x11.jpg");

laplacian3x3 = [-1, -1, -1; -1, 8, -1; -1, -1, -1];
imwrite(imageConvolver(I, laplacian3x3), "3x3LaplacianOriginal.jpg");
imwrite(imageConvolver(J, laplacian3x3), "3x3LaplacianNoisy.jpg");

imwrite(imageConvolver(I, secondDerivativeBinomialFilterGenerator(5)), "gaussianSecondDerivativeOriginal5x5.jpg");
imwrite(imageConvolver(J, secondDerivativeBinomialFilterGenerator(5)), "gaussianSecondDerivativeNoisy5x5.jpg");
imwrite(imageConvolver(I, secondDerivativeBinomialFilterGenerator(7)), "gaussianSecondDerivativeOriginal7x7.jpg");
imwrite(imageConvolver(J, secondDerivativeBinomialFilterGenerator(7)), "gaussianSecondDerivativeNoisy7x7.jpg");
imwrite(imageConvolver(I, secondDerivativeBinomialFilterGenerator(11)), "gaussianSecondDerivativeOriginal11x11.jpg");
imwrite(imageConvolver(J, secondDerivativeBinomialFilterGenerator(11)), "gaussianSecondDerivativeNoisy11x11.jpg");

imwrite(imageConvolver(I, lowPassBlockFilterGenerator(5)), "5x5LowPassOriginal.jpg");
imwrite(imageConvolver(J, lowPassBlockFilterGenerator(5)), "5x5LowPassNoisy.jpg");

blurred3x3BlockOriginal = imageConvolver(I, lowPassBlockFilterGenerator(3));
blurred7x7BlockOriginal = imageConvolver(I, lowPassBlockFilterGenerator(7));
imwrite(unsharpMasking(I, blurred3x3BlockOriginal), "unsharpBlock3x3Original.jpg");
imwrite(unsharpMasking(I, blurred7x7BlockOriginal), "unsharpBlock7x7Original.jpg");

blurred3x3BlockNoisy = imageConvolver(J, lowPassBlockFilterGenerator(3));
blurred7x7BlockNoisy = imageConvolver(J, lowPassBlockFilterGenerator(7));
imwrite(unsharpMasking(J, blurred3x3BlockNoisy), "unsharpBlock3x3Noisy.jpg");
imwrite(unsharpMasking(J, blurred7x7BlockNoisy), "unsharpBlock7x7Noisy.jpg");

blurred3x3BinomialOriginal = imageConvolver(I, binomialFilterGenerator(3));
blurred7x7BinomialOriginal = imageConvolver(I, binomialFilterGenerator(7));
imwrite(unsharpMasking(I, blurred3x3BinomialOriginal), "unsharpBinomial3x3Original.jpg");
imwrite(unsharpMasking(I, blurred7x7BinomialOriginal), "unsharpBinomial7x7Original.jpg");

blurred3x3BinomialNoisy = imageConvolver(J, binomialFilterGenerator(3));
blurred7x7BinomialNoisy = imageConvolver(J, binomialFilterGenerator(7));
imwrite(unsharpMasking(J, blurred3x3BinomialNoisy), "unsharpBinomial3x3Noisy.jpg");
imwrite(unsharpMasking(J, blurred7x7BinomialNoisy), "unsharpBinomial7x7Noisy.jpg");

imwrite(imageConvolver(imageConvolver(I, lowPassBlockFilterGenerator(5)), ...
    lowPassBlockFilterGenerator(3)), "5x5LowPassThenBlock3x3Original.jpg");

imwrite(imageConvolver(imageConvolver(I, lowPassBlockFilterGenerator(5)), lowPassBlockFilterGenerator(7)), "5x5LowPassThenBlock7x7Original.jpg");

imwrite(imageConvolver(imageConvolver(J, lowPassBlockFilterGenerator(5)), lowPassBlockFilterGenerator(3)), "5x5LowPassThenBlock3x3Noisy.jpg");
imwrite(imageConvolver(imageConvolver(J, lowPassBlockFilterGenerator(5)), lowPassBlockFilterGenerator(7)), "5x5LowPassThenBlock7x7Noisy.jpg");

imwrite(imageConvolver(imageConvolver(I, lowPassBlockFilterGenerator(5)), binomialFilterGenerator(3)), "5x5LowPassThenBinomial3x3Original.jpg");
imwrite(imageConvolver(imageConvolver(I, lowPassBlockFilterGenerator(5)), binomialFilterGenerator(7)), "5x5LowPassThenBinomial7x7Original.jpg");

imwrite(imageConvolver(imageConvolver(J, lowPassBlockFilterGenerator(5)), binomialFilterGenerator(3)), "5x5LowPassThenBinomial3x3Noisy.jpg");
imwrite(imageConvolver(imageConvolver(J, lowPassBlockFilterGenerator(5)), binomialFilterGenerator(7)), "5x5LowPassThenBinomial7x7Noisy.jpg");