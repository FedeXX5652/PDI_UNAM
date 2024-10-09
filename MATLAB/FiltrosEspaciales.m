% Se lee la imagen
I = imread('Aerial04.jpg');

% Convertir la imagen a escala de grises
I_gray = rgb2gray(I);

% Se le agrega ruido a la imagen en escala de grises
J = imnoise(I_gray, 'gaussian', 0.25);

% Crear un arreglo de filtros a aplicar
filters = {
    lowPassBlockFilterGenerator(5), ...  % Filtro paso bajo 5x5
    binomialFilterGenerator(5), ...        % Filtro binomial
    secondDerivativeBinomialFilterGenerator(5), ...  % Filtro de Laplaciano
    firstDerivativeBinomialFilterGenerator(5) ...    % Filtro gaussiano de primera derivada
};

function filter = lowPassBlockFilterGenerator(N)
    % Crea un filtro de paso bajo de tamaño NxN
    filter = ones(N) / (N * N);
end

function f = binomialFilterGenerator(size)
    % Crea un filtro binomial de tamaño 'size x size'
    if mod(size, 2) == 0
        error('El tamaño del filtro debe ser impar.');
    end

    % Calcular los coeficientes del filtro binomial
    coeff = zeros(1, size);
    for i = 0:size-1
        coeff(i+1) = nchoosek(size-1, i);
    end
    coeff = coeff / sum(coeff); % Normalizar

    f = coeff' * coeff; % Producto exterior
end

function filter = firstDerivativeBinomialFilterGenerator(size)
    if mod(size, 2) == 0
        error('Tamaño de filtro incorrecto');
    end
    
    binomialCoefficients = zeros(1, size);
    for k = 0:size-1
        binomialCoefficients(k+1) = nchoosek(size-1, k);
    end
    
    % Normalizar
    binomialCoefficients = binomialCoefficients / sum(binomialCoefficients);
    
    % Primera derivada
    filter = zeros(1, size);
    for i = 1:size
        if i < (size + 1) / 2
            filter(i) = -binomialCoefficients(i);
        else
            filter(i) = binomialCoefficients(i);
        end
    end
end

function filter = secondDerivativeBinomialFilterGenerator(size)
    if mod(size, 2) == 0
        error('El tamaño del filtro debe ser impar.');
    end
    
    % Crear un vector de Gauss 1D
    x = -floor(size/2):floor(size/2);
    binomialCoeffs = binomialCoefficients(size-1);
    filter = -x.^2 .* binomialCoeffs;  % Segunda derivada
    
    % Normalizar el filtro
    filter = filter / sum(abs(filter));
    
    % Convertir a 2D
    filter = filter' * filter;
end

function coeffs = binomialCoefficients(n)
    % Calcular los coeficientes binomiales
    coeffs = zeros(1, n + 1);
    for k = 0:n
        coeffs(k + 1) = nchoosek(n, k);
    end
    coeffs = coeffs / sum(coeffs);  % Normalizar
end

% Aplicar los filtros a la imagen original y a la imagen ruidosa
for i = 1:length(filters)
    filter = filters{i};
    
    % Difuminar la imagen original
    blurredImageWithoutNoise = imageConvolver(I_gray, filter);
    figure, imshow(blurredImageWithoutNoise);
    title(['Blurred Image without Noise - Filter ', num2str(i)]);
    
    % Difuminar la imagen con ruido
    blurredImageWithNoise = imageConvolver(J, filter);
    figure, imshow(blurredImageWithNoise);
    title(['Blurred Image with Noise - Filter ', num2str(i)]);
end

function filter = prewittFilterGenerator(direction)
    if direction == 'x'
        filter = [-1, 0, 1; -1, 0, 1; -1, 0, 1];  % Filtro Prewitt en  X
    elseif direction == 'y'
        filter = [-1, -1, -1; 0, 0, 0; 1, 1, 1];  % Filtro Prewitt en  Y
    else
        error('Direction must be either ''x'' or ''y''.');
    end
end

function filter = sobelFilterGenerator(direction)
    if strcmp(direction, 'x')
        filter = [-1, 0, 1; -2, 0, 2; -1, 0, 1]; % Filtro Sobel para X
    elseif strcmp(direction, 'y')
        filter = [1, 2, 1; 0, 0, 0; -1, -2, -1]; % Filtro Sobel para Y
    else
        error('Direccion invalida');
    end
end

function sharpened = unsharpMasking(original, blurred)
    mascara = original - blurred;
    
    sharpened = original + mascara;
    
    % Normalizar
    sharpened = uint8(min(max(sharpened, 0), 255));
end

function outputImage = imageConvolver(inputImage, filter)
    % Aplica un filtro a la imagen de entrada utilizando la convolución
    % inputImage: imagen de entrada (grayscale)
    % filter: filtro a aplicar (matriz)

    % Realiza la convolución
    outputImage = conv2(double(inputImage), filter, 'same');

    % Asegúrate de que el resultado sea una imagen de tipo uint8
    outputImage = uint8(outputImage);
end


% Aquí comienzan las líneas que guardan las imágenes

% Guardar imágenes filtradas con diferentes filtros
imwrite(imageConvolver(I_gray, lowPassBlockFilterGenerator(3)), "3x3LowPassOriginal.jpg");
imwrite(imageConvolver(J, lowPassBlockFilterGenerator(3)), "3x3LowPassNoisy.jpg");
imwrite(imageConvolver(I_gray, lowPassBlockFilterGenerator(7)), "7x7LowPassOriginal.jpg");
imwrite(imageConvolver(J, lowPassBlockFilterGenerator(7)), "7x7LowPassNoisy.jpg");
imwrite(imageConvolver(I_gray, lowPassBlockFilterGenerator(9)), "9x9LowPassOriginal.jpg");
imwrite(imageConvolver(J, lowPassBlockFilterGenerator(9)), "9x9LowPassNoisy.jpg");
imwrite(imageConvolver(I_gray, lowPassBlockFilterGenerator(11)), "11x11LowPassOriginal.jpg");
imwrite(imageConvolver(J, lowPassBlockFilterGenerator(11)), "11x11LowPassNoisy.jpg");

imwrite(imageConvolver(I_gray, binomialFilterGenerator(3)), "3x3LowPassGaussianOriginal.jpg");
imwrite(imageConvolver(J, binomialFilterGenerator(3)), "3x3LowPassGaussianNoisy.jpg");
imwrite(imageConvolver(I_gray, binomialFilterGenerator(7)), "7x7LowPassGaussianOriginal.jpg");
imwrite(imageConvolver(J, binomialFilterGenerator(7)), "7x7LowPassGaussianNoisy.jpg");
imwrite(imageConvolver(I_gray, binomialFilterGenerator(9)), "9x9LowPassGaussianOriginal.jpg");
imwrite(imageConvolver(J, binomialFilterGenerator(9)), "9x9LowPassGaussianNoisy.jpg");
imwrite(imageConvolver(I_gray, binomialFilterGenerator(11)), "11x11LowPassGaussianOriginal.jpg");
imwrite(imageConvolver(J, binomialFilterGenerator(11)), "11x11LowPassGaussianNoisy.jpg");

imwrite(imageConvolver(I_gray, [-1, 1]), "basicVerticalBorderDetectorOriginal.jpg");
imwrite(imageConvolver(J, [-1, 1]), "basicVerticalBorderDetectorNoisy.jpg");
imwrite(imageConvolver(I_gray, [-1; 1]), "basicHorizontalBorderDetectorOriginal.jpg");
imwrite(imageConvolver(J, [-1; 1]), "basicHorizontalBorderDetectorNoisy.jpg");

imwrite(imageConvolver(I_gray, prewittFilterGenerator('x')), "prewittXOriginal.jpg");
imwrite(imageConvolver(I_gray, prewittFilterGenerator('y')), "prewittYOriginal.jpg");
imwrite(imageConvolver(J, prewittFilterGenerator('x')), "prewittXNoisy.jpg");
imwrite(imageConvolver(J, prewittFilterGenerator('y')), "prewittYNoisy.jpg");

imwrite(imageConvolver(I_gray, sobelFilterGenerator('x')), "sobelXOriginal.jpg");
imwrite(imageConvolver(I_gray, sobelFilterGenerator('y')), "sobelYOriginal.jpg");
imwrite(imageConvolver(J, sobelFilterGenerator('x')), "sobelXNoisy.jpg");
imwrite(imageConvolver(J, sobelFilterGenerator('y')), "sobelYNoisy.jpg");

imwrite(imageConvolver(I_gray, firstDerivativeBinomialFilterGenerator(5)), "gaussianDerivativeOriginal5x5.jpg");
imwrite(imageConvolver(J, firstDerivativeBinomialFilterGenerator(5)), "gaussianDerivativeNoisy5x5.jpg");
imwrite(imageConvolver(I_gray, firstDerivativeBinomialFilterGenerator(7)), "gaussianDerivativeOriginal7x7.jpg");
imwrite(imageConvolver(J, firstDerivativeBinomialFilterGenerator(7)), "gaussianDerivativeNoisy7x7.jpg");
imwrite(imageConvolver(I_gray, firstDerivativeBinomialFilterGenerator(11)), "gaussianDerivativeOriginal11x11.jpg");
imwrite(imageConvolver(J, firstDerivativeBinomialFilterGenerator(11)), "gaussianDerivativeNoisy11x11.jpg");

laplacian3x3 = [-1, -1, -1; -1, 8, -1; -1, -1, -1];
imwrite(imageConvolver(I_gray, laplacian3x3), "3x3LaplacianOriginal.jpg");
imwrite(imageConvolver(J, laplacian3x3), "3x3LaplacianNoisy.jpg");

imwrite(imageConvolver(I_gray, secondDerivativeBinomialFilterGenerator(5)), "gaussianSecondDerivativeOriginal5x5.jpg");
imwrite(imageConvolver(J, secondDerivativeBinomialFilterGenerator(5)), "gaussianSecondDerivativeNoisy5x5.jpg");
imwrite(imageConvolver(I_gray, secondDerivativeBinomialFilterGenerator(7)), "gaussianSecondDerivativeOriginal7x7.jpg");
imwrite(imageConvolver(J, secondDerivativeBinomialFilterGenerator(7)), "gaussianSecondDerivativeNoisy7x7.jpg");
imwrite(imageConvolver(I_gray, secondDerivativeBinomialFilterGenerator(11)), "gaussianSecondDerivativeOriginal11x11.jpg");
imwrite(imageConvolver(J, secondDerivativeBinomialFilterGenerator(11)), "gaussianSecondDerivativeNoisy11x11.jpg");

imwrite(imageConvolver(I_gray, lowPassBlockFilterGenerator(5)), "5x5LowPassOriginal.jpg");
imwrite(imageConvolver(J, lowPassBlockFilterGenerator(5)), "5x5LowPassNoisy.jpg");

blurred3x3BlockOriginal = imageConvolver(I_gray, lowPassBlockFilterGenerator(3));
blurred7x7BlockOriginal = imageConvolver(I_gray, lowPassBlockFilterGenerator(7));
imwrite(unsharpMasking(I_gray, blurred3x3BlockOriginal), "unsharpBlock3x3Original.jpg");
imwrite(unsharpMasking(I_gray, blurred7x7BlockOriginal), "unsharpBlock7x7Original.jpg");

blurred3x3BlockNoisy = imageConvolver(J, lowPassBlockFilterGenerator(3));
blurred7x7BlockNoisy = imageConvolver(J, lowPassBlockFilterGenerator(7));
imwrite(unsharpMasking(J, blurred3x3BlockNoisy), "unsharpBlock3x3Noisy.jpg");
imwrite(unsharpMasking(J, blurred7x7BlockNoisy), "unsharpBlock7x7Noisy.jpg");

blurred3x3BinomialOriginal = imageConvolver(I_gray, binomialFilterGenerator(3));
blurred7x7BinomialOriginal = imageConvolver(I_gray, binomialFilterGenerator(7));
imwrite(unsharpMasking(I_gray, blurred3x3BinomialOriginal), "unsharpBinomial3x3Original.jpg");
imwrite(unsharpMasking(I_gray, blurred7x7BinomialOriginal), "unsharpBinomial7x7Original.jpg");

blurred3x3BinomialNoisy = imageConvolver(J, binomialFilterGenerator(3));
blurred7x7BinomialNoisy = imageConvolver(J, binomialFilterGenerator(7));
imwrite(unsharpMasking(J, blurred3x3BinomialNoisy), "unsharpBinomial3x3Noisy.jpg");
imwrite(unsharpMasking(J, blurred7x7BinomialNoisy), "unsharpBinomial7x7Noisy.jpg");

imwrite(imageConvolver(imageConvolver(I_gray, lowPassBlockFilterGenerator(5)), ...
    lowPassBlockFilterGenerator(3)), "5x5LowPassThenBlock3x3Original.jpg");

imwrite(imageConvolver(imageConvolver(I_gray, lowPassBlockFilterGenerator(5)), lowPassBlockFilterGenerator(7)), "5x5LowPassThenBlock7x7Original.jpg");

imwrite(imageConvolver(imageConvolver(J, lowPassBlockFilterGenerator(5)), lowPassBlockFilterGenerator(3)), "5x5LowPassThenBlock3x3Noisy.jpg");
imwrite(imageConvolver(imageConvolver(J, lowPassBlockFilterGenerator(5)), lowPassBlockFilterGenerator(7)), "5x5LowPassThenBlock7x7Noisy.jpg");

imwrite(imageConvolver(imageConvolver(I_gray, lowPassBlockFilterGenerator(5)), binomialFilterGenerator(3)), "5x5LowPassThenBinomial3x3Original.jpg");
imwrite(imageConvolver(imageConvolver(I_gray, lowPassBlockFilterGenerator(5)), binomialFilterGenerator(7)), "5x5LowPassThenBinomial7x7Original.jpg");

imwrite(imageConvolver(imageConvolver(J, lowPassBlockFilterGenerator(5)), binomialFilterGenerator(3)), "5x5LowPassThenBinomial3x3Noisy.jpg");
imwrite(imageConvolver(imageConvolver(J, lowPassBlockFilterGenerator(5)), binomialFilterGenerator(7)), "5x5LowPassThenBinomial7x7Noisy.jpg");