% Cargar imagen sin ruido
ImagenSinRuido = imread('Glaciar512.jpg'); 

if size(ImagenSinRuido, 3) == 3
    ImagenSinRuido = rgb2gray(ImagenSinRuido);
end

% ----------------

% Punto 1
Sn = 0.01; % Varianza
% Transformada
ImagenSinRuidoFrec = fft2(double(ImagenSinRuido)); % Transformada de Fourier

% Densidad espectral de la imagen original
PImagenSinRuido = abs(ImagenSinRuidoFrec).^2; 

% Generar ruido en el dominio de frecuencias
ImagenConRuido = imnoise(zeros(size(ImagenSinRuido)), 'gaussian', 0.25, Sn); % Crear imagen de ruido
ImagenConRuidoFrec = fft2(double(ImagenConRuido)); % Transformada de Fourier del ruido

% Imagen con ruido en el dominio de frecuencias
ImagenConRuidoFrec = ImagenSinRuidoFrec + ImagenConRuidoFrec; 

% Calcular la nueva varianza del ruido
Sn = var(double(ImagenConRuido(:)));

% Densidad espectral de la imagen con ruido
PImagenConRuido = abs(ImagenConRuidoFrec).^2; 

WienerFiltro = 1 ./ (1 + (Sn ./ PImagenConRuido)); % Filtro de Wiener

ImagenFiltradaWienerFrec = ImagenConRuidoFrec .* WienerFiltro;

% Transformada inversa para volver al dominio espacial
ImagenFiltradaWiener = ifft2(ImagenFiltradaWienerFrec);

% ----------------

%Punto 2
% G = F . H (En dominio de frecuencias)

% Filtro binomial 9x9 (Matriz H)
FiltroBinomial = [1 8 28 56 70 56 28 8 1]' * [1 8 28 56 70 56 28 8 1];
FiltroBinomial = FiltroBinomial / sum(FiltroBinomial(:)); % Normalizar

% Transformada de Fourier de la imagen (Matriz F)
ImagenFrec = fft2(double(ImagenSinRuido));
% Transformada de Fourier del filtro (Matriz H)
FiltroFrec = fft2(FiltroBinomial, 512, 512); % Padding del filtro

% Multiplicacion en dominio de frecuencias (Matriz G)
ImagenFiltradaFrec = ImagenFrec .* FiltroFrec; 

% Transformada inversa para volver al dominio espacial
ImagenFiltrada = ifft2(ImagenFiltradaFrec);

% Calcular la matriz inversa del filtro 
% en el dominio de la frecuencia (Matriz H^-1)
FiltroFrecInverso = zeros(size(FiltroFrec));

for i = 1:numel(FiltroFrec) % Evitar div entre 0
    if FiltroFrec(i) ~= 0
        FiltroFrecInverso(i) = 1 / FiltroFrec(i);
    end
end

% Producto de las matrices G y H^-1
ProductoGHInverso = ImagenFiltradaFrec .* FiltroFrecInverso; 

% Transformada inversa para volver al dominio espacial
ImagenProductoGHInverso = ifft2(ProductoGHInverso);

% ----------------

% Punto 3: Imagen con ruido gaussiano y pérdida de nitidez

% 1. Agregar ruido gaussiano a la imagen original
ImagenConRuidoYDesenfoque = imnoise(ImagenSinRuido, 'gaussian', 0.25, Sn); % Usar el mismo Sn que se definió antes

% 2. Filtrar la imagen con el filtro de paso bajo (Filtro binomial)
ImagenConRuidoYDesenfoqueFrec = fft2(double(ImagenConRuidoYDesenfoque)); % Transformada de Fourier de la imagen con ruido
ImagenFiltradaConRuidoYDesenfoqueFrec = ImagenConRuidoYDesenfoqueFrec .* FiltroFrec; % Multiplicación en el dominio de frecuencias

% 3. Transformada inversa para obtener la imagen con ruido y desenfoque
ImagenConRuidoYDesenfoqueFiltrada = ifft2(ImagenFiltradaConRuidoYDesenfoqueFrec);

% 4. Calcular la nueva varianza del ruido en la imagen con ruido y desenfoque
SnDesenfoque = var(double(ImagenConRuidoYDesenfoqueFiltrada(:)));

% 5. Densidad espectral de la imagen con ruido y desenfoque
PImagenConRuidoYDesenfoque = abs(ImagenFiltradaConRuidoYDesenfoqueFrec).^2; 

% 6. Calcular el filtro de Wiener
WienerFiltroDesenfoque = 1 ./ (1 + (SnDesenfoque ./ PImagenConRuidoYDesenfoque)); 

% 7. Filtrar la imagen con el filtro de Wiener
ImagenFiltradaWienerDesenfoqueFrec = ImagenFiltradaConRuidoYDesenfoqueFrec .* WienerFiltroDesenfoque;

% 8. Transformada inversa para volver al dominio espacial
ImagenFiltradaWienerDesenfoque = ifft2(ImagenFiltradaWienerDesenfoqueFrec);

% ----------------

% Resultados
figure;
imshow(uint8(ImagenSinRuido));
title('Imagen Sin Ruido');

figure;
imshow(log(1 + abs(fftshift(ImagenFrec))), []);
title('Imagen Sin Ruido en frecuencias (Punto 1)');

figure;
imshow(log(1 + abs(fftshift(ImagenConRuidoFrec))), []);
title('Imagen en frecuencias con ruido añadido (Punto 1)');

figure;
imshow(uint8(real(ImagenFiltradaWiener)));
title('Imagen filtrada con reducción de ruido (Filtro de Wiener) (Punto 1)');

figure;
imshow(log(1 + abs(fftshift(ImagenFrec))), []);
title('Imagen en frecuencias sin ruido (Punto 2)');

figure;
imshow(log(1 + abs(fftshift(FiltroFrec))), []);
title('Filtro en frecuencias (Punto 2)');

figure;
imshow(log(1 + abs(fftshift(ImagenFiltradaFrec))), []);
title('Imagen filtrada en frecuencias (Punto 2)');

figure;
imshow(uint8(ImagenFiltrada));
title('Imagen filtrada (Punto 2)');

figure;
imshow(log(1 + abs(fftshift(FiltroFrecInverso))), []);
title('Filtro Inverso en Frecuencias (Punto 2)');

figure;
imshow(log(1 + abs(fftshift(ProductoGHInverso))), []);
title('Producto G * H^{-1} en Frecuencias (Punto 2)');

figure;
imshow(uint8(ImagenProductoGHInverso));
title('Imagen Producto G * H^{-1} (Punto 2)');

figure;
imshow(uint8(real(ImagenFiltradaWienerDesenfoque)));
title('Imagen restaurada con Wiener (Punto 3)');