% Cargar imagen sin ruido
ImagenSinRuido = imread('Glaciar512.jpg'); 

if size(ImagenSinRuido, 3) == 3
    ImagenSinRuido = rgb2gray(ImagenSinRuido);
end

ImagenConRuido = imnoise(ImagenSinRuido, 'gaussian', 0.25, 0.01);

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

%%%%%%%%%%%%%%%

% Resultados
figure;
imshow(uint8(ImagenSinRuido));
title('Imagen sin Ruido');

% Mostrar magnitud de la imagen en frecuencia
figure;
imshow(log(1 + abs(fftshift(ImagenFrec))), []);
title('Imagen en frecuencias');

% Mostrar magnitud del filtro en frecuencia
figure;
imshow(log(1 + abs(fftshift(FiltroFrec))), []);
title('Filtro en frecuencias');

% Mostrar magnitud de la imagen filtrada en frecuencias
figure;
imshow(log(1 + abs(fftshift(ImagenFiltradaFrec))), []);
title('Imagen filtrada en frecuencias');

% Mostrar imagen filtrada en el dominio espacial
figure;
imshow(uint8(ImagenFiltrada));
title('Imagen filtrada');

% Mostrar la magnitud del filtro inverso en frecuencia
figure;
imshow(log(1 + abs(fftshift(FiltroFrecInverso))), []);
title('Filtro Inverso en Frecuencias');

% Mostrar magnitud del producto G * H^-1 en frecuencia
figure;
imshow(log(1 + abs(fftshift(ProductoGHInverso))), []);
title('Producto G * H^{-1} en Frecuencias');

% Mostrar imagen del producto G * H^-1 en el dominio espacial
figure;
imshow(uint8(ImagenProductoGHInverso)); % Mostrar la parte real de la imagen resultante
title('Imagen Producto G * H^{-1}');