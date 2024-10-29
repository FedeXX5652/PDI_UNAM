% Cargar imagen sin ruido
ImagenSinRuido_Espacial = imread('Glaciar512.jpg'); 

if size(ImagenSinRuido_Espacial, 3) == 3
    ImagenSinRuido_Espacial = rgb2gray(ImagenSinRuido_Espacial);
end

% ----------------

% Transformada de Fourier
ImagenSinRuido_Frec = fft2(double(ImagenSinRuido_Espacial)); 

% Densidad espectral de la imagen original
PImagenSinRuido = abs(ImagenSinRuido_Frec).^2; 

% Generar ruido en el dominio de frecuencias (aumentar la varianza del ruido)
ImagenConRuido_Espacial = imnoise(ImagenSinRuido_Espacial, 'gaussian', 0.5);
ImagenConRuido_Frec = fft2(double(ImagenConRuido_Espacial));

% Calcular la nueva varianza del ruido, evitar 0
Sn = var(double(ImagenConRuido_Espacial(:))) + 1e-10;

% Densidad espectral de la imagen con ruido (Para filtro wiener)
PImagenConRuido = abs(ImagenConRuido_Frec).^2;

% Filtro de Wiener
WienerFiltro = PImagenSinRuido ./ (PImagenSinRuido + Sn); % Filtro de Wiener

% Aplicar el filtro Wiener a la imagen con ruido
ImagenFiltradaWiener_Frec = ImagenConRuido_Frec .* WienerFiltro;

% Transformada inversa para volver al dominio espacial
ImagenFiltradaWiener_Espacial = ifft2(ImagenFiltradaWiener_Frec);

% Tomar solo la parte real
ImagenFiltradaWiener_Espacial = real(ImagenFiltradaWiener_Espacial);
%Asegurar dentro de rango
ImagenFiltradaWiener_Espacial = uint8(255 * mat2gray(ImagenFiltradaWiener_Espacial));

% ----------------

%Punto 2
% G = F . H (En dominio de frecuencias)

% Filtro binomial 9x9 (Matriz H)
FiltroBinomial_Espacial = [1 8 28 56 70 56 28 8 1]' * [1 8 28 56 70 56 28 8 1];
FiltroBinomial_Espacial = FiltroBinomial_Espacial / sum(FiltroBinomial_Espacial(:)); % Normalizar

% Transformada de Fourier del filtro (Matriz H)
Filtro_Frec = fft2(FiltroBinomial_Espacial, 512, 512); % Padding del filtro

% Multiplicacion en dominio de frecuencias (Matriz G)
ImagenFiltrada_Frec = ImagenSinRuido_Frec .* Filtro_Frec; 

% Transformada inversa para volver al dominio espacial
ImagenFiltrada_Espacial = ifft2(ImagenFiltrada_Frec);

% Calcular la matriz inversa del filtro 
% en el dominio de la frecuencia (Matriz H^-1)
FiltroFrecInverso = zeros(size(Filtro_Frec));

for i = 1:numel(Filtro_Frec) % Evitar div entre 0
    if Filtro_Frec(i) ~= 0
        FiltroFrecInverso(i) = 1 / Filtro_Frec(i);
    end
end

% Producto de las matrices G y H^-1
ImagenFiltroPasoBajasHInversa_Frec = ImagenFiltrada_Frec .* FiltroFrecInverso; 

% Transformada inversa para volver al dominio espacial
ImagenFiltroPasoBajasHInversa_Espacial = ifft2(ImagenFiltroPasoBajasHInversa_Frec);

% ----------------

% Resultados
figure;
imshow(uint8(ImagenSinRuido_Espacial));
title('Imagen Sin Ruido');

figure;
imshow(uint8(ImagenConRuido_Espacial));
title('Imagen Con Ruido añadido');

figure;
imshow(log(1 + abs(fftshift(ImagenSinRuido_Frec))), []);
title('Imagen Sin Ruido en frecuencias (Punto 1)');

figure;
imshow(log(1 + abs(fftshift(ImagenConRuido_Frec))), []);
title('Imagen en frecuencias con ruido añadido (Punto 1)');

figure;
imshow(uint8(real(ImagenFiltradaWiener_Espacial)));
title('Imagen filtrada con reducción de ruido (Filtro de Wiener) (Punto 1)');

figure;
imshow(log(1 + abs(fftshift(ImagenSinRuido_Frec))), []);
title('Imagen en frecuencias sin ruido (Punto 2)');

figure;
imshow(log(1 + abs(fftshift(Filtro_Frec))), []);
title('Filtro en frecuencias (Punto 2)');

figure;
imshow(log(1 + abs(fftshift(ImagenFiltrada_Frec))), []);
title('Imagen filtrada en frecuencias (Punto 2)');

figure;
imshow(uint8(ImagenFiltrada_Espacial));
title('Imagen filtrada (Punto 2)');

figure;
imshow(log(1 + abs(fftshift(FiltroFrecInverso))), []);
title('Filtro Inverso en Frecuencias (Punto 2)');

figure;
imshow(log(1 + abs(fftshift(ImagenFiltroPasoBajasHInversa_Frec))), []);
title('Producto G * H^{-1} en Frecuencias (Punto 2)');

figure;
imshow(uint8(ImagenFiltroPasoBajasHInversa_Espacial));
title('Imagen Producto G * H^{-1} (Punto 2)');