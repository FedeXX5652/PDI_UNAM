function filtro = GeneradorFiltroPasoBajasBloque(N)
    filtro = ones(N) / (N * N); % Filtro normalizado
end

function imagen_filtrada = imageConvolver(imagen, filtro)
    imagen_filtrada = imfilter(imagen, filtro, 'replicate'); 
end

ImagenSinRuido = imread('Glaciar.jpg'); 

% 1: Restauracion con Filtro de Wiener de imagen con ruido

ImagenConRuido = imnoise(ImagenSinRuido, 'gaussian', 0.25, 0.01); % Ruido
ImagenRestauradaRuido = wiener2(ImagenConRuido, [5 5]); % Filtro de wiener

% 2: Restauracion con Filtro de Wiener con imagen con perdida de nitidez

filtroPasoBajas = GeneradorFiltroPasoBajasBloque(9);
ImagenBorrosa = imageConvolver(ImagenSinRuido, filtroPasoBajas);
ImagenRestauradaPasoBajas = wiener2(ImagenBorrosa, [5 5]);

% 3: Restauracion (Primero ruido y despues filtro)

Imagen_Ruido_y_Filtro = imageConvolver(ImagenConRuido, filtroPasoBajas);
ImagenRestauradaRuido_y_Filtro = wiener2(Imagen_Ruido_y_Filtro, [5 5]);

% 4: Restauración (Primero filtro y despues ruido)

Imagen_Filtro_y_Ruido = imnoise(ImagenBorrosa, 'gaussian', 0.25, 0.01);
ImagenRestauradaFiltro_y_Ruido = wiener2(Imagen_Filtro_y_Ruido, [5 5]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;
imshow(ImagenSinRuido); % Original
title('Imagen Original');

figure;
imshow(ImagenConRuido); % Con ruido
title('Imagen Original con Ruido');

% 1

figure;
imshow(ImagenRestauradaRuido);
title('Imagen con ruido tras filtro de Wiener');

% 2

figure;
imshow(ImagenBorrosa);
title('Imagen con perdida de nitidez (Filtro paso bajas 9x9)');

figure;
imshow(ImagenRestauradaPasoBajas);
title('Imagen con perdida de nitidez tras filtro de Wiener');

% 3

figure;
imshow(Imagen_Ruido_y_Filtro);
title('Imagen con primero ruido y despues perdida de nitidez');

figure;
imshow(ImagenRestauradaRuido_y_Filtro);
title('Imagen Restaurada (Primero ruido y después perdida de nitidez)');

% 4

figure;
imshow(Imagen_Filtro_y_Ruido);
title('Imagen con primero perdida de nitidez y despues ruido');

figure;
imshow(ImagenRestauradaFiltro_y_Ruido);
title('Imagen Restaurada (Primero perdida de nitidez y después ruido)');