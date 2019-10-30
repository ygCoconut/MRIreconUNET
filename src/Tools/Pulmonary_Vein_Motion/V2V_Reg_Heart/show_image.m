clc
clear all
load mehdi.mat
load mehdi2.mat



for i = 1:30
    
imshow(squeeze(VV(:,:,i)),[])

figure

imshow(squeeze(VV1(:,:,i)),[])
figure

MM(i)=getframe;
end
