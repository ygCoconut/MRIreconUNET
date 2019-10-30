
close all
%clear all

load test_ims
ims = cat(4, ims,ims);
ims = abs(ims);

ims = uint16(ims);
% imagescn(ims(:,:,:),[] , [ 3 3] , 10, 3);
% colormap(gray)
% axis('equal');axis('tight'); 

imagescn(single(cat(4, ims(:,:,:), ims(:,:,:), ims(:,:,:), ims(:,:,:), ims(:,:,:), ims(:,:,:))) ,[] , [2 3] , 10, 3);
colormap(gray)
axis('equal');axis('tight'); 


%figure
%i = imagescn(Im, [], [ 2 3]);
%axis('equal'); axis('tight'); 
%ROI_tool;
