function [V aR aT spacing info AngM OrM] = get_image(n1,n2,Filename,cut)

%Filename = 'C:\ThirdParty\RawData\Grimaldi\';
No = 0;

for i = n1 : n1 + n2 -1  %557--- 557 + 120 - 1  %  
    if ( i <10 )
        File_Name = strcat(Filename,'IM_000');
    elseif (i < 100)
        File_Name = strcat(Filename,'IM_00');
   
    elseif (i < 1000)
        File_Name = strcat(Filename,'IM_0');
    else
        File_Name = strcat(Filename,'IM_');
    end
% % 
File_Name = strcat(File_Name,num2str(i));

%Im = dicomread(info);
No = No +1;

info = dicominfo(File_Name,'dictionary','dicom-dict-philips.txt');
aR = info.ImageOrientationPatient;
aT(:,No) = info.ImagePositionPatient;
AngM(:,No)=[info.Private_2005_1000;info.Private_2005_1001;info.Private_2005_1002];
OrM(:,No)=[info.Private_2005_1008;info.Private_2005_1009;info.Private_2005_100a];
Im = dicomread(File_Name);
if (cut==1)
 V(:,:,No) = (Im(128:size(Im,1)-128,128:size(Im,2)-128));   
else
V(:,:,No) = (Im); 
end
imshow(V(:,:,No),[]);
MM(i) = getframe;
end
clear MM;
spacing(1:3,1)=[info.PixelSpacing;info.SpacingBetweenSlices];
return