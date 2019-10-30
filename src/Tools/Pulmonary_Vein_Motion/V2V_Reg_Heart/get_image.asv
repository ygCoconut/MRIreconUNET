function [V aR aT spacing info] = get_image(n1,n2,Filename,cut)

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
info = dicominfo(File_Name,'dictionary','dicom-dict-philips.txt');

%Im = dicomread(info);
No = No +1;

aT(:,No)=info.ImagePositionPatient;
Image_Plane = info.Private_2001_100b;
Im = dicomread(File_Name);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Image Plane Configurations
if(Image_Plane(1)=='T')
    if (cut==1)
        V(:,:,No) = double(Im(128:size(Im,1)-128,128:size(Im,2)-128));   
    else
        V(:,:,No) = double(Im); 
    end
elseif(Image_Plane(1)=='C')
    if (cut==1)
        V(:,:,No) = double(Im(128:size(Im,1)-128,128:size(Im,2)-128));   
    else
        V(:,No,:) = double(Im); 
    end
else
    if (cut==1)
        V(No,:,:) = double(Im(128:size(Im,1)-128,128:size(Im,2)-128));   
    else
        V(No,:,:) = double(Im); 
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Spacing Correction and Finding the Centre
imshow(squeeze(V(:,No,:)),[]);
MM(i) = getframe;
end
clear MM;

info = dicominfo(File_Name,'dictionary','dicom-dict-philips.txt');
aR = info.ImageOrientationPatient;
%aT = info.ImagePositionPatient;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Centre of Volume
return