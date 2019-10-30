%read the data 
clc
clear
info = dicominfo('C:\ThirdParty\RawData\Phantom_mehdi\IM_0637','dictionary','dicom-dict-philips.txt');
Yor = dicomread(info);
imshow(Yor,[]);
%info.ImagePositionPatient
%info.ImageOrientationPatient
V1 = double(Yor);

info = dicominfo('C:\ThirdParty\RawData\Phantom_mehdi\IM_0635','dictionary','dicom-dict-philips.txt');
Yor = dicomread(info);
imshow(Yor,[]);
%info.ImagePositionPatient
%info.ImageOrientationPatient
V2 = double(Yor);
figure




%File_Name = 'IM';
%Y2 = Y1(136:200,135:216);
% 
% Y2 = typecast(Y1,'uint16'); 
%5C=normxcorr2(Y2,Y1);
strM = 'C:\ThirdParty\RawData\Phantom_mehdi\';
No = 0;
for i = 891 : 891 + 17-1
     if ( i <10 )
         File_Name = strcat(strM,'IM_000');
     elseif (i < 100)
         File_Name = strcat(strM,'IM_00');
     elseif (i < 1000)
         File_Name = strcat(strM,'IM_0');
     else
         File_Name = strcat(strM,'IM_');;
     end
% 
File_Name = strcat(File_Name,num2str(i));
info = dicominfo(File_Name,'dictionary','dicom-dict-philips.txt');



 Y = dicomread(info);
 if(info.ImagePositionPatient(3,1)==0)
    VVV=Y;
end
No = No +1;

aT(:,No)=info.ImagePositionPatient;
bT=info.ImageOrientationPatient;

V(:,:,No) = double(Y); 

imshow(V(:,:,No),[]);
MM(i) = getframe;
end    
spacing=info.PixelSpacing;
Cx = (double(spacing(1,1)*info.Height)/2) + aT(1,1)
Cy = (double(spacing(2,1)*info.Width)/2) + aT(2,1)
Cz = (info.SpacingBetweenSlices)*(size(V,3)-1)/2+aT(3,1) 