%read the data 
clc
clear
info = dicominfo('Y:\USERS\Mehdi\DICOM_Bridge\DICOM\IM_0001','dictionary','dicom-dict-philips.txt');
Yor = dicomread(info);
YorT = Yor(128:146,50:85);%85and 128, 85 146% 50 128% 50 126 
%YorT1 = Yor(80:110,74:105);
YorT1 = Yor(115:140,95:120);
%Y1 = typecast(Y(:),'uint16');
%Y1 = reshape(Y1,512,512);
imshow(Yor,[]);

figure
imshow(YorT1,[]);
figure
C = normxcorr2(YorT,Yor);
[temp index_b] = max(max(C));
[temp index_a] = max(max(C'));
%return

Pos(1:2,1) = [index_a;index_b];

C = normxcorr2(YorT1,Yor);
[temp index_b] = max(max(C));
[temp index_a] = max(max(C'));
Pos1(1:2,1) = [index_a;index_b];


%File_Name = 'IM';
%Y2 = Y1(136:200,135:216);
% 
% Y2 = typecast(Y1,'uint16'); 
%5C=normxcorr2(Y2,Y1);

 for i = 1 : 200
     if ( i <10 )
         File_Name = 'Y:\USERS\Mehdi\DICOM_Bridge\DICOM\IM_000';
     elseif (i < 100)
         File_Name = 'Y:\USERS\Mehdi\DICOM_Bridge\DICOM\IM_00';
     elseif (i < 1000)
         File_Name = 'Y:\USERS\Mehdi\DICOM_Bridge\DICOM\IM_0';
     else
         File_Name = 'Y:\USERS\Mehdi\DICOM_Bridge\DICOM\IM_';
     end
% 
 File_Name = strcat(File_Name,num2str(i));
 info = dicominfo(File_Name,'dictionary','dicom-dict-philips.txt');
 Y = dicomread(info);
 
 
C = normxcorr2(YorT,Y);
[temp index_b] = max(max(C));
[temp index_a] = max(max(C'));
Pos(1:2,i) = [index_a;index_b];

C = normxcorr2(YorT1,Y);
imshow(Y,[]);
M(i) = getframe;
%PosM(:,i) = ginput(1); 
[temp index_b] = max(max(C));
[temp index_a] = max(max(C'));



Pos1(1:2,i) = [index_a;index_b];
 
end    
