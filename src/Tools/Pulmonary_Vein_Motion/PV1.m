%read the data 
clc
clear
info = dicominfo('D:\DICOM\00000004\IM_1802','dictionary','dicom-dict-philips.txt');
Yor = dicomread(info);
YorT = Yor(128:146,50:85);%85and 128, 85 146% 50 128% 50 126 82:92,94:105
YorT1 = Yor(128:146,50:85);
%Y1 = typecast(Y(:),'uint16');
%Y1 = reshape(Y1,512,512);
imshow(YorT,[]);
figure
imshow(Yor,[]);
C = normxcorr2(YorT,Yor);
[temp index_b] = max(max(C));
[temp index_a] = max(max(C'));
return

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
strM = 'D:\DICOM\00000004\';

 for i = 1800 : 1900-1
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
 
 
C = normxcorr2(YorT,Y);
[temp index_b] = max(max(C));
[temp index_a] = max(max(C'));
Pos(1:2,i) = [index_a;index_b];

%C = normxcorr2(YorT1,Y);
imshow(Y);
M(i) = getframe;

YorT = Y(index_a-18:index_a,index_b-35:index_b);

C = normxcorr2(YorT1,Y);
[temp index_b] = max(max(C));
[temp index_a] = max(max(C'));



Pos1(1:2,i) = [index_a;index_b]-Pos1(1:2,1);
 
 end    
