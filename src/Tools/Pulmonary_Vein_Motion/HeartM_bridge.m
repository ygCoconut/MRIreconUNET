clc
clear

info = dicominfo('C:\Program Files\MATLAB71\work\Pulmonary_Vein_Motion\Images\Bridge\IM_0001','dictionary','dicom-dict-philips.txt');
Yor = dicomread(info);

YorT = Yor(142:173,64:74);

YorT1 = Yor(167:188,60:80);

YorT2 = Yor(167:188,84:99);

imshow(Yor,[]);

figure
imshow(YorT,[]);

figure
imshow(YorT1,[]);

figure
imshow(YorT2,[]);


C = normxcorr2(YorT,Yor);
[temp index_b] = max(max(C));
[temp index_a] = max(max(C'));
%return

Pos(1:2,1) = [index_a;index_b];

%C = normxcorr2(YorT1,Yor);
%[temp index_b] = max(max(C));
%[temp index_a] = max(max(C'));
%Pos1(1:2,1) = [index_a;index_b];


%File_Name = 'IM';
%Y2 = Y1(136:200,135:216);
% 
% Y2 = typecast(Y1,'uint16'); 
%5C=normxcorr2(Y2,Y1);

 for i = 1 : 200
     if ( i <10 )
         File_Name = 'C:\Program Files\MATLAB71\work\Pulmonary_Vein_Motion\Images\Bridge\IM_000';
     elseif (i < 100)
         File_Name = 'C:\Program Files\MATLAB71\work\Pulmonary_Vein_Motion\Images\Bridge\IM_00';
     elseif (i < 1000)
         File_Name = 'C:\Program Files\MATLAB71\work\Pulmonary_Vein_Motion\Images\Bridge\IM_0';
     else
         File_Name = 'C:\Program Files\MATLAB71\work\Pulmonary_Vein_Motion\Images\Bridge\IM_';
     end
% 
File_Name = strcat(File_Name,num2str(i));
info = dicominfo(File_Name,'dictionary','dicom-dict-philips.txt');
Y = dicomread(info);
 
C = normxcorr2(YorT,Y);
[temp index_b] = max(max(C));
[temp index_a] = max(max(C'));
Pos(1:2,i) = [index_a;index_b];
imshow(Y,[]);
hold on
plot(index_b,index_a,'r*');

C = normxcorr2(YorT1,Y);
%M(i) = getframe;
%PosM(:,i) = ginput(1); 
[temp index_b] = max(max(C));
[temp index_a] = max(max(C'));
Pos1(1:2,i) = [index_a;index_b];
plot(index_b,index_a,'k*');

C = normxcorr2(YorT2,Y);
[temp index_b] = max(max(C));
[temp index_a] = max(max(C'));
Pos2(1:2,i) = [index_a;index_b];
plot(index_b,index_a,'b*');
M(i) = getframe;
 
end    