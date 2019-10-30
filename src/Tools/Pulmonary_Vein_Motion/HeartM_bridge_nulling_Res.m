clc
clear

info = dicominfo('C:\Program Files\MATLAB71\work\Pulmonary_Vein_Motion\Images\Bridge\IM_0403','dictionary','dicom-dict-philips.txt');
Yor = dicomread(info);

YorT = Yor(105:132,45:77);

YorT1 = Yor(57:73,140:156);

YorT2 = Yor(53:79,80:96);

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
No = 0;
 for i = 403 : 602
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
No = No +1; 
C = normxcorr2(YorT,Y);
[temp index_b] = max(max(C));
[temp index_a] = max(max(C'));
Pos(1:2,No) = [index_a;index_b];
imshow(Y,[]);
hold on
plot(index_b-11,index_a-12,'r*');

C = normxcorr2(YorT1,Y);
%M(i) = getframe;
%PosM(:,i) = ginput(1); 
[temp index_b] = max(max(C));
[temp index_a] = max(max(C'));
Pos1(1:2,No) = [index_a;index_b];
plot(index_b-6,index_a-7,'g*');

C = normxcorr2(YorT2,Y);
[temp index_b] = max(max(C));
[temp index_a] = max(max(C'));
Pos2(1:2,No) = [index_a;index_b];
plot(index_b-10,index_a-7,'b*');
%imshow(C,[]);
%hold on
%plot(index_b,index_a,'r*');
M(i) = getframe;
 end

%  for i = 1: 200
%      
%      if ( i <10 )
%          File_Name = 'C:\Program Files\MATLAB71\work\Pulmonary_Vein_Motion\Images\Bridge\IM_000';
%      elseif (i < 100)
%          File_Name = 'C:\Program Files\MATLAB71\work\Pulmonary_Vein_Motion\Images\Bridge\IM_00';
%      elseif (i < 1000)
%          File_Name = 'C:\Program Files\MATLAB71\work\Pulmonary_Vein_Motion\Images\Bridge\IM_0';
%      else
%          File_Name = 'C:\Program Files\MATLAB71\work\Pulmonary_Vein_Motion\Images\Bridge\IM_';
%      end
%      
%     File_Name = strcat(File_Name,num2str(i));
%     info = dicominfo(File_Name,'dictionary','dicom-dict-philips.txt');
%     Y = dicomread(info);
%     
%     YY = [zeros(50,324);zeros(224,50),Y,zeros(224,50);zeros(50,324)];
%     FYY = ifftn(YY);
%     for k = 1 : size(YY,2)
%         FYYS(k,:) = exp(j*2*pi*k/size(YY,2)*(Pos(1,i)-Pos(1,1)))*FYY(k,:);
%     end
%     FYYSF = fftn(FYYS);
%     imshow(abs(FYYSF),[]);
%     MM(i) = getframe;
%  end
%  
 
     
     
     
