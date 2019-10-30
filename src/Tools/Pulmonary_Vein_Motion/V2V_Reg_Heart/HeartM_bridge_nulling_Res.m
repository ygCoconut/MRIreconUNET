clc
clear

%info = dicominfo('C:\ThirdParty\RawData\Grimaldi\IM_0403','dictionary','dicom-dict-philips.txt');
%[Yor] = dicomread(info);
%imshow(Yor,[]);

%[ImageMehdi,map] = dicomread('C:\ThirdParty\RawData\Grimaldi\IM_0538');
%figure
%imshow(Yor,[]);
%Dim = size(Yor);
%[X,Y,Z]=meshgrid(1:1:Dim(1,1),1:1:Dim(1,2),1:1:13);




V = get_image(763,112/2,'C:\ThirdParty\RawData\Grimaldi\');

[meanx,meany,meanz,meanx2,meany2,meanz2,meanxy,meanxz,meanyz] = Moment_Gen(V);

U(1,1) = meanx2 - meanx^2;
U(2,2) = meany2 - meany^2;
U(3,3) = meanz2 - meanz^2;

U(1,2) = meanxy - meanx*meany; 
U(1,3) = meanxz - meanx*meanz; 
U(2,3) = meanyz - meany*meanz; 

U(2,1) = U(1,2);
U(3,1) = U(1,3);
U(3,2) = U(2,3);

[v d] = eig(U);


%clear V1;
VV = zeros(size(V));
for iz = 1 : size(V,3)
    for iy = 1 : size(V,2)
        for ix = 1:size(V,1)
            temp = round(v'*([ix;iy;iz]-[meanx;meany;meanz])+[meanx;meany;meanz]);
            if ( temp(1,1) == 0)
                temp(1,1) = 1;
            end
            if ( temp(2,1) == 0)
                temp(2,1) = 1;
            end
            if ( temp(3,1) == 0)
                temp(3,1) = 1;
            end
          if ((temp(1,1)>0) && (temp(2,1)>0) && (temp(3,1)>0))
            VV(temp(1,1),temp(2,1),temp(3,1)) = V(ix,iy,iz);
          end
          
        end
    end
end

[fx,fy,fz] = getsign(VV,meanx,meany,meanz,50);

save mehdi.mat VV meanx meany meanz;
clear VV
clear V

V1 = get_image(557,120,'C:\ThirdParty\RawData\Grimaldi\');
[meanx1,meany1,meanz1,meanx21,meany21,meanz21,meanxy1,meanxz1,meanyz1] = Moment_Gen(V1);

U(1,1) = meanx21 - meanx1^2;
U(2,2) = meany21 - meany1^2;
U(3,3) = meanz21 - meanz1^2;

U(1,2) = meanxy1 - meanx1*meany1; 
U(1,3) = meanxz1 - meanx1*meanz1; 
U(2,3) = meanyz1 - meany1*meanz1; 

U(2,1) = U(1,2);
U(3,1) = U(1,3);
U(3,2) = U(2,3);

[v1 d1] = eig(U)

VV1 = zeros(size(V1));
for iz = 1 : size(V1,3)
    for iy = 1 : size(V1,2)
        for ix = 1:size(V1,1)
            temp = round(v1'*([ix;iy;iz]-[meanx1;meany1;meanz1])+[meanx1;meany1;meanz1]);
            if ( temp(1,1) == 0)
                temp(1,1) = 1;
            end
            if ( temp(2,1) == 0)
                temp(2,1) = 1;
            end
            if ( temp(3,1) == 0)
                temp(3,1) = 1;
            end
             if ((temp(1,1)>0) && (temp(2,1)>0) && (temp(3,1)>0))
            VV1(temp(1,1),temp(2,1),temp(3,1)) = V1(ix,iy,iz);
             end
        end
    end
end
[fx1,fy1,fz1] = getsign(VV1,meanx1,meany1,meanz1,10);

%VV1 = adjust_ref(VV1,fx,fy,fz,fx1,fy1,fz1);

save mehdi2.mat VV1 meanx1 meany1 meanz1;

% hold on
% plot(index_b-11,index_a-12,'r*');
% 
% C = normxcorr2(YorT1,Y);
% %M(i) = getframe;
% %PosM(:,i) = ginput(1); 
% [temp index_b] = max(max(C));
% [temp index_a] = max(max(C'));
% Pos1(1:2,No) = [index_a;index_b];
% plot(index_b-6,index_a-7,'g*');
% 
% C = normxcorr2(YorT2,Y);
% [temp index_b] = max(max(C));
% [temp index_a] = max(max(C'));
% Pos2(1:2,No) = [index_a;index_b];
% plot(index_b-10,index_a-7,'b*');
% %imshow(C,[]);
% %hold on
% %plot(index_b,index_a,'r*');
% M(i) = getframe;
%  end

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
 
     
     
     
