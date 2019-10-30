clc
clear

[V aR aT spacing info] = get_image(1157,33,'C:\ThirdParty\RawData\Phantom_mehdi\',0);%1157%1123%1089%1157%1088-33
Ra = [aR(1:3),aR(4:6)];
tempM = cross(aR(1:3),aR(4:6));
%Ra = [tempM,aR(4:6),aR(1:3)];
Ra = [Ra,tempM];
Rb=[aR(1:3),cross(aR(4:6),aR(1:3)),aR(4:6)]
% [Vi bR bT spacing1 info1] = get_image(999,17,'C:\ThirdParty\RawData\Phantom_mehdi\',0);%729
% Rb = [bR(1:3),bR(4:6)];
% tempM = cross(bR(1:3),bR(4:6));
% Rb = [Rb,tempM];

%Rb =[0.6392   -0.4094   -0.6510;0.7135         0    0.7006;-0.2868   -0.9124    0.2921];

%size(V1)
%Rb=eye(3)

 %[meanx,meany,meanz,meanx2,meany2,meanz2,meanxy,meanxz,meanyz] = Moment_gen(V);

% V1 = V;
% clear V
% 
% [meanx1,meany1,meanz1,meanx21,meany21,meanz21,meanxy1,meanxz1,meanyz1] = Moment_gen(V1,spacing);

% save mehdi_delete_test V
%clear V1
%clear V1;
%VV = ones(size(V1));
%VV=zeros(460,460,320);
No =1;
%%%%%%%%%%%%%%%%%%%%%%%%%calculate Center%%%%%%%%%%%%%%%%%%%%%
Cx = (double(spacing(1,1)*info.Height)/2) + aT(1,1)
Cy = (double(spacing(2,1)*info.Width)/2) + aT(2,1)
Cz = (info.SpacingBetweenSlices)*(size(V,3)-1)/2+aT(3,1)

CCx = -(double(spacing(1,1)*info.Height)/2)
CCy = -(double(spacing(1,1)*info.Width)/2)
CCz = -(info.SpacingBetweenSlices)*(size(V,3)-1)/2
C = aT(:,1)-Rb*[CCx;CCy;CCz]
return



for iz = 1 : 2%size(V,3)
     for iy = 1 : 1%round(sqrt(1)*size(V,2))
         for ix = 1:1%round(sqrt(1)*size(V,1))
             
%              temp(1,1) = (ix-1)*spacing(1,1) + aT(1,iz);
%              temp(2,1) = (iy-1)*spacing(2,1) + aT(2,iz);
%              temp(3,1) = (iz-1)*spacing(3,1)/2 + aT(3,iz);

             temp(1,1) = (ix-1)*spacing(1,1) + aT(1,1);
             temp(2,1) = (iy-1)*spacing(2,1) + aT(2,1);
             temp(3,1) = (iz-1)*info.SpacingBetweenSlices + aT(3,1);
         
             %temp1 = Rb'*(temp-[meanx1;meany1;meanz1])+[meanx1;meany1;meanz1];
             temp1 = Rb'*(temp-[Cx;Cy;Cz])+[Cx;Cy;Cz]
             
             temp1(1,1) = ceil((temp1(1,1)-aT(1,1))/spacing(1,1))+1;
             temp1(2,1) = ceil((temp1(2,1)-aT(2,1))/spacing(2,1))+1;
             temp1(3,1) = ceil((temp1(3,1)-aT(3,1))/(info.SpacingBetweenSlices))+1;
             
             
                         
             %VV(ix,iy,iz) = 
% %          
%            if ( temp1(1,1) == 0)
%                temp1(1,1) = 1;
%            end
% % %              
%           if ( temp1(2,1) == 0)
%               temp1(2,1) = 1;
%           end
%           if ( temp1(3,1) == 0)
%               temp1(3,1) = 1;
%           end
         
         %bias = min(min(temp);
          if ((temp1(1,1)>0 && temp1(1,1)<=size(V,1)) && (temp1(2,1)>0&& temp1(2,1)<=size(V,2)) && (temp1(3,1)>0&& temp1(3,1)<=size(V,3)))
              %VV(temp1(1,1),temp1(2,1),temp1(3,1)) = V(ix,iy,iz);
              VV(ix,iy,iz) = V(temp1(1,1),temp1(2,1),temp1(3,1));
          end
       % VV(temp1(1,1),temp1(2,1),temp1(3,1)) = V(ix,iy,iz);
         end
     end
     iz
 end
% 
% [fx,fy,fz] = getsign(VV,meanx,meany,meanz,50);

