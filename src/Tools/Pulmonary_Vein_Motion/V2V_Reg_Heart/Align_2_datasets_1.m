clc
clear

[V aR aT] = get_image(557,120,'C:\ThirdParty\RawData\Grimaldi\',0);
Ra = [aR(1:3),aR(4:6)];
tempM = cross(aR(1:3),aR(4:6));
Ra = [tempM,aR(4:6),aR(1:3)];
%Ra = [Ra,tempM];

% [V1 bR bT] = get_image(763,112/2,'C:\ThirdParty\RawData\Grimaldi\',1);
% Rb = [bR(1:3),bR(4:6)];
% tempM = cross(bR(1:3),bR(4:6));
% Rb = [tempM,bR(4:6),bR(1:3)];
%Rb = [Rb,tempM];

Rb =[0.6392   -0.4094   -0.6510;0.7135         0    0.7006;-0.2868   -0.9124    0.2921];

%size(V1)


 %[meanx,meany,meanz,meanx2,meany2,meanz2,meanxy,meanxz,meanyz] = Moment_gen(V);

V1 = V;
clear V
[meanx1,meany1,meanz1,meanx21,meany21,meanz21,meanxy1,meanxz1,meanyz1] = Moment_gen(V1);

% save mehdi_delete_test V
%clear V1
%clear V1;
%VV = ones(size(V1));
VV=zeros(490,350,480);
No =1;
for iz = 1 : size(V1,3)
     for iy = 1 : size(V1,2)
         for ix = 1:size(V1,1)
         
         temp = round(Rb'*([ix;iy;iz]-[meanx1;meany1;meanz1])+[meanx1;meany1;meanz1]);
% %          
%           if ( temp(1,1) <= 0)
%               temp(1,1) = 1;
%           end
% %              
%           if ( temp(2,1) <= 0)
%               temp(2,1) = 1;
%           end
%           if ( temp(3,1) <= 0)
%               temp(3,1) = 1;
%           end
%          
         %bias = min(min(temp);
         
      
         VV(temp(1,1)+60,temp(2,1)+20,temp(3,1)+170) = V1(ix,iy,iz);
         %VV(ix,iy,iz) = V1(temp(1,1),temp(2,1),temp(3,1)); 
         %VV(ix(1,:),ix(2,:),ix(3,:)) = V1(ix(1,:),ix(2,:),ix(3,:));
              %V1(ix,iy,iz)
%              end
         end
     end
     iz
 end
% 
% [fx,fy,fz] = getsign(VV,meanx,meany,meanz,50);

