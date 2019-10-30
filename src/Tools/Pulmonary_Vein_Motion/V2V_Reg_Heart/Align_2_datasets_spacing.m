clc
clear

[V aR aT spacing info] = get_image(557,120,'C:\ThirdParty\RawData\Grimaldi\',0);
Ra = [aR(1:3),aR(4:6)];
tempM = cross(aR(1:3),aR(4:6));
%Ra = [tempM,aR(4:6),aR(1:3)];
Ra = [Ra,tempM];

%   [V1 bR bT spacing info] = get_image(763,112/2,'C:\ThirdParty\RawData\Grimaldi\',1);
%   Rb = [bR(1:3),bR(4:6)];
%   tempM = cross(bR(1:3),bR(4:6));
% %  Rb = [tempM,bR(4:6),bR(1:3)];
%   Rb = [Rb,tempM];
%return

Rb =[0.6392   -0.4094   -0.6510;0.7135         0    0.7006;-0.2868   -0.9124    0.2921];

%size(V1)


 %[meanx,meany,meanz,meanx2,meany2,meanz2,meanxy,meanxz,meanyz] = Moment_gen(V);

V1 = V;
clear V
[meanx1,meany1,meanz1,meanx21,meany21,meanz21,meanxy1,meanxz1,meanyz1] = Moment_gen(V1,spacing);

% save mehdi_delete_test V
%clear V1
%clear V1;
%VV = ones(size(V1));
VV=zeros(460,460,320);
No =1;
for iz = 1 : 2*size(V1,3)
     for iy = 1 : round(sqrt(2)*size(V1,2))
         for ix = 1: round(sqrt(2)*size(V1,1))
             
%              temp(1,1) = (ix-1)*spacing(1,1) + aT(1,iz);
%              temp(2,1) = (iy-1)*spacing(2,1) + aT(2,iz);
%              temp(3,1) = (iz-1)*spacing(3,1)/2 + aT(3,iz);

              temp(1,1) = (ix-1)*spacing(1,1) -133;
              temp(2,1) = (iy-1)*spacing(2,1) -210;
%              temp(1,1) = (ix-1)*spacing(1,1) -200;
%              temp(2,1) = (iy-1)*spacing(2,1) -200;
             temp(3,1) = (iz-1)*(spacing(3,1)/2)-48;
         
             %temp1 = Rb'*(temp-[meanx1;meany1;meanz1])+[meanx1;meany1;meanz1];
             temp1 = Rb*(temp);
             
              temp1(1,1) = ceil((temp1(1,1)+133)/spacing(1,1))+1;
              temp1(2,1) = ceil((temp1(2,1)+210)/spacing(2,1))+1;
             
             temp1(3,1) = ceil((temp1(3,1)+48)/(spacing(3,1)/2))+1;
             
             
                         
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
         if ((temp1(1,1)>0 && temp1(1,1)<=size(V1,1)) && (temp1(2,1)>0&& temp1(2,1)<=size(V1,2)) && (temp1(3,1)>0&& temp1(3,1)<=size(V1,3)))
             VV(ix,iy,iz) = V1(temp1(1,1),temp1(2,1),temp1(3,1));
         end
      
         end
     end
     iz
 end
% 
% [fx,fy,fz] = getsign(VV,meanx,meany,meanz,50);


