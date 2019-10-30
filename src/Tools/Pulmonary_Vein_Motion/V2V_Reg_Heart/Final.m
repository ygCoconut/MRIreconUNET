clc
clear

% [V aR aT spacing info] = get_image(557,120,'C:\ThirdParty\RawData\Grimaldi\',0);
% Ra = [aR(1:3),aR(4:6)];
% tempM = cross(aR(1:3),aR(4:6));
% %Ra = [tempM,aR(4:6),aR(1:3)];
% Ra = [Ra,tempM];


[V aR aT spacing info] = get_image(557,120,'C:\ThirdParty\RawData\Grimaldi\',0);
Ra = [aR(1:3),aR(4:6)];
tempM = cross(aR(1:3),aR(4:6));
%Ra = [tempM,aR(4:6),aR(1:3)];
Ra = [Ra,tempM];

[Vi bR bT spacing1 info] = get_image(763,112/2,'C:\ThirdParty\RawData\Grimaldi\',0);
tempM = cross(-aR(4:6),aR(1:3));
Rb = [bR(1:3),bR(4:6),tempM];
% Rb = [bR(1:3),bR(4:6)];
% tempM = cross(bR(1:3),bR(4:6));
% %Rb = [tempM,bR(4:6),bR(1:3)];
% Rb = [Rb,tempM];

%Rb =[0.6392   -0.4094   -0.6510;0.7135         0    0.7006;-0.2868   -0.9124    0.2921];
%Rb =[0.6392   -0.4094   -0.6510;0.7135         0    0.7006;-0.2868   -0.9124    0.2921];


%size(V1)


 %[meanx,meany,meanz,meanx2,meany2,meanz2,meanxy,meanxz,meanyz] = Moment_gen(V);

V1 = V;
clear V Vi

%[meanx1,meany1,meanz1,meanx21,meany21,meanz21,meanxy1,meanxz1,meanyz1] = Moment_gen(V1,spacing);

% save mehdi_delete_test V
%clear V1
%clear V1;
%VV = ones(size(V1));
%VV=zeros(460,460,320);
%No =1;
%%%%%%%%%%%%%%%%%%%%%%%%%calculate Center%%%%%%%%%%%%%%%%%%%%%
CCx = -(double(spacing(1,1)*info.Height)/2);
CCy = -(double(spacing(1,1)*info.Width)/2);
CCz = -(info.SpacingBetweenSlices)*(size(V1,3)-1)/2;
C = aT(:,1)-Ra*[CCx;CCy;CCz];


% [Vi bR bT spacing1 info] = get_image(763,112/2,'C:\ThirdParty\RawData\Grimaldi\',0);
% Rb = [bR(1:3),bR(4:6)];
% tempM = cross(bR(1:3),bR(4:6));
% Rb = [Rb,tempM];
% 
% CCx = -(double(spacing1(1,1)*info.Height)/2);
% CCy = -(double(spacing1(1,1)*info.Width)/2);
% CCz = -(info.SpacingBetweenSlices)*(size(Vi,3)-1)/2;
% C1 = bT(:,1)-Rb*[CCx;CCy;CCz];

C1 = [21.7374  -93.8208    7.9585]';

for iz = 1 : 1*size(V1,3)
     for iy = 1 : round(sqrt(1)*size(V1,2))
         for ix = 1: round(sqrt(1)*size(V1,1))
             
             temp(1,1) = CCx + (ix-1)*spacing(1,1);
             temp(2,1) = CCy + (iy-1)*spacing(2,1);
             temp(3,1) = CCz + (iz-1)*info.SpacingBetweenSlices;
             
%              temp(1,1) = (ix-1)*spacing(1,1) + CCx;
%              temp(2,1) = (iy-1)*spacing(2,1) + CCy;
%              temp(3,1) = (iz-1)*info.SpacingBetweenSlices + CCz;
         
             temp1 = Rb*(temp)+C1;
             
%            temp1(1,1) = ceil((temp1(1,1)-CCx)/spacing(1,1))+1;
%            temp1(2,1) = ceil((temp1(2,1)-CCy)/spacing(2,1))+1;
%            temp1(3,1) = ceil((temp1(3,1)-CCz)/(info.SpacingBetweenSlices))+1;
             temp1(1,1) = ceil((temp1(1,1)-CCx)/spacing(1,1))+1;
             temp1(2,1) = ceil((temp1(2,1)-CCy)/spacing(2,1))+1;
             temp1(3,1) = ceil((temp1(3,1)-CCz)/(info.SpacingBetweenSlices))+1;
             
             
                         
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
             %VV(ix,iy,iz) = V1(temp1(1,1),temp1(2,1),temp1(3,1));
             VV(temp1(1,1),temp1(2,1),temp1(3,1)) = V1(ix,iy,iz);
         end
      
         end
     end
     iz
 end
% 
% [fx,fy,fz] = getsign(VV,meanx,meany,meanz,50);


