clc
clear

[V1 bR bT spacing] = get_image(763,112/2,'C:\ThirdParty\RawData\Grimaldi\',0);
Rb = [bR(1:3),bR(4:6)];
tempM = cross(bR(1:3),bR(4:6));
%Rb = [tempM,bR(4:6),bR(1:3)];
Rb = [Rb,tempM];

[meanx1,meany1,meanz1,meanx21,meany21,meanz21,meanxy1,meanxz1,meanyz1] = Moment_gen(V1,spacing);

U(1,1) = meanx21 - meanx1^2;
U(2,2) = meany21 - meany1^2;
U(3,3) = meanz21 - meanz1^2;

U(1,2) = meanxy1 - meanx1*meany1; 
U(1,3) = meanxz1 - meanx1*meanz1; 
U(2,3) = meanyz1 - meany1*meanz1; 

U(2,1) = U(1,2);
U(3,1) = U(1,3);
U(3,2) = U(2,3);

[v1 d1] = eig(U);

save Mehdi_Test.mat v1 d1 V1 meanx1 meany1 meanz1

clear all
clc

load Mehdi_Test v1 meanx1 meany1 meanz1

[V aR aT spacing] = get_image(557,120,'C:\ThirdParty\RawData\Grimaldi\',0);
Ra = [aR(1:3),aR(4:6)];
tempM = cross(aR(1:3),aR(4:6));
%Ra = [tempM,aR(4:6),aR(1:3)];
Ra = [Ra,tempM];

[meanx,meany,meanz,meanx2,meany2,meanz2,meanxy,meanxz,meanyz] = Moment_gen(V,spacing);

D_origin(1,1) = meanx1 - meanx;
D_origin(2,1) = meany1 - meany;
D_origin(3,1) = meanz1 - meanz;

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
        for ix = 1: size(V,1)
            
             temp(1,1) = (ix-1)*spacing(1,1);
             temp(2,1) = (iy-1)*spacing(2,1);
             temp(3,1) = (iz-1)*spacing(3,1);
         
             temp1 = inv(v1*v')*(temp-[meanx1;meany1;meanz1])+[meanx;meany;meanz];
             
             temp1(1,1) = ceil(temp1(1,1)/spacing(1,1))+1;
             temp1(2,1) = ceil(temp1(2,1)/spacing(2,1))+1;
             temp1(3,1) = ceil(temp1(3,1)/spacing(3,1))+1;
            
            if ((temp1(1,1)>0 && temp1(1,1)<=size(V,1)) && (temp1(2,1)>0&& temp1(2,1)<=size(V,2)) && (temp1(3,1)>0&& temp1(3,1)<=size(V,3)))
             VV(ix,iy,iz) = V(temp1(1,1),temp1(2,1),temp1(3,1));
            end
            
        end
    end
end


