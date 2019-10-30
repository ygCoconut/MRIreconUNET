function [meanx,meany,meanz,meanx2,meany2,meanz2,meanxy,meanxz,meanyz,signx,signy,signz]=Moment_gen(V,spacing)

sumo = 0;
sumx = 0;
sumy = 0;
sumz = 0;

sumx2 = 0;
sumy2 = 0;
sumz2 = 0;

sumxy = 0;
sumxz = 0;
sumyz = 0;

signx = 0;
signy = 0;
signz = 0;

deltaV = spacing(1,1)*spacing(2,1)*spacing(3,1);
for iz = 1 : size(V,3)
    for iy = 1 : size(V,2)
        for ix = 1:size(V,1)
            sumo = sumo + V(ix,iy,iz)*deltaV;
            sumx = sumx + (ix-1)*spacing(1,1)*V(ix,iy,iz)*deltaV;
            sumy = sumy + (iy-1)*spacing(2,1)*V(ix,iy,iz)*deltaV;
            sumz = sumz + (iz-1)*spacing(3,1)*V(ix,iy,iz)*deltaV;
            sumx2 = sumx2 + (((ix-1)*spacing(1,1))^2)*V(ix,iy,iz)*deltaV;
            sumy2 = sumy2 + (((iy-1)*spacing(2,1))^2)*V(ix,iy,iz)*deltaV;
            sumz2 = sumz2 + (((iz-1)*spacing(3,1))^2)*V(ix,iy,iz)*deltaV;
            sumxy = sumxy + (ix-1)*spacing(1,1)*(iy-1)*spacing(2,1)*V(ix,iy,iz)*deltaV;
            sumxz = sumxz + (ix-1)*spacing(1,1)*(iz-1)*spacing(3,1)*V(ix,iy,iz)*deltaV;
            sumyz = sumyz + (iy-1)*spacing(2,1)*(iz-1)*spacing(3,1)*V(ix,iy,iz)*deltaV;
        end
    end
end

meanx = sumx/sumo;
meany = sumy/sumo;
meanz = sumz/sumo;

meanx2 = sumx2/sumo;
meany2 = sumy2/sumo;
meanz2 = sumz2/sumo;

meanxy = sumxy/sumo;
meanxz = sumxz/sumo;
meanyz = sumyz/sumo;

