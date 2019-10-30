function [signx,signy,signz]=getsign(V,meanx,meany,meanz,spacing);

signx = 0;
signy = 0;
signz = 0;

Delta_V = spacing(1,1)*spacing(2,1)*spacing(3,1);

for iz = 1 : size(V,3)
    for iy = 1 : size(V,2)
        for ix = 1:size(V,1)
            
            ixp = (ix-1)*spacing(1,1);
            iyp = (iy-1)*spacing(2,1);
            izp = (iz-1)*spacing(3,1);
            
            signx = signx + sign(ixp-meanx)*(ixp-meanx)*(ixp-meanx)*V(ix,iy,iz)*Delta_V;
            signy = signy + sign(iyp-meany)*(iyp-meany)*(iyp-meany)*V(ix,iy,iz)*Delta_V;
            signz = signz + sign(izp-meanz)*(izp-meanz)*(izp-meanz)*V(ix,iy,iz)*Delta_V;
        end
    end
end
signx =sign(signx);
signy =sign(signy);
signz =sign(signz);
return
