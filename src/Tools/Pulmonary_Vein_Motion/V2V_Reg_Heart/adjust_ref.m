function VV=adjust_ref(V,fx,fy,fz,fx1,fy1,fz1,meanx,meany,meanz,bias)

for iz = 1 : size(V,3)
    for iy = 1 : size(V,2)
        for ix = 1:size(V,1)
            %temp = round(diag([fx/fx1,fy/fy1,fz/fz1])*([ix;iy;iz]-[meanx+bias;meany+bias;meanz+bias])+[meanx+bias;meany+bias;meanz+bias]);
            if(fx/fx1<0)
                temp(1,1) = size(V,1) - ix + 1;
            else
                temp(1,1) = ix;
            end

            if(fy/fy1<0)
                temp(2,1) = size(V,2) - iy + 1;
            else
                temp(2,1) = iy;
            end
            
            
            if(fz/fz1<0)
                temp(3,1) = size(V,3) - iz + 1;
            else
                temp(3,1) = iz;
            end
            
            if ( temp(1,1) == 0)
                 temp(1,1) = 1;
             end
             if ( temp(2,1) == 0)
                 temp(2,1) = 1;
             end
             if ( temp(3,1) == 0)
                 temp(3,1) = 1;
             end
            

            VV(temp(1,1),temp(2,1),temp(3,1)) = V(ix,iy,iz);
        end
    end
end