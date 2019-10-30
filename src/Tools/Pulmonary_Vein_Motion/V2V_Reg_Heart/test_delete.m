clc
clear
%%%%%%%%%%%%%%%%%%%%%%%%Convention is AP FH RL in reading
%%%%%%%%%%%%%%%%%%%%%%%%informations%%%%%%%%%%%%%%%%%%%%%
[V aR aT spacing info AngM OrM] = get_image(557,120,'C:\ThirdParty\RawData\Grimaldi\',0);%7291579
%Ra = rotateM([AngM(1,1),0,0])'* rotateM([0,AngM(3,1),0])'* rotateM([0,0,AngM(2,1)])';
if (info.Private_2001_100b(1)=='C')
    Ra = rotateM([0,0,AngM(1,1)])'* rotateM([0,AngM(3,1),0])'*rotateM([AngM(2,1),0,0]);
    CCx = -(double(spacing(1,1)*info.Height)/2);
    CCy = -(double(spacing(2,1)*info.Width)/2);
    CCz = -(double(spacing(3,1)*(size(V,3)-1)))/2;
    C = [-OrM(2,1);OrM(3,1);OrM(1,1)]-Ra*[0;0;CCz];
elseif(info.Private_2001_100b(1)=='T')
    Ra = rotateM([0,0,AngM(2,1)])'* rotateM([0,AngM(3,1),0])'*rotateM([AngM(1,1),0,0])';
    CCx = -(double(spacing(1,1)*info.Height)/2);
    CCy = -(double(spacing(2,1)*info.Width)/2);
    CCz = -(double(spacing(3,1)*(size(V,3)-1)))/2;
    C = [OrM(1,1);OrM(3,1);OrM(2,1)]-Ra*[0;0;CCz];
else
    disp('Error happened in reading DCM files. The files should be either Coronal or Transversal.');
end
  
%C = [OrM(2,1);OrM(3,1);OrM(1,1)]-Ra*[0;0;CCz];




[Vi bR bT spacing1 info1 AngM1 OrM1] = get_image(763,112/2,'C:\ThirdParty\RawData\Grimaldi\',0);
%Rb = rotateM([AngM1(1,1),0,0])* rotateM([0,AngM1(3,1),0])* rotateM([0,0,AngM1(2,1)]);
if (info1.Private_2001_100b(1)=='C')
    Rb = rotateM([0,0,AngM1(1,1)])'* rotateM([0,AngM1(3,1),0])'*rotateM([AngM1(2,1),0,0]);
    CCx1 = -(double(spacing1(1,1)*info.Height)/2);
    CCy1 = -(double(spacing1(2,1)*info.Width )/2);
    CCz1 = -(double(spacing1(3,1)*(size(Vi,3)-1)))/2;
    C1 = [-OrM1(2,1);OrM1(3,1);OrM1(1,1)]-Rb*[0;0;CCz1];
elseif(info1.Private_2001_100b(1)=='T')
    Rb = rotateM([0,0,AngM1(2,1)])'* rotateM([0,AngM1(3,1),0])'*rotateM([AngM1(1,1),0,0])';
    CCx1 = -(double(spacing1(1,1)*info.Height)/2);
    CCy1 = -(double(spacing1(2,1)*info.Width )/2);
    CCz1 = -(double(spacing1(3,1)*(size(Vi,3)-1)))/2;
    C1 = [OrM1(1,1);OrM1(3,1);OrM1(2,1)]-Rb*[0;0;CCz1];
else
    disp('Error happened in reading DCM files. The files should be either Coronal or Transversal.');
end




No = 0;
VM = zeros(size(V));
for iz = 1:size(V,3)
     for iy = 1 : (sqrt(1)*size(V,2))
         for ix = 1: (sqrt(1)*size(V,1))
             
             temp(1,1) = CCx + (ix-1)*spacing(1,1);
             temp(2,1) = CCy + (iy-1)*spacing(2,1);
             temp(3,1) = CCz + (iz-1)*spacing(3,1);
             
%              tempOne = Ra*temp;
%              TempOneO = [temp(3,1),temp(2,2),-tempOne(1,1)] + [C()];
             
             tempOne = Ra*temp + C;
             
             if (info.Private_2001_100b(1)=='T' & info1.Private_2001_100b(1)=='C')
                 tempOne = [-tempOne(3,1);tempOne(2,1);tempOne(1,1)];
             elseif (info.Private_2001_100b(1)=='C' & info1.Private_2001_100b(1)=='T')
                 tempOne = [tempOne(3,1);tempOne(2,1);-tempOne(1,1)];
             end

         
             tempTwo = Rb'*(tempOne-C1);
             


             temp1(1,1) = round((tempTwo(1,1)-CCx1)/spacing1(1,1))+1;
             temp1(2,1) = round((tempTwo(2,1)-CCy1)/spacing1(2,1))+1;
             temp1(3,1) = round((tempTwo(3,1)-CCz1)/spacing1(3,1))+1;
             
             %temp1
                    
           if ((temp1(1,1)>0 && temp1(1,1)<=size(Vi,1)) && (temp1(2,1)>0&& temp1(2,1)<=size(Vi,2)) && (temp1(3,1)>0&& temp1(3,1)<=size(Vi,3)))
       
             %VV(temp1(1,1),temp1(2,1),temp1(3,1)) = V(ix,iy,iz);
             VM(ix,iy,iz) = Vi(temp1(1,1),temp1(2,1),temp1(3,1));
             end
             

         end
     end
     iz
end



