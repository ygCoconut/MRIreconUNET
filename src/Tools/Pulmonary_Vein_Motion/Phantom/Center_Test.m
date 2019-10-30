clc
clear

[Vi bR bT spacing1 info] = get_image(1511,33,'C:\ThirdParty\RawData\Mehdi_Phantom1\',0);
tempM = cross(-bR(4:6),bR(1:3));
Rb = [bR(1:3),bR(4:6),tempM];
CCx1 = -(double(spacing1(1,1)*info.Height)/2);
CCy1 = -(double(spacing1(1,1)*info.Width)/2);
CCz1 = -(info.SpacingBetweenSlices)*(size(Vi,3)-1)/2;
C1 = bT(:,1)-Rb*[CCx1;CCy1;CCz1]

tempM = cross(bR(1:3),bR(4:6));
%Ra = [tempM,aR(4:6),aR(1:3)];
Ra = [bR(1:3),bR(4:6),tempM]
C1 = bT(:,1)-Ra*[CCx1;CCy1;CCz1]


