function R = rotateM(Rot)
Rot = (Rot *  pi /180);
    
%R =[1 0 0; 0 cos(Rot(1)) -sin(Rot(1)); 0 sin(Rot(1)) cos(Rot(1))]*[cos(Rot(2)) 0 sin(Rot(2)); 0 1 0; -sin(Rot(2)) 0 cos(Rot(2))] * [cos(Rot(3)) -sin(Rot(3)) 0; sin(Rot(3)) cos(Rot(3)) 0; 0 0 1] ;
R =[cos(Rot(3)) -sin(Rot(3)) 0; sin(Rot(3)) cos(Rot(3)) 0; 0 0 1]*[cos(Rot(2)) 0 sin(Rot(2)); 0 1 0; -sin(Rot(2)) 0 cos(Rot(2))]*[1 0 0; 0 cos(Rot(1)) -sin(Rot(1)); 0 sin(Rot(1)) cos(Rot(1))];

    