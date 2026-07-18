% DiscrGF computes a free space Discrete Green's function associated  
% to the scalar potential potential created in the point (xc,yc,zc)  
% by a rectangular cell of dimensions a*b, centered at the origin 
% and located in the plane xy, with sides parallel to the x and y axes 
% The computation is done by evaluating the normalized surface integral  
% (giving the mean value of the integrand 1/r, through the use of  
% the primitive function prim1 
 
% DiscrGF is to be used in Point-Matching formulations. If the Mean Theorem 
%is used, DiscrGF can be approximated by the corresponding GF value (=1/r) 
 
% To avoid 0/0 indeterminacies the value zc is replaced by zc+eps with 
% eps=2.2204e-16. This has in negligeable effect in the obtained values  
% A total unit charge is assumed in the cell. Hence, the electrostatic  
% potential is obtained multiplying the result by the factor:  
% (total cell charge)/(4*pi*eps0) 
 
% J. R. Mosig, EPFL Switzerland, 01/06/2023 
 
function [result] = DiscrGF(xc,yc,zc,a,b) 
 
   prim1 = @(u,v,c) u*asinh(v/sqrt(u^2+c^2))+v*asinh(u/sqrt(v^2+c^2))... 
      -c*atan(u*v/c/sqrt(u^2+v^2+c^2)); 
   zc=zc+eps;  
   result=prim1(xc+a/2,yc+b/2,zc)+prim1(xc-a/2,yc-b/2,zc)... 
         -prim1(xc+a/2,yc-b/2,zc)-prim1(xc-a/2,yc+b/2,zc); 
     result=result/(a*b);  
end