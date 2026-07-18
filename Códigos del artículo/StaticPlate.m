% StaticPlate 
% Study of the charge distribution in a metallic rectangular plate. 
 
% The rectangle of sides (aside,bside) is divided into MxN square cells. 
% Hence, there will be MxN cells with unknown charge values.  
% Therefore, the MoM matrix will have (M*N)^2 elements to be computed and  
% stored, since no symmetries are considered here. 
 
% Both Galerkin and PM implementations are included. 
% They can be called with the MatLab functions StatMoM. 
 
% J.R. Mosig, EPFL Switzerland, June 2023 
 
clear all 
% Rectangular plate data: dimensions (meters) of the plate 
aside=1 
bside=1 
% The plate is divided into MxN cells 
M=20 
N=20             
a=aside/M;    
b=bside/N;      %  a,b are the cells' dimensions 
 
% 1) A SIMPLE VERSION OF WHAT A GOOD MESHER SOFTWARE SHOULD PROVIDE 
% Coordinates (x,y)of all cell centers 
% First the x,y coordinates are created as square matrices  
% xmat, ymat, zmat since this is natural for a 2D plate. 
x=linspace(a/2,aside-a/2,M); 
y=linspace(b/2,bside-b/2,N); 
[xmat,ymat]=meshgrid(x,y); 
 
% But then,the coordinates are reformatted as 1D-tables (vectors) since 
% this is required by the MoM method. 
xvec=reshape(xmat,1,M*N); 
yvec=reshape(ymat,1,M*N); 
 
% 2) FILLING THE MoM MATRIX 
% Here the Green's matrix is filled in an old-fashioned way using a double FOR loop.  
% More elegant strategies using the matrix/array operations typical of MatLab  
% could be implemented. 
% For the MoM matrix elements Galerkin, PM or plain GF values are selected, 
% depending on the values of "tresh" and "flagG" in the call to StatMoM 
 
NDIM=M*N       % NDIM is the order of the matrix 
tresh=10 
flagG=0 
mom=zeros(NDIM,NDIM); 
for i=1:NDIM 
  for j=1:NDIM 
     mom(i,j)=StatMoM(xvec(j),yvec(j),0,xvec(i),yvec(i),0,a,b,tresh,flagG);    
  end 
end 
 
% 3) COMPUTING THE EXCITATION VECTOR 
% Scenario 1):  A constant normalized potential Unorm is imposed on the 
% metallic plate. This corresponds to a constant excitation vector "excv" 
Unorm=1; 
excv=Unorm*ones(NDIM,1); 
 
% Scenario 2):  A point unit charge is placed at point (xq,yq,zq) 
%xq=0.5 
%yq=0.5 
%zq=0.5 
%excv=zeros(NDIM,1); 
%for i=1:NDIM 
%    excv(i)=1/sqrt((xq-xvec(i))^2+(yq-yvec(i))^2+zq^2); 
%end 
 
% 4) SOLVING THE LINEAR SYSTEM AND COMPUTING CHARGES AND THE CAPACITANCE 
% The solution of the linear system is the array "charge" which contains  
% the values of the total charges in each cell (not the charge density!!). 
charge=mom\excv; 
charge_tot=sum(charge,'all')   % charge_tot is the total charge in the plate 
% And in the excitation scenario 1), the normalized capacity is simply: 
Capnorm=charge_tot/Unorm 
% This value must be multiplied by 4*pi*epsilon0 to obtain the true capacity in farads